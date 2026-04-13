import argparse
import json
import os
from typing import Any, Dict

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from src.data_pipeline.build_background_embeddings import (
    build_background_prompt,
    generate_background_and_hidden_states,
)
from src.eval.oracle_gag.utils import llm_for_open_generation
from src.language_modeling.memory_utils import (
    build_memory_slots_from_layer_dict,
    normalize_layer_keys,
)
from src.language_modeling.preprocessing import _concat_messages_qwen3, build_gag_prompt
from src.language_modeling.utils import (
    GAG_TOKEN,
    get_yaml_file,
    load_tokenizer_with_fast_fallback,
    resolve_path,
)
from src.model import XQwen3ForCausalLM
from src.ppr.prototype_router import PrototypeRouter


DEFAULT_GENERAL_INSTRUCTION = "You are a helpful assistant."
DEFAULT_GENERAL_REQUEST = "Please answer the following question."
DEFAULT_MATERIALS_BG_SYSTEM = (
    "You are an expert in materials science and engineering. Your research and practice have equipped you "
    "with a deep understanding of how composition, structure, processing and environment determine material "
    "properties and performance. You excel in providing relevant and professional background knowledge that "
    "can help answer research-level questions in this field. Please provide the background knowledge related "
    "to the following question. Do not fabricate specific numerical data that is not generally known in the field."
)
DEFAULT_ADJUVANT_BG_SYSTEM = (
    "You are an expert in immunology and adjuvant, with a strong background in vaccine development. "
    "Your research and practice in this field have equipped you with a deep understanding of the mechanisms "
    "of immune response and how to optimize vaccine efficacy through adjuvants. You excel in providing "
    "relevant and professional background knowledge that can help answer the question. Please provide the "
    "background knowledge related to the following question."
)


def resolve_model_class(config):
    architecture = None
    if getattr(config, "architectures", None):
        architecture = config.architectures[0]
    if architecture == "XQwen3ForCausalLM":
        return XQwen3ForCausalLM
    return AutoModelForCausalLM


def canonicalize_domain(domain_name: str) -> str:
    if domain_name == "material":
        return "materials"
    return domain_name


def parse_args():
    parser = argparse.ArgumentParser(description="Mixed-domain online inference with Prototype-based Plug-and-play Routing.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_file_path", type=str, default=None)
    args = parser.parse_args()

    config = get_yaml_file(args.config)
    for key, value in config.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    args.workdir = resolve_path(None, args.workdir or os.path.dirname(os.path.abspath(args.config)))
    args.data_path = resolve_path(args.workdir, args.data_path)
    args.output_file_path = resolve_path(args.workdir, args.output_file_path)

    args.prototype_files = {
        canonicalize_domain(domain_name): resolve_path(args.workdir, path)
        for domain_name, path in args.prototype_files.items()
    }
    for section_name in ["general", "adjuvant", "materials"]:
        if getattr(args, section_name, None):
            section = getattr(args, section_name)
            for key in ["model_name_or_path", "small_model_name_or_path", "big_model_name_or_path"]:
                if key in section:
                    section[key] = resolve_path(args.workdir, section[key])
            setattr(args, section_name, section)

    return args


def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if not data:
        raise RuntimeError(f"No data loaded from {path}")
    return data


def save_jsonl(records, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_qwen3_generation_prompt(tokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    return _concat_messages_qwen3(messages, tokenizer) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def build_general_prompt(question: str, instruction_text: str, request_text: str) -> str:
    lines = []
    if instruction_text.strip():
        lines.append(instruction_text.strip())
    if request_text.strip():
        lines.append(request_text.strip())
    lines.append(f"Question: {question.strip()}")
    return "\n".join(lines) + "\n"


def load_big_model_and_tokenizer(model_name_or_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
        add_eos_token=False,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model_class = resolve_model_class(config)
    model = model_class.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    if hasattr(model, "set_gag_token_id") and GAG_TOKEN in tokenizer.get_vocab():
        model.set_gag_token_id(tokenizer.convert_tokens_to_ids(GAG_TOKEN))
    return model, tokenizer


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    generation_cfg: Dict[str, Any],
    retrieval_embeds: torch.Tensor | None = None,
):
    retrieval_list = None
    if retrieval_embeds is not None:
        retrieval_list = [retrieval_embeds]
    answers = llm_for_open_generation(
        llm=model,
        llm_tokenizer=tokenizer,
        prompts=[prompt],
        retrieval_embeds=retrieval_list,
        batch_size=1,
        enable_progress_bar=False,
        do_sample=bool(generation_cfg.get("do_sample", False)),
        max_new_tokens=int(generation_cfg.get("max_new_tokens", 512)),
        temperature=float(generation_cfg.get("temperature", 0.7)),
        top_p=float(generation_cfg.get("top_p", 0.8)),
        top_k=int(generation_cfg.get("top_k", 20)),
        repetition_penalty=float(generation_cfg.get("repetition_penalty", 1.05)),
        no_repeat_ngram_size=int(generation_cfg.get("no_repeat_ngram_size", 3)),
    )
    return answers[0]


def main():
    args = parse_args()
    data = load_jsonl(args.data_path)

    router = PrototypeRouter(
        encoder_name_or_path=args.router_encoder_name_or_path,
        prototype_files=args.prototype_files,
        device=args.router_device,
        max_seq_length=int(args.router_max_seq_length),
    )

    big_model_cache: Dict[str, Any] = {}
    small_model_cache: Dict[str, Any] = {}

    def get_big_model(model_path: str, device_str: str):
        cache_key = f"{model_path}::{device_str}"
        if cache_key not in big_model_cache:
            device = torch.device(device_str if torch.cuda.is_available() else "cpu")
            big_model_cache[cache_key] = load_big_model_and_tokenizer(model_path, device)
        return big_model_cache[cache_key]

    def get_small_model(model_path: str, device_str: str):
        cache_key = f"{model_path}::{device_str}"
        if cache_key not in small_model_cache:
            device = torch.device(device_str if torch.cuda.is_available() else "cpu")
            tokenizer = load_tokenizer_with_fast_fallback(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            ).to(device)
            model.eval()
            small_model_cache[cache_key] = (model, tokenizer, device)
        return small_model_cache[cache_key]

    results = []
    for item in tqdm(data, ncols=100, desc="PPR routed eval"):
        question = (item.get("question") or item.get("input") or "").strip()
        if not question:
            continue

        routed_domain, router_score, router_scores = router.route(question, return_scores=True)
        routed_domain = canonicalize_domain(routed_domain)

        if routed_domain == "general":
            general_cfg = args.general
            general_model, general_tokenizer = get_big_model(
                general_cfg["model_name_or_path"],
                general_cfg.get("device", "cuda:0"),
            )
            general_user_prompt = build_general_prompt(
                question=question,
                instruction_text=general_cfg.get("instruction_text", DEFAULT_GENERAL_INSTRUCTION),
                request_text=general_cfg.get("request_text", DEFAULT_GENERAL_REQUEST),
            )
            general_prompt = build_qwen3_generation_prompt(general_tokenizer, general_user_prompt)
            answer = generate_answer(general_model, general_tokenizer, general_prompt, general_cfg)
            background_text = ""
        else:
            domain_cfg = getattr(args, routed_domain)
            small_model, small_tokenizer, small_device = get_small_model(
                domain_cfg["small_model_name_or_path"],
                domain_cfg.get("small_device", "cuda:0"),
            )
            big_model, big_tokenizer = get_big_model(
                domain_cfg["big_model_name_or_path"],
                domain_cfg.get("big_device", "cuda:0"),
            )

            background_prompt = build_background_prompt(
                question=question,
                tokenizer=small_tokenizer,
                system_prompt=domain_cfg.get(
                    "background_system_prompt",
                    DEFAULT_MATERIALS_BG_SYSTEM if routed_domain == "materials" else DEFAULT_ADJUVANT_BG_SYSTEM,
                ),
            )
            background_text, _, hidden_states = generate_background_and_hidden_states(
                model=small_model,
                tokenizer=small_tokenizer,
                prompt=background_prompt,
                layer_keys=normalize_layer_keys(domain_cfg.get("layer_keys")),
                max_new_tokens=int(domain_cfg.get("max_new_tokens_small", 640)),
                do_sample=bool(domain_cfg.get("do_sample_small", False)),
                temperature=float(domain_cfg.get("temperature_small", 0.7)),
                top_p=float(domain_cfg.get("top_p_small", 0.8)),
                top_k=int(domain_cfg.get("top_k_small", 20)),
                device=small_device,
            )
            memory_slots = build_memory_slots_from_layer_dict(
                record=hidden_states,
                layer_keys=domain_cfg.get("layer_keys"),
                num_memory_slots=int(domain_cfg.get("num_memory_slots", 4)),
                pooling=domain_cfg.get("slot_pooling", "segment_softmax"),
                temperature=float(domain_cfg.get("slot_pooling_temperature", 1.0)),
            )
            prompt = build_gag_prompt(
                instruction_text=domain_cfg["instruction_text"],
                query=question,
                num_memory_slots=int(domain_cfg.get("num_memory_slots", 4)),
                request_text=domain_cfg.get("request_text"),
            )
            user_prompt = build_qwen3_generation_prompt(big_tokenizer, prompt)
            retrieval_embeds = memory_slots.permute(1, 0, 2).contiguous()
            answer = generate_answer(big_model, big_tokenizer, user_prompt, {
                "do_sample": bool(domain_cfg.get("do_sample_big", False)),
                "max_new_tokens": int(domain_cfg.get("max_new_tokens_big", 512)),
                "temperature": float(domain_cfg.get("temperature_big", 0.7)),
                "top_p": float(domain_cfg.get("top_p_big", 0.8)),
                "top_k": int(domain_cfg.get("top_k_big", 20)),
                "repetition_penalty": float(domain_cfg.get("repetition_penalty", 1.05)),
                "no_repeat_ngram_size": int(domain_cfg.get("no_repeat_ngram_size", 3)),
            }, retrieval_embeds=retrieval_embeds)

        result = dict(item)
        result["router_domain"] = routed_domain
        result["router_score"] = float(router_score)
        result["router_all_scores"] = {k: float(v) for k, v in router_scores.items()}
        result["qwen3-1.7B_answer_background"] = background_text
        result["Qwen3-8B_answer"] = answer
        results.append(result)

    save_jsonl(results, args.output_file_path)
    print(f"[Done] Saved routed evaluation results to {args.output_file_path}")


if __name__ == "__main__":
    main()
