# src/eval/run_eval_with_plug_router.py
# -*- coding: utf-8 -*-
"""
Multi-domain eval with Prototype-based Plug-and-play Router (PPR)

- 输入：混合数据集 JSONL，字段至少包含 "id", "question"
- 路由：PrototypeRouter(general, adjuvant, material)
- general
- adjuvant 
- material 

"""

import os
import json
import yaml
import argparse
from typing import List, Dict, Any, Tuple, Optional

import torch
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)

from src.model import XQwen3ForCausalLM
from src.ppr.prototype_router import PrototypeRouter
from src.language_modeling.utils import GAG_TOKEN


# ======================= 一些通用小工具 =======================

def print_param_device_summary(model):
    from collections import Counter
    c = Counter()
    for _, p in model.named_parameters():
        c[str(p.device)] += p.numel()
    print("[Param devices]", dict(c))


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            data.append(obj)
    if len(data) == 0:
        raise RuntimeError(f"No data loaded from: {path}")
    return data


def save_jsonl(records: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] Results written to: {path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True,
                    help="YAML config for multi-domain eval with prototype router.")
    return ap.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ======================= Qwen3 Chat Template =======================

def apply_qwen3_chat_template(tokenizer, user_content: str) -> str:
    """
    与你之前的一致：
      messages=[{"role":"user","content": user_content}]
      再在末尾拼上 assistant 起始（包含 <think> 空段）
    """
    messages = [{"role": "user", "content": user_content}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    user_prompt = text + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return user_prompt


# ======================= 小模型：背景知识 Prompt =======================

# ---- 佐剂小模型（background）system prompt，来自你之前的代码 ----
ADJ_BG_SYSTEM = (
    "You are an expert in immunology and adjuvant, with a strong background in vaccine development. "
    "Your research and practice in this field have equipped you with a deep understanding of the mechanisms of immune response "
    "and how to optimize vaccine efficacy through adjuvants. You excel in providing relevant and professional background knowledge "
    "that can help answer the question. Please provide the background knowledge related to the following question.\n"
)

# ---- 材料小模型（background）system prompt，来自你给我的脚本 ----
MAT_BG_SYSTEM = (
    "You are an expert in materials science and engineering. Your research and practice have equipped you "
    "with a deep understanding of how composition, structure, processing and environment determine material "
    "properties and performance. You excel in providing relevant and professional background knowledge that "
    "can help answer research-level questions in this field. Please provide the background knowledge related "
    "to the following question. Do not fabricate specific numerical data that is not generally known in the field."
)


def build_small_model_prompt(domain: str, tokenizer_sm, question: str) -> str:
    """
    根据领域构造小模型的 Chat Prompt，用于生成背景知识 answer。
    返回已经 apply_chat_template 的“完整对话文本字符串”。
    """
    if domain == "adjuvant":
        system_text = ADJ_BG_SYSTEM
    elif domain == "material":
        system_text = MAT_BG_SYSTEM
    else:
        raise ValueError(f"Unsupported domain for small model prompt: {domain}")

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": f"{question}\n"},
    ]
    prompt = tokenizer_sm.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return prompt




@torch.no_grad()
def get_background_text_and_embeddings(
    domain: str,
    model_sm,
    tokenizer_sm,
    query: str,
    device: torch.device,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    embed_key: str = "last_layer_answer_tokens_embedding",   # ★新增：与 oracle_gag 对齐
) -> Tuple[str, torch.Tensor]:
    """
    模仿 oracle_gag:
    1) 用 generate_to_get_answer_embedding_different_layers 得到 5 个层的 [1,L,2048]
    2) 通过 embed_key 选择层
    3) 只保留最后一个 token embedding -> [1,1,2048]
    """
    prompt = build_small_model_prompt(domain, tokenizer_sm, query)
    inputs = tokenizer_sm(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    input_len = inputs["input_ids"].shape[1]

    # ★关键：必须用 different_layers（否则无法选择倒数第几层）
    if not hasattr(model_sm, "generate_to_get_answer_embedding_different_layers"):
        raise AttributeError(
            "model_sm has no method generate_to_get_answer_embedding_different_layers(). "
            "Your oracle_gag pipeline relies on this; please ensure the small model is patched the same way."
        )

    outputs = model_sm.generate_to_get_answer_embedding_different_layers(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_hidden_states=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        use_cache=False,
    )

    # 解码背景文本
    background_text = tokenizer_sm.decode(
        outputs.sequences[0][input_len:], skip_special_tokens=True
    ).strip()

    # ---- 完全照 oracle_gag：收集 5 个层的序列 ----
    step_tensors_last = []
    step_tensors_m2 = []
    step_tensors_m4 = []
    step_tensors_m6 = []
    step_tensors_m8 = []

    for step_item in outputs.hidden_states:
        h_last, h_m2, h_m4, h_m6, h_m8 = step_item  # 每个都是 [1,1,2048]
        step_tensors_last.append(h_last)
        step_tensors_m2.append(h_m2)
        step_tensors_m4.append(h_m4)
        step_tensors_m6.append(h_m6)
        step_tensors_m8.append(h_m8)

    last_layer_answer_tokens_hidden = torch.cat(step_tensors_last, dim=1).cpu()  # [1,L,2048]
    minus2_layer_answer_tokens_hidden = torch.cat(step_tensors_m2, dim=1).cpu()
    minus4_layer_answer_tokens_hidden = torch.cat(step_tensors_m4, dim=1).cpu()
    minus6_layer_answer_tokens_hidden = torch.cat(step_tensors_m6, dim=1).cpu()
    minus8_layer_answer_tokens_hidden = torch.cat(step_tensors_m8, dim=1).cpu()

    layer_dict = {
        "last_layer_answer_tokens_embedding": last_layer_answer_tokens_hidden,
        "layer_minus2_answer_tokens_embedding": minus2_layer_answer_tokens_hidden,
        "layer_minus4_answer_tokens_embedding": minus4_layer_answer_tokens_hidden,
        "layer_minus6_answer_tokens_embedding": minus6_layer_answer_tokens_hidden,
        "layer_minus8_answer_tokens_embedding": minus8_layer_answer_tokens_hidden,
    }
    if embed_key not in layer_dict:
        raise ValueError(f"embed_key={embed_key} not in {list(layer_dict.keys())}")

    answer_tokens_hidden = layer_dict[embed_key]  # [1,L,2048]

    # ★完全照 oracle_gag preprocessing：只保留最后一个 token embedding
    answer_tokens_hidden = answer_tokens_hidden[:, -1:, :].contiguous()  # [1,1,2048]

    # 强校验 prefix
    full_ids = outputs.sequences[0]
    assert torch.equal(full_ids[:input_len], inputs["input_ids"][0]), \
        "prefix mismatch between inputs['input_ids'] and outputs.sequences prefix!"

    return background_text, answer_tokens_hidden



# ======================= 大模型：GAG Answer Prompt =======================

# ---- 佐剂大模型 GAG prompt instruction（你 run_eval_with_gate.py 里的 INSTRUCTION_PRO）----
ADJ_ANSWER_INSTRUCTION = (
    "You are an expert in immunology and adjuvant, with a strong background in vaccine development. "
    "Your research and practice in this field have equipped you with a deep understanding of the mechanisms of immune response "
    "and how to optimize vaccine efficacy through adjuvants. You excel in providing concise, precise, and professional responses "
    "to questions related to adjuvants and immunology.\n"
)

# ---- 材料大模型 GAG prompt instruction（你给的 prepare_prompts 里的 instruction）----
MAT_ANSWER_INSTRUCTION = (
    "You are an expert in materials science and engineering, with extensive experience in the design, "
    "synthesis, and characterization of functional materials. Your research and practice have equipped you "
    "with a deep understanding of structure–property relationships, reaction mechanisms, and performance "
    "optimization strategies. You excel in providing concise, precise, and professional responses to questions "
    "related to materials design, processing, and applications.\n"
)


def build_user_prompt_general(query: str) -> str:
    # 精简通用 prompt：你之前 general 通路用的那种
    return f"Please answer the following question.\nQuestion: {query}\n"


def build_user_prompt_with_gag(domain: str, query: str, gag_len: int) -> str:
    """
    构造带 GAG 占位符的 user_content（后续再用 apply_qwen3_chat_template 包成完整 prompt）。
    """
    if domain == "adjuvant":
        instruction = ADJ_ANSWER_INSTRUCTION
    elif domain == "material":
        instruction = MAT_ANSWER_INSTRUCTION
    else:
        raise ValueError(f"Unsupported domain for GAG prompt: {domain}")

    gag_tokens = "".join([GAG_TOKEN] * gag_len)
    prompt = (
        instruction
        + "Please answer the following question based on the knowledge provided.\n"
        + f"Question: {query}\n"
        + f"Knowledge: {gag_tokens}\n "
    )
    return prompt


# ======================= Big model load & per-domain pipeline =======================

def load_big_model_and_tokenizer(model_path: str, device: torch.device):
    """
    加载 8B 基座（可能带 Projector），自动识别 XQwen3ForCausalLM。
    """
    print(f"[Load] Big model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        add_eos_token=False,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(model_path)
    # 可选：支持 FA2
    try:
        config.attn_implementation = "flash_attention_2"
    except Exception:
        pass

    arch = ""
    if getattr(config, "architectures", None):
        arch = config.architectures[0]

    if "XQwen3ForCausalLM" in arch:
        ModelClass = XQwen3ForCausalLM
    else:
        ModelClass = AutoModelForCausalLM

    model = ModelClass.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        # device_map={"": 0},        # 整模放到 cuda:0
        # offload_state_dict=False,
        trust_remote_code=True,
    )
    model.to(device)
    print_param_device_summary(model)

    # 如果有 GAG_TOKEN，则设定 gag_token_id
    if hasattr(model, "set_gag_token_id") and (GAG_TOKEN in tokenizer.get_vocab()):
        model.set_gag_token_id(tokenizer.convert_tokens_to_ids(GAG_TOKEN))

    model.eval()
    return model, tokenizer


@torch.no_grad()
def run_general_pipeline(
    question: str,
    model_bg,
    tokenizer_bg,
    gen_cfg: Dict[str, Any],
) -> str:
    user_content = build_user_prompt_general(question)
    user_prompt = apply_qwen3_chat_template(tokenizer_bg, user_content)

    tok = tokenizer_bg(user_prompt, padding=False, return_tensors="pt")
    device = next(model_bg.parameters()).device
    input_ids = tok.input_ids.to(device)
    attention_mask = tok.attention_mask.to(device)

    outputs = model_bg.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=bool(gen_cfg.get("do_sample", True)),
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 1000)),
        pad_token_id=tokenizer_bg.pad_token_id,
        use_cache=True,
        temperature=float(gen_cfg.get("temperature", 0.7)),
        top_p=float(gen_cfg.get("top_p", 0.8)),
        top_k=int(gen_cfg.get("top_k", 20)),
    )
    input_len = input_ids.shape[1]
    ans = tokenizer_bg.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0].strip()
    return ans


@torch.no_grad()
def run_gag_pipeline(
    domain: str,
    question: str,
    model_sm,
    tokenizer_sm,
    model_bg,
    tokenizer_bg,
    small_gen_cfg: Dict[str, Any],
    big_gen_cfg: Dict[str, Any],
) -> str:
    """
    GAG 领域通路：小模型生成背景 → Projector 注入 → 8B 回答
    """
    # 小模型所在 device
    device_sm = next(model_sm.parameters()).device

    bg_text, answer_tokens_hidden = get_background_text_and_embeddings(
        domain=domain,
        model_sm=model_sm,
        tokenizer_sm=tokenizer_sm,
        query=question,
        device=device_sm,
        max_new_tokens=int(small_gen_cfg.get("max_new_tokens_small", 1024)),
        temperature=float(small_gen_cfg.get("temperature_small", 0.7)),
        top_p=float(small_gen_cfg.get("top_p_small", 0.8)),
        top_k=int(small_gen_cfg.get("top_k_small", 20)),
        embed_key=str(small_gen_cfg.get("embed_key", "last_layer_answer_tokens_embedding")),  # ★新增
    )
    L = int(answer_tokens_hidden.size(1))

    # 构造带 GAG 的 user prompt
    user_content = build_user_prompt_with_gag(domain, question, gag_len=L)
    user_prompt = apply_qwen3_chat_template(tokenizer_bg, user_content)

    tok = tokenizer_bg(user_prompt, padding=False, return_tensors="pt")
    device_bg = next(model_bg.parameters()).device
    input_ids = tok.input_ids.to(device_bg)
    attention_mask = tok.attention_mask.to(device_bg)

    retrieval_embeds = answer_tokens_hidden.squeeze(0).to(device_bg)  # [L, H_sm]

    # print("DEBUG embed_key:", small_gen_cfg.get("embed_key"))
    # print("DEBUG answer_tokens_hidden.shape:", tuple(answer_tokens_hidden.shape))
    # print("DEBUG retrieval_embeds.shape:", tuple(retrieval_embeds.shape))


    outputs = model_bg.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=bool(big_gen_cfg.get("do_sample_big", True)),
        max_new_tokens=int(big_gen_cfg.get("max_new_tokens_big", 3000)),
        pad_token_id=tokenizer_bg.pad_token_id,
        use_cache=True,
        temperature=float(big_gen_cfg.get("temperature_big", 0.7)),
        top_p=float(big_gen_cfg.get("top_p_big", 0.8)),
        top_k=int(big_gen_cfg.get("top_k_big", 20)),
        retrieval_embeds=retrieval_embeds,  # 关键：走 Projector 替换 GAG
        no_repeat_ngram_size=int(big_gen_cfg.get("no_repeat_ngram_size", 3)),
        repetition_penalty=float(big_gen_cfg.get("repetition_penalty", 1.05)),
    )
    # 走 inputs_embeds 通路时，你之前已经用过 input_len=0 的解码方式
    input_len = 0
    ans = tokenizer_bg.batch_decode(outputs[:, input_len:], skip_special_tokens=True)[0].strip()
    return ans


# ======================= 主流程：多领域 + 路由 =======================

def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    data_path = cfg["data_path"]
    output_file_path = cfg["output_file_path"]

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # 始终用当前进程可见的第一个 GPU（逻辑 0）
        general_device = torch.device(cfg.get("general_device", "cuda:0"))
        adjuvant_device = torch.device(cfg.get("adjuvant_device", "cuda:0"))
        material_device = torch.device(cfg.get("material_device", "cuda:0"))
    else:
        general_device = adjuvant_device = material_device = torch.device("cpu")

    # ---------- 1) Prototype Router ----------
    router_encoder = cfg["router_encoder_name_or_path"]
    router_max_len = int(cfg.get("router_max_seq_length", 256))
    prototype_files = cfg["prototype_files"]  # dict: {domain_name: proto_path}

    # 新增：单独指定 router 跑在哪个 device，默认 'cpu' 防止 OOM
    router_device = cfg.get("router_device", "cpu")  # "cpu" 或 "cuda" 或 "cuda:1" 等

    print("[Load] PrototypeRouter")
    router = PrototypeRouter(
        encoder_name_or_path=router_encoder,
        prototype_files=prototype_files,
        device=router_device,
        max_seq_length=router_max_len,
    )

    # 标记有哪些领域在用 router（决定要不要加载对应的小模型）
    has_adjuvant = "adjuvant" in prototype_files
    has_material = "material" in prototype_files

    # ---------- 2) 各领域模型 ----------

    big_model_cache = {}  # {model_path: (model, tokenizer)}

    def load_or_share_big_model(path: str, device: torch.device):
        if path in big_model_cache:
            return big_model_cache[path]
        model, tokenizer = load_big_model_and_tokenizer(path, device)
        big_model_cache[path] = (model, tokenizer)
        return model, tokenizer

    # ---- GENERAL ----
    general_cfg = cfg.get("general", {})
    general_big_model_path = general_cfg["big_model_path"]
    general_model_bg, general_tokenizer_bg = load_or_share_big_model(
        general_big_model_path, general_device
    )

    # ---- ADJUVANT ----
    ad_model_sm = ad_tokenizer_sm = ad_model_bg = ad_tokenizer_bg = None
    ad_cfg = None
    if has_adjuvant:
        ad_cfg = cfg.get("adjuvant", {})
        ad_small_model_path = ad_cfg["small_model_path"]

        print("[Load] Adjuvant small model:", ad_small_model_path)
        ad_tokenizer_sm = AutoTokenizer.from_pretrained(
            ad_small_model_path, trust_remote_code=True
        )
        ad_model_sm = AutoModelForCausalLM.from_pretrained(
            ad_small_model_path,
            trust_remote_code=True,
        ).to(adjuvant_device)
        ad_model_sm.eval()

        ad_big_model_path = ad_cfg["big_model_path"]
        ad_model_bg, ad_tokenizer_bg = load_or_share_big_model(ad_big_model_path, adjuvant_device)

    # # ---- ADJUVANT ----
    # ad_cfg = cfg.get("adjuvant", {})
    # ad_small_model_path = ad_cfg["small_model_path"]
    # ad_big_model_path = ad_cfg["big_model_path"]

    # print("[Load] Adjuvant small model:", ad_small_model_path)
    # ad_tokenizer_sm = AutoTokenizer.from_pretrained(ad_small_model_path, trust_remote_code=True)
    # ad_model_sm = AutoModelForCausalLM.from_pretrained(
    #     ad_small_model_path,
    #     trust_remote_code=True,
    # ).to(device)
    # ad_model_sm.eval()

    # ad_model_bg, ad_tokenizer_bg = load_big_model_and_tokenizer(
    #     ad_big_model_path, device
    # )

    # ---- MATERIAL ----
    mat_model_sm = mat_tokenizer_sm = mat_model_bg = mat_tokenizer_bg = None
    mat_cfg = None
    if has_material:
        mat_cfg = cfg.get("material", {})
        mat_small_model_path = mat_cfg["small_model_path"]

        print("[Load] Material small model:", mat_small_model_path)
        mat_tokenizer_sm = AutoTokenizer.from_pretrained(
            mat_small_model_path, trust_remote_code=True
        )
        mat_model_sm = AutoModelForCausalLM.from_pretrained(
            mat_small_model_path,
            trust_remote_code=True,
        ).to(material_device)
        mat_model_sm.eval()

        mat_big_model_path = mat_cfg["big_model_path"]
        mat_model_bg, mat_tokenizer_bg = load_or_share_big_model(mat_big_model_path, material_device)

    # # ---- MATERIAL ----
    # mat_cfg = cfg.get("material", {})
    # mat_small_model_path = mat_cfg["small_model_path"]
    # mat_big_model_path = mat_cfg["big_model_path"]

    # print("[Load] Material small model:", mat_small_model_path)
    # mat_tokenizer_sm = AutoTokenizer.from_pretrained(mat_small_model_path, trust_remote_code=True)
    # mat_model_sm = AutoModelForCausalLM.from_pretrained(
    #     mat_small_model_path,
    #     trust_remote_code=True,
    # ).to(device)
    # mat_model_sm.eval()

    # mat_model_bg, mat_tokenizer_bg = load_big_model_and_tokenizer(
    #     mat_big_model_path, device
    # )

    # ---------- 3) 数据 ----------
    test_data = load_jsonl(data_path)
    print(f"[Data] Loaded {len(test_data)} samples from {data_path}")

    results = []
    for item in tqdm(test_data, ncols=100, desc="Multi-domain Eval (PPR router)"):
        q = (
            item.get("question", "")
            or item.get("input", "")
        ).strip()
        if q == "":
            out = dict(item)
            out["Qwen3_answer"] = ""
            out["router_domain"] = "none"
            out["router_score"] = 0.0
            results.append(out)
            continue

        # ---------- 路由 ----------
        domain, score, score_dict = router.route(q, return_scores=True)
        # 如需 margin 版本，可以改成：
        # domain, score, score_dict = router.route_with_margin(
        #     q, general_domain_name="general", margin=0.05, return_scores=True
        # )

        # ---------- 按领域走不同 pipeline ----------
        if domain == "general":
            ans = run_general_pipeline(
                question=q,
                model_bg=general_model_bg,
                tokenizer_bg=general_tokenizer_bg,
                gen_cfg=general_cfg,
            )
        elif domain == "adjuvant" and has_adjuvant:
            ans = run_gag_pipeline(
                domain="adjuvant",
                question=q,
                model_sm=ad_model_sm,
                tokenizer_sm=ad_tokenizer_sm,
                model_bg=ad_model_bg,
                tokenizer_bg=ad_tokenizer_bg,
                small_gen_cfg=ad_cfg,
                big_gen_cfg=ad_cfg,
            )
        elif domain == "material" and has_material:
            ans = run_gag_pipeline(
                domain="material",
                question=q,
                model_sm=mat_model_sm,
                tokenizer_sm=mat_tokenizer_sm,
                model_bg=mat_model_bg,
                tokenizer_bg=mat_tokenizer_bg,
                small_gen_cfg=mat_cfg,
                big_gen_cfg=mat_cfg,
            )
        else:
            # 理论上 router 不会给出一个没在 prototype_files 里的 domain；
            # 这里留作安全兜底：全都回退到 general。
            print(f"[WARN] domain='{domain}' not configured, fallback to general.")
            ans = run_general_pipeline(
                question=q,
                model_bg=general_model_bg,
                tokenizer_bg=general_tokenizer_bg,
                gen_cfg=general_cfg,
            )

        out = dict(item)
        out["Qwen3_answer"] = ans
        out["router_domain"] = domain
        out["router_score"] = float(score)
        out["router_all_scores"] = {k: float(v) for k, v in (score_dict or {}).items()}
        results.append(out)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_jsonl(results, output_file_path)
    print("[Done] Multi-domain eval with Prototype Router.")


if __name__ == "__main__":
    main()


# python -m src.eval.router.run_eval_with_plug_router \
# --config projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/src/config/eval/run_eval_with_plug_router.yaml