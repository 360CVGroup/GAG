import argparse
import os

from transformers import AutoConfig, AutoTokenizer

from src.eval.oracle_gag.preprocessing import load_data, prepare_prompts
from src.eval.oracle_gag.utils import llm_for_open_generation, save_with_answers
from src.language_modeling.memory_utils import normalize_layer_keys
from src.language_modeling.utils import GAG_TOKEN, get_yaml_file, resolve_path
from src.model import XQwen3ForCausalLM


def resolve_model_class(config):
    architecture = None
    if getattr(config, "architectures", None):
        architecture = config.architectures[0]

    if architecture in (None, "XQwen3ForCausalLM", "Qwen3ForCausalLM"):
        return XQwen3ForCausalLM
    raise ValueError(f"Unsupported architecture for GAG evaluation: {architecture}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_file_path", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--chat_format", type=str, default=None)
    parser.add_argument("--enable_progress_bar", type=eval, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--layer_keys", nargs="*", default=None)
    parser.add_argument("--embed_key", type=str, default=None)
    parser.add_argument("--num_memory_slots", type=int, default=None)
    parser.add_argument("--slot_pooling", type=str, default=None)
    parser.add_argument("--slot_pooling_temperature", type=float, default=None)
    parser.add_argument("--instruction_text", type=str, default=None)
    parser.add_argument("--request_text", type=str, default=None)
    parser.add_argument("--do_sample_big", type=eval, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    yaml_config = get_yaml_file(args.config)
    for key, value in yaml_config.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    if args.embed_key and not args.layer_keys:
        args.layer_keys = [args.embed_key]
    args.layer_keys = normalize_layer_keys(args.layer_keys)
    args.workdir = resolve_path(None, args.workdir or os.path.dirname(os.path.abspath(args.config)))
    args.data_path = resolve_path(args.workdir, args.data_path)
    args.output_file_path = resolve_path(args.workdir, args.output_file_path)
    args.model_name_or_path = resolve_path(args.workdir, args.model_name_or_path)
    return args


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="left",
        add_eos_token=False,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_data = load_data(args.data_path)
    if args.max_test_samples is not None:
        test_data = test_data[: args.max_test_samples]

    prompts, backgrounds = prepare_prompts(
        test_data=test_data,
        tokenizer=tokenizer,
        chat_format=args.chat_format,
        layer_keys=args.layer_keys,
        num_memory_slots=args.num_memory_slots,
        slot_pooling=args.slot_pooling,
        slot_pooling_temperature=args.slot_pooling_temperature,
        instruction_text=args.instruction_text,
        request_text=args.request_text,
    )

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model_class = resolve_model_class(config)
    model = model_class.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    model.set_gag_token_id(tokenizer.convert_tokens_to_ids(GAG_TOKEN))

    generated_results = llm_for_open_generation(
        llm=model,
        llm_tokenizer=tokenizer,
        prompts=prompts,
        retrieval_embeds=backgrounds,
        batch_size=args.eval_batch_size,
        enable_progress_bar=args.enable_progress_bar,
        do_sample=args.do_sample_big,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    save_with_answers(
        test_data=test_data,
        generated_results=generated_results,
        output_file_path=args.output_file_path,
    )


if __name__ == "__main__":
    main()
