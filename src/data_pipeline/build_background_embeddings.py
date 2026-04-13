import argparse
import heapq
import json
import os
import pickle
from typing import Dict, Iterable, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from src.language_modeling.memory_utils import normalize_layer_keys
from src.language_modeling.utils import get_yaml_file, load_tokenizer_with_fast_fallback, resolve_path


LAYER_KEY_TO_OUTPUT_FIELD = {
    "last": "last_layer_answer_tokens_embedding",
    "minus2": "layer_minus2_answer_tokens_embedding",
    "minus4": "layer_minus4_answer_tokens_embedding",
    "minus6": "layer_minus6_answer_tokens_embedding",
    "minus8": "layer_minus8_answer_tokens_embedding",
}


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret boolean value from: {value}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--small_model_name_or_path", type=str, default=None)
    parser.add_argument("--question_key", type=str, default=None)
    parser.add_argument("--answer_key", type=str, default=None)
    parser.add_argument("--instruction_key", type=str, default=None)
    parser.add_argument("--default_instruction", type=str, default=None)
    parser.add_argument("--background_system_prompt", type=str, default=None)
    parser.add_argument("--layer_keys", nargs="*", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--do_sample", type=str_to_bool, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resume", type=str_to_bool, default=None)
    parser.add_argument("--trust_remote_code", type=str_to_bool, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--shard_rank", type=int, default=None)
    parser.add_argument("--merge_shards", type=str_to_bool, default=None)
    parser.add_argument("--clean_merged_shards", type=str_to_bool, default=None)
    args = parser.parse_args()

    yaml_config = get_yaml_file(args.config)
    for key, value in yaml_config.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    args.layer_keys = normalize_layer_keys(args.layer_keys)
    args.workdir = resolve_path(None, args.workdir or os.path.dirname(os.path.abspath(args.config)))
    args.input_path = resolve_path(args.workdir, args.input_path)
    args.output_path = resolve_path(args.workdir, args.output_path)
    args.small_model_name_or_path = resolve_path(args.workdir, args.small_model_name_or_path)
    args.num_shards = int(args.num_shards or 1)
    args.shard_rank = int(args.shard_rank or 0)
    args.merge_shards = bool(args.merge_shards) if args.merge_shards is not None else False
    args.clean_merged_shards = bool(args.clean_merged_shards) if args.clean_merged_shards is not None else False
    if args.num_shards <= 0:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= args.shard_rank < args.num_shards):
        raise ValueError(f"shard_rank must be in [0, {args.num_shards}), got {args.shard_rank}")
    return args


def load_records(path: str) -> List[Dict]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as file:
            return [json.loads(line) for line in file]
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def count_pickled_objects(path: str) -> int:
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path, "rb") as file:
        while True:
            try:
                pickle.load(file)
                count += 1
            except EOFError:
                break
    return count


def iter_pickled_objects(path: str) -> Iterable[Dict]:
    with open(path, "rb") as file:
        while True:
            try:
                yield pickle.load(file)
            except EOFError:
                break


def get_shard_output_path(output_path: str, shard_rank: int, num_shards: int) -> str:
    if num_shards <= 1:
        return output_path
    return f"{output_path}.shard{shard_rank:02d}of{num_shards:02d}"


def select_shard_records(records: List[Dict], shard_rank: int, num_shards: int) -> List[Tuple[int, Dict]]:
    return [(index, record) for index, record in enumerate(records) if index % num_shards == shard_rank]


def merge_sharded_outputs(output_path: str, num_shards: int, clean_merged_shards: bool = False) -> None:
    shard_paths = [get_shard_output_path(output_path, rank, num_shards) for rank in range(num_shards)]
    missing = [path for path in shard_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing shard outputs: {missing}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    shard_files = [open(path, "rb") for path in shard_paths]
    heap: List[Tuple[int, int, Dict]] = []

    try:
        for shard_id, file in enumerate(shard_files):
            try:
                record = pickle.load(file)
            except EOFError:
                continue
            source_index = int(record.get("__source_index", shard_id))
            heapq.heappush(heap, (source_index, shard_id, record))

        with open(output_path, "wb") as merged_file:
            while heap:
                _, shard_id, record = heapq.heappop(heap)
                record.pop("__source_index", None)
                pickle.dump(record, merged_file)

                try:
                    next_record = pickle.load(shard_files[shard_id])
                except EOFError:
                    continue
                source_index = int(next_record.get("__source_index", shard_id))
                heapq.heappush(heap, (source_index, shard_id, next_record))
    finally:
        for file in shard_files:
            file.close()

    if clean_merged_shards:
        for path in shard_paths:
            os.remove(path)


def build_background_prompt(question: str, tokenizer, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": question.strip()},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def select_next_token(logits, do_sample, temperature, top_p, top_k):
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-5)
    if top_k and top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    if top_p and 0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = probs.cumsum(dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_logits[sorted_mask] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(1, sorted_indices, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_background_and_hidden_states(
    model,
    tokenizer,
    prompt: str,
    layer_keys: List[str],
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    device: torch.device,
):
    tokenized = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    outputs = model(
        **tokenized,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values

    generated_ids: List[int] = []
    hidden_state_cache = {key: [] for key in layer_keys}

    for _ in range(max_new_tokens):
        logits = outputs.logits[:, -1, :]
        next_token = select_next_token(
            logits=logits,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        token_id = int(next_token.item())
        if token_id == tokenizer.eos_token_id:
            break

        generated_ids.append(token_id)
        outputs = model(
            input_ids=next_token,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values

        step_hidden_states = outputs.hidden_states
        for layer_key in layer_keys:
            if layer_key == "last":
                selected = step_hidden_states[-1]
            elif layer_key == "minus2":
                selected = step_hidden_states[-2]
            elif layer_key == "minus4":
                selected = step_hidden_states[-4]
            elif layer_key == "minus6":
                selected = step_hidden_states[-6]
            elif layer_key == "minus8":
                selected = step_hidden_states[-8]
            else:
                raise ValueError(f"Unsupported layer key: {layer_key}")
            hidden_state_cache[layer_key].append(selected[:, -1:, :].detach().cpu())

    if not generated_ids:
        generated_ids = [tokenizer.eos_token_id]

    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    hidden_states = {}
    for layer_key in layer_keys:
        cached = hidden_state_cache[layer_key]
        if cached:
            hidden_states[LAYER_KEY_TO_OUTPUT_FIELD[layer_key]] = torch.cat(cached, dim=1)
        else:
            hidden_size = model.config.hidden_size
            hidden_states[LAYER_KEY_TO_OUTPUT_FIELD[layer_key]] = torch.zeros(1, 1, hidden_size)

    return answer_text, generated_ids, hidden_states


def main():
    args = parse_args()
    if args.merge_shards:
        merge_sharded_outputs(
            output_path=args.output_path,
            num_shards=args.num_shards,
            clean_merged_shards=args.clean_merged_shards,
        )
        return

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    records = load_records(args.input_path)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    sharded_records = select_shard_records(records, args.shard_rank, args.num_shards)

    tokenizer = load_tokenizer_with_fast_fallback(
        args.small_model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.small_model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    ).to(device)
    model.eval()

    shard_output_path = get_shard_output_path(args.output_path, args.shard_rank, args.num_shards)
    processed_count = count_pickled_objects(shard_output_path) if args.resume else 0
    output_mode = "ab" if args.resume else "wb"

    with open(shard_output_path, output_mode) as output_file:
        for shard_index, (source_index, record) in enumerate(
            tqdm(
                sharded_records,
                desc=f"Building GAG memories [shard {args.shard_rank + 1}/{args.num_shards}]",
                ncols=100,
            )
        ):
            if shard_index < processed_count:
                continue

            question = record[args.question_key]
            answer = record[args.answer_key]
            instruction = (
                record.get(args.instruction_key)
                if args.instruction_key and args.instruction_key in record
                else args.default_instruction
            )
            prompt = build_background_prompt(
                question=question,
                tokenizer=tokenizer,
                system_prompt=args.background_system_prompt,
            )
            answer_background, generated_ids, hidden_states = generate_background_and_hidden_states(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                layer_keys=args.layer_keys,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                device=device,
            )

            standardized = {
                "id": record.get("id", source_index + 1),
                "instruction": instruction,
                "input": question,
                "output": answer,
                "qwen3-1.7B_answer_background": answer_background,
                "answer_ids": generated_ids,
            }
            if args.num_shards > 1:
                standardized["__source_index"] = source_index
            standardized.update(hidden_states)
            pickle.dump(standardized, output_file)
            output_file.flush()


if __name__ == "__main__":
    main()
