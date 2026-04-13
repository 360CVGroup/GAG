import argparse
import json
import os
import random
import types
from typing import Dict, List

import torch
import torch.distributed as dist
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

from src.language_modeling.utils import get_yaml_file, load_tokenizer_with_fast_fallback, resolve_path


try:
    import datasets as _datasets_module
except ImportError:
    _datasets_module = types.SimpleNamespace()

if not hasattr(_datasets_module, "Dataset"):
    class _TrainerDatasetSentinel:  # pragma: no cover - compatibility shim
        pass

    _datasets_module.Dataset = _TrainerDatasetSentinel


Dataset = _datasets_module.Dataset


class ListDataset(TorchDataset):
    def __init__(self, records: List[Dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def _concat_messages_qwen3(messages, tokenizer):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def encode_example(example, tokenizer, max_seq_length: int):
    user_prompt = example["instruction"].strip() + "\nQuestion: " + example["input"].strip()
    answer = example["output"].strip()

    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": answer},
    ]
    full_text = _concat_messages_qwen3(messages, tokenizer).strip()
    tokenized = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    )
    input_ids = tokenized.input_ids[0]
    labels = input_ids.clone()

    user_prefix = _concat_messages_qwen3(messages[:1], tokenizer) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    user_length = tokenizer(
        user_prefix,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    ).input_ids.shape[1]
    labels[:user_length] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized.attention_mask[0],
        "labels": labels,
    }


class SupervisedDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]
        labels = [feature["labels"] for feature in features]

        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )
        label_batch = self.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
        )["input_ids"]
        label_batch[label_batch == self.tokenizer.pad_token_id] = -100
        batch["labels"] = label_batch
        return batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--bf16", type=eval, default=None)
    parser.add_argument("--gradient_checkpointing", type=eval, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--validation_split_ratio", type=float, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--eval_strategy", type=str, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_strategy", type=str, default=None)
    parser.add_argument("--load_best_model_at_end", type=eval, default=None)
    parser.add_argument("--metric_for_best_model", type=str, default=None)
    parser.add_argument("--greater_is_better", type=eval, default=None)
    parser.add_argument("--ddp_find_unused_parameters", type=eval, default=None)
    parser.add_argument("--fsdp", type=str, default=None)
    parser.add_argument("--fsdp_config", type=str, default=None)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--save_only_model", type=eval, default=None)
    args = parser.parse_args()

    yaml_config = get_yaml_file(args.config)
    for key, value in yaml_config.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    args.workdir = resolve_path(None, args.workdir or os.path.dirname(os.path.abspath(args.config)))
    args.train_path = resolve_path(args.workdir, args.train_path)
    args.eval_path = resolve_path(args.workdir, args.eval_path) if args.eval_path else None
    args.model_name_or_path = resolve_path(args.workdir, args.model_name_or_path)
    args.output_dir = resolve_path(args.workdir, args.output_dir)
    args.fsdp_config = resolve_path(args.workdir, args.fsdp_config) if args.fsdp_config else None
    args.deepspeed = resolve_path(args.workdir, args.deepspeed) if args.deepspeed else None
    return args


def load_json_records(path: str, max_samples: int | None = None) -> List[Dict]:
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                if max_samples is not None and index >= max_samples:
                    break
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with open(path, "r", encoding="utf-8") as file:
        records = json.load(file)
    if max_samples is not None:
        records = records[:max_samples]
    return records


def split_records_for_train_eval(
    records: List[Dict],
    validation_split_ratio: float,
    seed: int,
):
    if not validation_split_ratio or validation_split_ratio <= 0.0:
        return records, None
    if validation_split_ratio >= 1.0:
        raise ValueError("validation_split_ratio must be < 1.0")

    grouped_records = {}
    for record in records:
        group_key = (record["instruction"].strip(), record["input"].strip())
        grouped_records.setdefault(group_key, []).append(record)

    group_keys = list(grouped_records.keys())
    if len(group_keys) < 2:
        raise RuntimeError("Need at least two unique prompt groups to create a validation split.")

    rng = random.Random(seed)
    rng.shuffle(group_keys)
    num_eval_groups = max(1, int(round(len(group_keys) * validation_split_ratio)))
    num_eval_groups = min(num_eval_groups, len(group_keys) - 1)
    eval_group_keys = set(group_keys[:num_eval_groups])

    train_records = []
    eval_records = []
    for group_key, group in grouped_records.items():
        if group_key in eval_group_keys:
            eval_records.extend(group)
        else:
            train_records.extend(group)

    if not train_records or not eval_records:
        raise RuntimeError("Validation split produced an empty train or eval set.")
    return train_records, eval_records


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = load_tokenizer_with_fast_fallback(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_json_records(args.train_path, args.max_samples)
    if args.eval_path:
        train_records = records
        eval_records = load_json_records(args.eval_path)
    else:
        train_records, eval_records = split_records_for_train_eval(
            records=records,
            validation_split_ratio=args.validation_split_ratio,
            seed=args.seed,
        )

    train_dataset = ListDataset(
        [encode_example(example, tokenizer, args.max_seq_length) for example in train_records]
    )
    eval_dataset = None
    if eval_records is not None:
        eval_dataset = ListDataset(
            [encode_example(example, tokenizer, args.max_seq_length) for example in eval_records]
        )

    print(
        json.dumps(
            {
                "num_train_examples": len(train_records),
                "num_eval_examples": len(eval_records) if eval_records is not None else 0,
                "num_train_unique_prompts": len({(x["instruction"], x["input"]) for x in train_records}),
                "num_eval_unique_prompts": len({(x["instruction"], x["input"]) for x in eval_records}) if eval_records is not None else 0,
            },
            indent=2,
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        low_cpu_mem_usage=True,
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    fsdp_config = None
    if args.fsdp_config:
        if args.fsdp_config.endswith(".json"):
            with open(args.fsdp_config, "r", encoding="utf-8") as file:
                fsdp_config = json.load(file)
        else:
            fsdp_config = get_yaml_file(args.fsdp_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_train_steps if args.max_train_steps is not None else -1,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        do_eval=eval_dataset is not None,
        eval_strategy=args.eval_strategy if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        fsdp=args.fsdp,
        fsdp_config=fsdp_config,
        deepspeed=args.deepspeed,
        save_only_model=bool(args.save_only_model) if args.save_only_model is not None else False,
        load_best_model_at_end=bool(args.load_best_model_at_end) if eval_dataset is not None else False,
        metric_for_best_model=args.metric_for_best_model if eval_dataset is not None else None,
        greater_is_better=args.greater_is_better if eval_dataset is not None else None,
        report_to=[] if not args.report_to or args.report_to == "none" else [args.report_to],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=SupervisedDataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
