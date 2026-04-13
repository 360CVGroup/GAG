import argparse
import importlib
import json
import os
import sys

import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.language_modeling.utils import get_yaml_file, load_tokenizer_with_fast_fallback, resolve_path


def import_hf_dataset_class():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    blocked_paths = {
        os.path.abspath(os.getcwd()),
        repo_root,
    }
    original_sys_path = list(sys.path)
    try:
        sys.path = [
            path
            for path in original_sys_path
            if os.path.abspath(path or os.getcwd()) not in blocked_paths
        ]
        hf_datasets = importlib.import_module("datasets")
    finally:
        sys.path = original_sys_path
    return hf_datasets.Dataset


Dataset = import_hf_dataset_class()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--corpus_path", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--text_key", type=str, default=None)
    parser.add_argument("--block_size", type=int, default=None)
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
    parser.add_argument("--packing_separator", type=str, default=None)
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--validation_split_ratio", type=float, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--eval_strategy", type=str, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_strategy", type=str, default=None)
    parser.add_argument("--load_best_model_at_end", type=eval, default=None)
    parser.add_argument("--metric_for_best_model", type=str, default=None)
    parser.add_argument("--greater_is_better", type=eval, default=None)
    args = parser.parse_args()

    yaml_config = get_yaml_file(args.config)
    for key, value in yaml_config.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    args.workdir = resolve_path(None, args.workdir or os.path.dirname(os.path.abspath(args.config)))
    args.corpus_path = resolve_path(args.workdir, args.corpus_path)
    args.model_name_or_path = resolve_path(args.workdir, args.model_name_or_path)
    args.output_dir = resolve_path(args.workdir, args.output_dir)
    return args


def load_jsonl_texts(path: str, text_key: str, max_samples: int | None = None):
    texts = []
    with open(path, "r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if max_samples is not None and index >= max_samples:
                break
            record = json.loads(line)
            text = record.get(text_key, "")
            if text and text.strip():
                texts.append(text.strip())
    if not texts:
        raise RuntimeError(f"No non-empty texts found in {path} with key `{text_key}`")
    return texts


def build_lm_dataset(tokenizer, texts, block_size: int, separator: str):
    dataset = Dataset.from_dict({"text": texts})

    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=True, truncation=False)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing domain corpus",
    )

    separator_ids = tokenizer(separator, add_special_tokens=False)["input_ids"]

    def group_texts(examples):
        concatenated = []
        for input_ids in examples["input_ids"]:
            concatenated.extend(input_ids)
            if separator_ids:
                concatenated.extend(separator_ids)
        total_length = len(concatenated)
        total_length = (total_length // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "labels": []}
        input_ids = [
            concatenated[i : i + block_size] for i in range(0, total_length, block_size)
        ]
        return {"input_ids": input_ids, "labels": [ids[:] for ids in input_ids]}

    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        remove_columns=tokenized_dataset.column_names,
        desc="Packing domain corpus",
    )
    return lm_dataset


def split_texts_for_train_eval(texts, validation_split_ratio: float, seed: int):
    if not validation_split_ratio or validation_split_ratio <= 0.0:
        return texts, None
    if validation_split_ratio >= 1.0:
        raise ValueError("validation_split_ratio must be < 1.0")

    dataset = Dataset.from_dict({"text": texts})
    split = dataset.train_test_split(test_size=validation_split_ratio, seed=seed, shuffle=True)
    train_texts = split["train"]["text"]
    eval_texts = split["test"]["text"]
    if not train_texts or not eval_texts:
        raise RuntimeError("Validation split produced an empty train or eval set.")
    return train_texts, eval_texts


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = load_tokenizer_with_fast_fallback(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = load_jsonl_texts(args.corpus_path, args.text_key, args.max_samples)
    train_texts, eval_texts = split_texts_for_train_eval(
        texts=texts,
        validation_split_ratio=args.validation_split_ratio,
        seed=args.seed,
    )

    train_dataset = build_lm_dataset(
        tokenizer=tokenizer,
        texts=train_texts,
        block_size=args.block_size,
        separator=args.packing_separator,
    )
    eval_dataset = None
    if eval_texts is not None:
        eval_dataset = build_lm_dataset(
            tokenizer=tokenizer,
            texts=eval_texts,
            block_size=args.block_size,
            separator=args.packing_separator,
        )

    print(
        json.dumps(
            {
                "num_train_documents": len(train_texts),
                "num_eval_documents": len(eval_texts) if eval_texts is not None else 0,
                "num_train_blocks": len(train_dataset),
                "num_eval_blocks": len(eval_dataset) if eval_dataset is not None else 0,
            },
            indent=2,
        )
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
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
        load_best_model_at_end=bool(args.load_best_model_at_end) if eval_dataset is not None else False,
        metric_for_best_model=args.metric_for_best_model if eval_dataset is not None else None,
        greater_is_better=args.greater_is_better if eval_dataset is not None else None,
        report_to=[] if not args.report_to or args.report_to == "none" else [args.report_to],
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
