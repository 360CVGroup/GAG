import argparse
import os
import pickle

from tqdm import tqdm

from src.language_modeling.memory_utils import build_memory_slots_from_record, normalize_layer_keys
from src.language_modeling.utils import get_yaml_file, resolve_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--layer_keys", nargs="*", default=None)
    parser.add_argument("--num_memory_slots", type=int, default=None)
    parser.add_argument("--slot_pooling", type=str, default=None)
    parser.add_argument("--slot_pooling_temperature", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.config:
        yaml_config = get_yaml_file(args.config)
        for key, value in yaml_config.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)

    args.layer_keys = normalize_layer_keys(args.layer_keys)
    args.workdir = resolve_path(None, args.workdir or (os.path.dirname(os.path.abspath(args.config)) if args.config else os.getcwd()))
    args.input_path = resolve_path(args.workdir, args.input_path)
    args.output_path = resolve_path(args.workdir, args.output_path)
    return args


def iter_pickled_records(path):
    with open(path, "rb") as file:
        while True:
            try:
                yield pickle.load(file)
            except EOFError:
                break


def main():
    args = parse_args()
    if os.path.exists(args.output_path) and not args.overwrite:
        raise FileExistsError(f"{args.output_path} already exists. Use --overwrite to replace it.")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path, "wb") as output_file:
        for record in tqdm(iter_pickled_records(args.input_path), desc="Compressing background cache", ncols=100):
            compact_record = {
                key: record.get(key)
                for key in [
                    "id",
                    "instruction",
                    "input",
                    "output",
                    "qwen3-1.7B_answer_background",
                    "answer_ids",
                ]
                if key in record
            }
            for layer_key in args.layer_keys:
                field_name = {
                    "last": "last_layer_answer_tokens_embedding",
                    "minus2": "layer_minus2_answer_tokens_embedding",
                    "minus4": "layer_minus4_answer_tokens_embedding",
                    "minus6": "layer_minus6_answer_tokens_embedding",
                    "minus8": "layer_minus8_answer_tokens_embedding",
                }[layer_key]
                compact_record[field_name] = build_memory_slots_from_record(
                    record=record,
                    layer_keys=[layer_key],
                    num_memory_slots=args.num_memory_slots,
                    pooling=args.slot_pooling,
                    temperature=args.slot_pooling_temperature,
                ).squeeze(0)
            pickle.dump(compact_record, output_file)


if __name__ == "__main__":
    main()
