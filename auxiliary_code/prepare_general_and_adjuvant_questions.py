#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
from typing import List, Tuple


def load_and_split_by_label(path: str) -> Tuple[List[str], List[str]]:
    """
    从 jsonl 中加载数据，按 label 分成两组：
      - positives: label == 1
      - negatives: label == 0
    返回值为 (positives, negatives)，元素为原始行字符串（不改动）
    """
    positives = []
    negatives = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw_line = line.rstrip("\n")
            if not raw_line.strip():
                continue

            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                # 非合法 JSON 行直接跳过
                continue

            label = obj.get("label", None)

            # 兼容 int 和 str
            if label == 1 or label == "1":
                positives.append(raw_line)
            elif label == 0 or label == "0":
                negatives.append(raw_line)
            else:
                # 其他 label 直接跳过
                continue

    return positives, negatives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="/home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/train_gate_10_26/dataset/with_popqa_mixed_dataset/eval_dataset/1professional_6general_mixed_dataset_same_quantity.jsonl",
        help="输入 jsonl 文件路径",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="/home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/dataset/eval/general_and_adjuvant_questions.jsonl",
        help="输出 jsonl 文件路径",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=980406,
        help="随机种子，保证可复现",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    print(f"[Load] from {args.input_jsonl}")
    positives, negatives = load_and_split_by_label(args.input_jsonl)
    num_pos = len(positives)
    num_neg = len(negatives)
    print(f"  label == 1 : {num_pos}")
    print(f"  label == 0 : {num_neg}")

    if num_pos == 0:
        raise RuntimeError("没有找到任何 label == 1 的样本，无法构造等量抽样。")

    # 抽取与 positives 等量的 0 样本
    sample_size = num_pos
    if num_neg < sample_size:
        print(f"[Warn] label==0 的样本数量 ({num_neg}) < label==1 的数量 ({num_pos})，"
              f"将使用所有 {num_neg} 个 label==0 样本。")
        sample_size = num_neg

    sampled_negatives = random.sample(negatives, sample_size)
    print(f"[Sample] 使用 {len(sampled_negatives)} 条 label==0 样本。")

    # 合并并 shuffle
    all_lines = positives + sampled_negatives
    random.shuffle(all_lines)

    # 原样写入输出文件
    print(f"[Write] to {args.output_jsonl}")
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for line in all_lines:
            fout.write(line + "\n")

    print(f"[OK] 总写入 {len(all_lines)} 条样本。")


if __name__ == "__main__":
    main()
