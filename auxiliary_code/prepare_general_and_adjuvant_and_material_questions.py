#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
from typing import List, Tuple


def load_eval_file(path: str) -> Tuple[List[str], int]:
    """
    读取 general_and_adjuvant_questions.jsonl：
      - 返回所有样本的原始行字符串列表 lines
      - 返回其中 label == 0 的样本数量 num_label0
    """
    lines = []
    num_label0 = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw_line = line.rstrip("\n")
            if not raw_line.strip():
                continue

            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                # 非法 JSON 直接跳过
                continue

            label = obj.get("label", None)
            if label == 0 or label == "0":
                num_label0 += 1

            # 保留原始行（后面原样写回）
            lines.append(raw_line)

    if not lines:
        raise RuntimeError(f"No valid JSON lines loaded from {path}")

    return lines, num_label0


def sample_and_transform_material(
    path: str,
    sample_size: int,
) -> List[str]:
    """
    从 material_test_SFT.jsonl 中随机抽取 sample_size 条数据，并做字段变换：
      - id -> "material" + 原始id
      - 删除 instruction
      - input -> question
      - output -> answer
      - 新增 label = 2
      - 新增 dataset_name = "material"
    返回：转换后的 JSON 行字符串列表（每元素是一行 json.dumps 的结果）
    """
    all_objs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw_line = line.rstrip("\n")
            if not raw_line.strip():
                continue

            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            all_objs.append(obj)

    if not all_objs:
        raise RuntimeError(f"No valid JSON lines loaded from {path}")

    if len(all_objs) < sample_size:
        print(
            f"[Warn] material 样本数 ({len(all_objs)}) 小于需要抽取的数量 ({sample_size})，"
            f"将仅使用 {len(all_objs)} 条 material 样本。"
        )
        sample_size = len(all_objs)

    sampled = random.sample(all_objs, sample_size)

    out_lines = []
    for obj in sampled:
        orig_id = str(obj.get("id", ""))
        new_obj = {
            "id": "material_" + orig_id,
            "question": obj.get("input", ""),
            "answer": obj.get("output", ""),
            "label": 2,
            "dataset_name": "material",
        }
        # 序列化为一行 JSON 字符串
        out_lines.append(json.dumps(new_obj, ensure_ascii=False))

    return out_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_jsonl",
        type=str,
        default="/home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/dataset/eval/general_and_adjuvant_questions.jsonl",
        help="通用+佐剂评估集（包含 label 0 和 1）",
    )
    parser.add_argument(
        "--material_jsonl",
        type=str,
        default="/home/jovyan/lirongji-2/projection/MinerU/material_code/train_projector/dataset/material_test_SFT.jsonl",
        help="材料领域 SFT 测试集",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="/home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/dataset/eval/general_and_adjuvant_and_material_questions.jsonl",
        help="输出的混合评估集路径",
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

    # 1. 读取 general+adjuvant eval 文件
    print(f"[Load] eval file from {args.eval_jsonl}")
    eval_lines, num_label0 = load_eval_file(args.eval_jsonl)
    print(f"  total eval samples = {len(eval_lines)}")
    print(f"  label == 0 samples = {num_label0}")

    if num_label0 == 0:
        raise RuntimeError("eval 文件中没有 label == 0 的样本，无法按要求匹配数量。")

    # 2. 从 material_test_SFT 中抽取与 label0 同数量的样本并转换为新 schema
    print(f"[Sample+Transform] from material file {args.material_jsonl}")
    material_lines = sample_and_transform_material(args.material_jsonl, num_label0)
    print(f"  sampled material samples = {len(material_lines)} (label == 2)")

    # 3. 合并三种 label 的数据（label 0 和 1 来自 eval_lines，label 2 来自 material_lines）
    all_lines = eval_lines + material_lines
    print(f"[Merge] total samples before shuffle = {len(all_lines)}")

    # 4. shuffle
    random.shuffle(all_lines)

    # 5. 写出到目标文件
    print(f"[Write] to {args.output_jsonl}")
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for line in all_lines:
            fout.write(line + "\n")

    print(f"[OK] final total samples = {len(all_lines)}")


if __name__ == "__main__":
    main()
