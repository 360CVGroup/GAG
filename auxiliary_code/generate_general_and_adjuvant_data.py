#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

def split_jsonl_by_label(
    input_path: str,
    output_path_label0: str,
    output_path_label1: str
) -> None:
    """
    按行读取 input_path 中的 JSONL 数据，根据 label 字段将原始行写入不同文件：
      - label == 0 -> 写入 output_path_label0
      - label == 1 -> 写入 output_path_label1
    其他情况会跳过。
    """
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path_label0, "w", encoding="utf-8") as fout0, \
         open(output_path_label1, "w", encoding="utf-8") as fout1:

        for line in fin:
            # 保留原始行内容，用于直接写回
            raw_line = line.rstrip("\n")
            if not raw_line.strip():
                # 空行直接跳过（如果想保留空行，可改成 continue 以外的逻辑）
                continue

            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                # 如果某行不是合法 JSON，就跳过或打印提示
                # print(f"跳过一行非合法 JSON: {raw_line[:100]}...")
                continue

            label = obj.get("label", None)

            # 兼容 label 是数字或字符串的情况
            if label == 0 or label == "0":
                fout0.write(raw_line + "\n")
            elif label == 1 or label == "1":
                fout1.write(raw_line + "\n")
            else:
                # 如果没有 label 或 label 不是 0/1，就跳过或根据需要处理
                # print(f"发现未知 label（{label}），跳过该行: {raw_line[:100]}...")
                continue


if __name__ == "__main__":
    input_file = "/home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/train_gate_10_26/dataset/with_popqa_mixed_dataset/1professional_6general_mixed_dataset_train.jsonl"

    output_label0 = "/home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/dataset/general_questions.jsonl"
    output_label1 = "/home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/dataset/adjuvant_questions.jsonl"

    split_jsonl_by_label(input_file, output_label0, output_label1)
