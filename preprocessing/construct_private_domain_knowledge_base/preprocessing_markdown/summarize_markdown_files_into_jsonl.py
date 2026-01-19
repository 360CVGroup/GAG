#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

# 基础目录：包含若干子文件夹，每个子文件夹下有 vlm/clean_子文件夹名.md
BASE_DIR = (
    "/home/jovyan/projection/MinerU/material_code/vsrag/MinerU_extract_markdown/mineru_extract_result_markdown"
)

# 输出 jsonl 路径
OUTPUT_JSONL = (
    "/home/jovyan/projection/MinerU/material_code/vsrag/MinerU_extract_markdown/dataset/material_domain_knowledge_base.jsonl"
)


def main():
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    num_written = 0

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        # 遍历 BASE_DIR 下的子文件夹
        for sub in sorted(os.listdir(BASE_DIR)):
            sub_dir = os.path.join(BASE_DIR, sub)
            if not os.path.isdir(sub_dir):
                continue  # 只处理子目录

            # 子目录名作为 source
            source_name = os.path.basename(sub_dir)

            # 该子目录下的 md 路径：sub/vlm/clean_子文件夹名.md
            vlm_dir = os.path.join(sub_dir, "vlm")
            md_path = os.path.join(vlm_dir, f"clean_{source_name}.md")

            if not os.path.isfile(md_path):
                # 如果没有这个文件，可以选择跳过并打印提示
                print(f"[WARN] 找不到文件: {md_path}")
                continue

            # 读取 md 文件全部内容
            with open(md_path, "r", encoding="utf-8") as fin:
                text = fin.read()

            # 组成一条 jsonl 数据
            record = {
                "text": text,
                "source": source_name,
            }

            # 写入一行 jsonl
            json.dump(record, fout, ensure_ascii=False)
            fout.write("\n")
            num_written += 1

    print(f"完成！共写入 {num_written} 条数据到: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
