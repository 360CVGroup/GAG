#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from colbert import Searcher
from colbert.infra import Run, RunConfig
from tqdm import tqdm

# ------------ 配置部分 ------------

# 与索引阶段对应的 index 名称和 experiment 名称
index_name      = "material_domain.chunked480.overlap0.2bits"
experiment_name = "material_domain"

# 用于检索的语料（索引时用的同一个 JSONL）
collection_path = "projection/MinerU/material_code/vsrag/MinerU_extract_markdown/dataset/material_domain_knowledge_base_chunked_480_overlap0.jsonl"

# 材料领域 SFT 测试集（每行一个样本）
input_file = "projection/MinerU/material_code/train_projector/dataset/material_test_SFT.jsonl"

# 输出文件：在原样本基础上附加 top1~top10 背景
output_file = "projection/MinerU/material_code/vsrag/colbert_retrieval/material_test_SFT_with_colbert_top10_backgrounds.jsonl"

# 要检索的 top-k 文档数
top_k = 10
# ----------------------------------


# 1. 加载 collection：同时记录 text 和 source
collection_texts = []
collection_sources = []

with open(collection_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line.strip())
        collection_texts.append(obj["text"])
        collection_sources.append(obj.get("source", ""))

print(f"Loaded {len(collection_texts):,} documents into memory for retrieval.")


# 2. 读取 material_train_SFT.jsonl，构建 queries 列表，同时保存原始样本
samples = []
queries = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line.strip())
        # 对应你的 SFT 格式：
        #   "input":  问题
        #   "output": 回答
        question = obj.get("input", "").strip()
        # answer 不直接参与 query，只放回输出里
        answer   = obj.get("output", "").strip()

        samples.append(obj)
        queries.append(question)

print(f"Loaded {len(queries)} queries from {input_file}")


# 3. 初始化 Searcher
with Run().context(RunConfig(experiment=experiment_name)):
    searcher = Searcher(index=index_name, collection=collection_texts)
print("Searcher initialized.")


# 4. 对每个 query 执行检索，在原样本基础上附加 top1~top10
with open(output_file, "w", encoding="utf-8") as out_f:
    for idx, query in enumerate(tqdm(queries, desc="Processing queries", unit="query")):
        # ColBERT 检索
        doc_ids, ranks, scores = searcher.search(query, k=top_k)

        # 组织 top-k 结果，包含 text 和 source
        record = dict(samples[idx])   # 先复制原来的整条样本（id, instruction, input, output 等都保留）

        for i, doc_id in enumerate(doc_ids, start=1):
            text = collection_texts[doc_id]
            src  = collection_sources[doc_id]
            record[f"top{i}"] = {
                "text": text,
                "source": src,
            }

        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("✅ Retrieval complete! Outputs saved to:")
print(f"   {output_file}")
