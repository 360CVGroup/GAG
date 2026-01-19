#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from colbert import Indexer
from colbert.infra import Run, RunConfig
from colbert.infra import ColBERTConfig

# ------------ 配置部分 ------------
# 材料领域 chunk 好的知识库
collection_path = "projection/MinerU/material_code/vsrag/MinerU_extract_markdown/dataset/material_domain_knowledge_base_chunked_480_overlap0.jsonl"

# Index 名称，可自定义
index_name   = "material_domain.chunked480.overlap0.2bits"

# ColBERT v2 checkpoint（沿用你之前的路径）
checkpoint   = "models/colbertv2.0"

# 索引参数（与佐剂脚本保持一致）
nbits         = 2       # 每个维度量化到 2 bits
doc_maxlen    = 512     # 每篇文档截断到 512 tokens
kmeans_niters = 4       # k-means 聚类迭代次数
# --------------------------------


def main():
    # 1. 读取 JSONL，提取 text 字段构建语料列表
    collection = []
    with open(collection_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            collection.append(obj["text"])

    print(f"Loaded {len(collection):,} documents for indexing")

    # 2. 创建索引
    # nranks=6 沿用你的 adjuvant 脚本，如果只有单卡，可改成 nranks=1
    with Run().context(RunConfig(nranks=6, experiment="material_domain")):
        config = ColBERTConfig(
            doc_maxlen=doc_maxlen,
            nbits=nbits,
            kmeans_niters=kmeans_niters,
        )

        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(
            name=index_name,
            collection=collection,
            overwrite=True,   # 若已存在同名索引则覆盖
        )

    print(f"Indexing complete! Index saved as '{index_name}'")


if __name__ == "__main__":
    main()
