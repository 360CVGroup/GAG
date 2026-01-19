#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from tqdm import tqdm
# from langchain.schema import Document
from langchain_core.documents import Document
from langchain_text_splitters.markdown import MarkdownTextSplitter
from transformers import AutoTokenizer

# ---------- 配置部分 ----------
INPUT_PATH  = "/home/jovyan/projection/MinerU/material_code/vsrag/MinerU_extract_markdown/dataset/material_domain_knowledge_base.jsonl"
OUTPUT_PATH = "/home/jovyan/projection/MinerU/material_code/vsrag/MinerU_extract_markdown/dataset/material_domain_knowledge_base_chunked_480_overlap0.jsonl"


MODEL_PATH  = "/home/jovyan/models/Qwen3-1.7B"
# --------------------------------

# 1. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 2. 初始化 Markdown 分块器
splitter = MarkdownTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=480,
    chunk_overlap=0,
)

# 根据经验，自定义分隔符优先级
splitter.separators = [
    "\n# ",   # 一级标题
    "\n## ",  # 二级标题
    "\n### ", # 三级标题
    "\n\n",   # 空行
    "\n",     # 换行
    ". ",     # 句号加空格（英文句子）
]

with open(INPUT_PATH, "r", encoding="utf-8") as fin, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as fout:

    for line in tqdm(fin, desc="Splitting material-domain docs"):
        if not line.strip():
            continue
        obj = json.loads(line)

        text = obj.get("text", "")
        source = obj.get("source", "")

        # LangChain 的 Document，metadata 里放 source，方便后面检索用
        doc = Document(
            page_content=text,
            metadata={"source": source}
        )

        # 用 splitter 切分
        chunks = splitter.split_documents([doc])

        # 每个子块写成一行 json
        for chunk in chunks:
            fout.write(json.dumps({
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "")
            }, ensure_ascii=False) + "\n")
