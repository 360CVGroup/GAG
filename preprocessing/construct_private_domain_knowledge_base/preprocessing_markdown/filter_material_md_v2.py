#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

# ========= 1. 正则模式：删除图片行 =========
IMG_PATTERN = re.compile(
    r'!\[[^\]]*\]\(images/.*\.(?:jpg|jpeg|png|gif|bmp|tiff|svg)\)',
    re.IGNORECASE
)

# ========= 2. 正则模式：截断尾部无用章节 =========
# 注意：这里不再包含 associated content，只匹配包含关键词的一级标题
CUT_PATTERN = re.compile(
    r'^#\s+.*('
    r'acknowledgment|acknowledgement|acknowledge|'
    r'reference|references|'
    r'supplementary material|supplementary materials|supplementary information|'
    r'author contribution|author contributions|'
    r'conflicts of interest|conflict of interest|'
    r'declaration of competing interest|declaration of competing interests|'
    r'author information|'
    r'data availability statement|'
    r'contribution statement|'
    r'supporting information|'
    r'questions'
    r')',
    re.IGNORECASE
)

# 精确匹配一些一级标题（# + 标题）就截断
# 这里加入 associated content，严格匹配 "# ASSOCIATED CONTENT"
EXACT_CUT_PATTERN = re.compile(
    r'^#\s+('
    r'contributors|'
    r'data availability|'
    r'funding|'
    r'authors|'
    r'associated content'
    r')\s*$',
    re.IGNORECASE
)

# ========= 3. Just Accepted 专门处理 =========
# 严格匹配一级标题 "# Just Accepted"（大小写不敏感）
JUST_ACCEPTED_PATTERN = re.compile(
    r'^#\s+just accepted\s*$',
    re.IGNORECASE
)

# 一级标题通用匹配（只匹配一个 # 开头）
H1_PATTERN = re.compile(r'^#\s+')


def filter_md_file(in_path, out_path):
    """
    对单个 md 文件做清洗：
    - 删除所有匹配 IMG_PATTERN 的行（图片）
    - 遇到匹配 CUT_PATTERN 或 EXACT_CUT_PATTERN 的一级标题时，
      删除该行及之后所有内容（直接 break）
    - 遇到严格匹配 "# Just Accepted" 这一节：
        * 删掉 "# Just Accepted" 那行
        * 删掉其后所有内容，直到下一个一级标题（# XXX）
        * 从下一个一级标题开始继续正常处理
    - 其他行原样保留
    """
    with open(in_path, 'r', encoding='utf-8') as fin, \
         open(out_path, 'w', encoding='utf-8') as fout:

        skip_just_accepted = False  # 当前是否处在 “Just Accepted” 这一节里

        for line in fin:
            stripped = line.rstrip('\n')

            # ---------- 先处理是否在 Just Accepted 区间 ----------
            if skip_just_accepted:
                # 遇到下一个一级标题，退出跳过模式，并继续按正常逻辑处理这一行
                if H1_PATTERN.match(stripped):
                    skip_just_accepted = False
                    # 不 continue，让下面逻辑接着处理这个新标题
                else:
                    # 仍在 Just Accepted 区块内，整行丢掉
                    continue

            # ---------- 检测是否为 "# Just Accepted"（严格匹配一级标题） ----------
            if JUST_ACCEPTED_PATTERN.match(stripped):
                skip_just_accepted = True
                # 标题行本身也不保留
                continue

            # ---------- 正常清洗逻辑 ----------

            # 1) 删除图片行
            if IMG_PATTERN.search(stripped):
                continue

            # 2) 遇到需要整体截断的尾部章节（包括严格匹配的 "# ASSOCIATED CONTENT" 等）
            if CUT_PATTERN.match(stripped):
                break
            if EXACT_CUT_PATTERN.match(stripped):
                break

            # 3) 其余行直接写出
            fout.write(line)


def process_material_folder(base_dir):
    """
    遍历 base_dir 下的每个子文件夹（每个子文件夹是一篇论文），
    进入其 vlm 子目录，处理里面所有 .md 文件。

    输出文件与源文件同目录，文件名前加前缀 'clean_'。
    """
    for paper in os.listdir(base_dir):
        paper_dir = os.path.join(base_dir, paper)
        if not os.path.isdir(paper_dir):
            continue

        vlm_dir = os.path.join(paper_dir, 'vlm')
        if not os.path.isdir(vlm_dir):
            continue

        for fname in os.listdir(vlm_dir):
            if not fname.lower().endswith('.md'):
                continue

            in_path = os.path.join(vlm_dir, fname)
            out_fname = 'clean_' + fname
            out_path = os.path.join(vlm_dir, out_fname)

            # print(f'Processing: {in_path} -> {out_path}')
            filter_md_file(in_path, out_path)


if __name__ == '__main__':
    BASE_DIR = (
        '/home/jovyan/projection/MinerU/material_code/vsrag/MinerU_extract_markdown/mineru_extract_result_markdown'
    )
    process_material_folder(BASE_DIR)
