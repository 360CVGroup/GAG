#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
逐数据集统计：
- adjuvant_qa / material: BERTScore（SciBERT）
- freebase_qa / hotpot_qa / nq / tqa / webqa / popqa: EM

注意：EM / BERTScore(adjuvant_qa, material) 的实现完全照你的原版，不改动。
"""

import os
import sys
import json
import argparse
import unicodedata
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import regex  # 用于 EM 分词
from transformers import AutoConfig  # 虽然当前未使用，保留以做最小改动

# =======================
# ==== EM（原样拷贝）====
# =======================

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

def _normalize(text):
    return unicodedata.normalize('NFD', text)

def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False

def get_substring_match_score(outputs,answers):
    """
    outputs: [string1,string2]
    answers: [
                [string1_1,string1_2],
                [string2_1,string2_2]
             ]
    """
    import numpy as np
    assert len(outputs) == len(answers)
    if not isinstance(answers[0],list):
        answers = [[x] for x in answers]
    substring_match_scores = []
    answer_lengths = []
    for output,answer in zip(outputs,answers):
        if has_answer(answer,output): # EM evaluation
            substring_match_scores.append(1.0)
        else:
            substring_match_scores.append(0.0)

        answer_lengths.append(len(output.split()))

    substring_match = round(sum(substring_match_scores)/len(outputs), 4)
    lens = round(np.mean(answer_lengths), 4)

    return substring_match,substring_match_scores


# ==========================================================
# ==== BERTScore（原样拷贝，路径保持一致）====
# ==========================================================
import argparse as _argparse_shadow  # 仅为保持环境一致，无实质用途
import os as _os_shadow
import json as _json_shadow
import numpy as _np_shadow
import torch
from bert_score import score as bert_score
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm as _tqdm_shadow

# 检查是否有可用的GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化 tokenizer（与你的实现保持完全一致，用于 adjuvant_qa / material）
bert_tokenizer = AutoTokenizer.from_pretrained("/home/jovyan/lirongji-2/models/allenai/scibert_scivocab_uncased")

def split_text(text, max_length=500):
    """拆分文本为多个片段，每个片段的标记数不超过 max_length"""
    tokens = bert_tokenizer.tokenize(text)  # 使用模型的 tokenizer

    if len(tokens) <= max_length:
        return [text]

    # 拆分 tokens 并确保每个片段是有效的字符串
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk_tokens = tokens[i:i + max_length]
        chunk = bert_tokenizer.convert_tokens_to_string(chunk_tokens).strip()
        if chunk:  # 确保不添加空字符串
            chunks.append(chunk)

    return chunks

def evaluate_responses(predictions, references):
    """计算BERTScore（用于 adjuvant_qa / material，SciBERT）"""
    weighted_bert_score_sum = 0.0
    total_weight = 0.0
    for pred, ref in zip(predictions, references):
        pred_segments = split_text(pred) if len(bert_tokenizer.tokenize(pred)) > 510 else [pred]
        ref_segments = split_text(ref) if len(bert_tokenizer.tokenize(ref)) > 510 else [ref]

        for pred_seg in pred_segments:
            for ref_seg in ref_segments:
                P, R, F1 = bert_score([pred_seg], [ref_seg], lang="en",
                                      verbose=False, model_type="/home/jovyan/lirongji-2/models/allenai/scibert_scivocab_uncased", num_layers=8)
                score = F1.item()

                pred_length = len(bert_tokenizer.tokenize(pred_seg))
                ref_length = len(bert_tokenizer.tokenize(ref_seg))

                weighted_bert_score_sum += score * (pred_length + ref_length) / 2
                total_weight += (pred_length + ref_length) / 2

    avg_bert_score = weighted_bert_score_sum / total_weight if total_weight > 0 else 0
    return avg_bert_score


# =======================
# ROUGE-L F1（仍保留函数，但当前未使用）
# =======================

def _lcs_length(xs: List[str], ys: List[str]) -> int:
    """计算两个 token 序列的 LCS 长度（动态规划）"""
    m, n = len(xs), len(ys)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        xi = xs[i - 1]
        for j in range(1, n + 1):
            if xi == ys[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1]
    return dp[m][n]

def rouge_l_f1(pred: str, ref: str) -> float:
    """
    简单实现 token 级 ROUGE-L F1：
      - 按空格分词
      - 忽略大小写
    """
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    prec = lcs / len(pred_tokens)
    rec = lcs / len(ref_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# =======================
# 主流程：逐数据集聚合
# =======================
def main():
    ap = argparse.ArgumentParser(
        description="逐数据集统计：adjuvant_qa/material=BERTScore(SciBERT)；其余=EM（包含popqa）。"
    )
    ap.add_argument(
        "--input",
        type=str,
        default="/home/jovyan/lirongji-2/projection/paper/acl_paper_for_supplementary_materials/GAG/saves/gag_dev_data/3_mixed_domains_gag_eval_result.jsonl",
        help="推理结果 JSONL（包含 dataset_name / answer / Qwen3-8B_answer）"
    )
    ap.add_argument("--pred_key",   type=str, default="Qwen3_answer")
    ap.add_argument("--answer_key", type=str, default="answer")
    ap.add_argument(
        "--save_summary",
        type=str,
        default="/home/jovyan/lirongji-2/projection/paper/acl_paper_for_supplementary_materials/GAG/saves/gag_dev_data/3_mixed_domains_gag_eval_result_save_summary.json"
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save_summary), exist_ok=True)

    # 读入
    records: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as fin:
        for ln, line in enumerate(fin, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                records.append(obj)
            except Exception as e:
                print(f"[WARN] 第 {ln} 行解析失败：{e}", file=sys.stderr)

    if not records:
        print("[ERROR] 输入为空。")
        return

    # 分桶
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for obj in records:
        ds = obj.get("dataset_name", "")
        buckets.setdefault(ds, []).append(obj)

    DATASETS = [
        "adjuvant_qa",
        "material",      # 新增
        "freebase_qa",
        "hotpot_qa",
        "nq",
        "tqa",
        "webqa",
        "popqa",
    ]

    summary: Dict[str, Any] = {}

    # ===== adjuvant_qa / material：仅 BERTScore =====
    for ds in ["adjuvant_qa", "material"]:
        prof = buckets.get(ds, [])
        bsc_list = []
        for obj in tqdm(prof, desc=f"[{ds}] BERTScore", ncols=100):
            pred = obj.get(args.pred_key, "") or ""
            gt = obj.get(args.answer_key, "")
            gt_text = gt if isinstance(gt, str) else " ".join([str(x) for x in gt]) if isinstance(gt, list) else str(gt)
            if pred.strip() and gt_text.strip():
                bsc = evaluate_responses([pred], [gt_text])
                bsc_list.append(bsc)
        summary[ds] = {
            "count": len(prof),
            "count_scored": len(bsc_list),
            "bertscore_mean": float(np.mean(bsc_list)) if bsc_list else 0.0,
        }

    # ===== 其余数据集：EM（保持原函数），新增 popqa =====
    def em_for_bucket(name: str):
        arr = buckets.get(name, [])
        em_scores = []
        for obj in tqdm(arr, desc=f"[{name}] EM", ncols=100):
            pred = obj.get(args.pred_key, "") or ""
            gt = obj.get(args.answer_key, "")
            if isinstance(gt, list):
                answers_list = [str(x) for x in gt]
            elif isinstance(gt, str) and gt != "":
                answers_list = [gt]
            else:
                answers_list = []
            _, em_list = get_substring_match_score([pred], [answers_list])  # 原函数
            em_scores.append(float(int(em_list[0])))
        return {
            "count": len(arr),
            "em_mean": float(np.mean(em_scores)) if em_scores else 0.0
        }

    for ds in ["freebase_qa", "hotpot_qa", "nq", "tqa", "webqa", "popqa"]:
        summary[ds] = em_for_bucket(ds)

    # ===== 打印汇总 =====
    print("\n=========== Per-dataset Metrics ===========")
    for ds in DATASETS:
        if ds not in summary:
            print(f"{ds}: <no records>")
            continue
        stats = summary[ds]
        if ds in ["adjuvant_qa", "material"]:
            print(f"{ds:15s}  count={stats['count']:6d}  scored={stats['count_scored']:6d}  "
                  f"BERTScore={stats['bertscore_mean']:.4f}")
        else:
            print(f"{ds:15s}  count={stats['count']:6d}  EM={stats['em_mean']:.4f}")

    with open(args.save_summary, "w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
    print(f"\n[OK] Summary saved to: {args.save_summary}")


if __name__ == "__main__":
    main()
