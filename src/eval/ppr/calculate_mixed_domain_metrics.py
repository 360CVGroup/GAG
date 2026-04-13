import argparse
import importlib
import json
import os
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import regex
import torch
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

from src.language_modeling.utils import resolve_path


GENERAL_DATASETS = ["webqa", "tqa", "popqa", "freebase_qa", "hotpot_qa", "nq"]
PROFESSIONAL_DATASETS = ["material", "adjuvant_qa"]


class SimpleTokenizer:
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self):
        self._regexp = regex.compile(
            f"({self.ALPHA_NUM})|({self.NON_WS})",
            flags=regex.IGNORECASE | regex.UNICODE | regex.MULTILINE,
        )

    def tokenize(self, text: str, uncased: bool = False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            return [m.group().lower() for m in matches]
        return [m.group() for m in matches]


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def has_answer(answers, text, tokenizer: SimpleTokenizer | None = None) -> bool:
    tokenizer = tokenizer or SimpleTokenizer()
    text = tokenizer.tokenize(_normalize(text), uncased=True)
    for answer in answers:
        answer_tokens = tokenizer.tokenize(_normalize(str(answer)), uncased=True)
        for i in range(0, len(text) - len(answer_tokens) + 1):
            if answer_tokens == text[i : i + len(answer_tokens)]:
                return True
    return False


def patch_bert_score_tokenizer_max_length(max_length: int = 512):
    score_module = importlib.import_module("bert_score.score")
    scorer_module = importlib.import_module("bert_score.scorer")
    original_score_get_tokenizer = score_module.get_tokenizer
    original_scorer_get_tokenizer = scorer_module.get_tokenizer

    def wrapped_get_tokenizer(original_fn, model_type, use_fast=False):
        tokenizer = original_fn(model_type, use_fast=use_fast)
        if getattr(tokenizer, "model_max_length", None) is None or tokenizer.model_max_length > 100000:
            tokenizer.model_max_length = max_length
        return tokenizer

    score_module.get_tokenizer = lambda model_type, use_fast=False: wrapped_get_tokenizer(
        original_score_get_tokenizer,
        model_type,
        use_fast=use_fast,
    )
    scorer_module.get_tokenizer = lambda model_type, use_fast=False: wrapped_get_tokenizer(
        original_scorer_get_tokenizer,
        model_type,
        use_fast=use_fast,
    )


def split_text(text: str, tokenizer, max_length: int = 500):
    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_length:
        return [text]
    chunks = []
    for start in range(0, len(tokens), max_length):
        chunk_tokens = tokens[start : start + max_length]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens).strip()
        if chunk_text:
            chunks.append(chunk_text)
    return chunks


def compute_sts_scores(model, predictions, references, batch_size: int = 128):
    pred_embeddings = model.encode(
        predictions,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    ref_embeddings = model.encode(
        references,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    similarities = torch.nn.functional.cosine_similarity(pred_embeddings, ref_embeddings)
    return similarities.detach().cpu().tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate mixed-domain metrics for PPR-routed GAG outputs.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--summary_json", type=str, required=True)
    parser.add_argument("--summary_tsv", type=str, required=True)
    parser.add_argument("--prediction_key", type=str, default="Qwen3-8B_answer")
    parser.add_argument("--answer_key", type=str, default="answer")
    parser.add_argument("--dataset_key", type=str, default="dataset_name")
    parser.add_argument("--bertscore_model_path", type=str, default="models/scibert_scivocab_uncased")
    parser.add_argument("--sentence_model_path", type=str, default="models/all-mpnet-base-v2")
    parser.add_argument("--bertscore_num_layers", type=int, default=8)
    parser.add_argument("--bertscore_batch_size", type=int, default=64)
    parser.add_argument("--sts_batch_size", type=int, default=128)
    parser.add_argument("--compute_sts", type=eval, default=True)
    parser.add_argument("--workdir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    args.workdir = resolve_path(None, args.workdir or os.getcwd())
    args.input_file = resolve_path(args.workdir, args.input_file)
    args.output_file = resolve_path(args.workdir, args.output_file)
    args.summary_json = resolve_path(args.workdir, args.summary_json)
    args.summary_tsv = resolve_path(args.workdir, args.summary_tsv)
    args.bertscore_model_path = resolve_path(args.workdir, args.bertscore_model_path)
    args.sentence_model_path = resolve_path(args.workdir, args.sentence_model_path)

    patch_bert_score_tokenizer_max_length()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_tokenizer = AutoTokenizer.from_pretrained(args.bertscore_model_path)
    if getattr(bert_tokenizer, "model_max_length", None) is None or bert_tokenizer.model_max_length > 100000:
        bert_tokenizer.model_max_length = 512
    bert_scorer = BERTScorer(
        model_type=args.bertscore_model_path,
        num_layers=args.bertscore_num_layers,
        lang="en",
        device=device,
        use_fast_tokenizer=False,
    )
    sentence_model = None
    if args.compute_sts:
        sentence_model = SentenceTransformer(args.sentence_model_path, device=device)

    with open(args.input_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    summary: dict[str, Any] = {}
    per_dataset_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        per_dataset_records[str(record.get(args.dataset_key, "unknown"))].append(record)

    prof_pair_predictions = []
    prof_pair_references = []
    prof_pair_weights = []
    prof_pair_indices = []
    prof_predictions = []
    prof_references = []
    prof_sts_indices = []
    prof_record_bert_scores: dict[tuple[str, int], float] = {}
    prof_record_sts_scores: dict[tuple[str, int], float] = {}

    enriched_records = []
    tokenizer = SimpleTokenizer()

    for dataset_name, dataset_records in per_dataset_records.items():
        if dataset_name in GENERAL_DATASETS:
            hits = 0
            for idx, record in enumerate(dataset_records):
                prediction = str(record.get(args.prediction_key, "") or "").strip()
                answers = record.get(args.answer_key, [])
                if not isinstance(answers, list):
                    answers = [answers]
                em = 1.0 if prediction and has_answer(answers, prediction, tokenizer=tokenizer) else 0.0
                hits += int(em)
                out = dict(record)
                out["metric_type"] = "EM"
                out["exact_match"] = em
                enriched_records.append(out)
            summary[dataset_name] = {
                "count": len(dataset_records),
                "metric_type": "EM",
                "EM": hits / len(dataset_records) if dataset_records else 0.0,
            }
        elif dataset_name in PROFESSIONAL_DATASETS:
            for idx, record in enumerate(dataset_records):
                prediction = str(record.get(args.prediction_key, "") or "").strip()
                reference = record.get(args.answer_key, "")
                if isinstance(reference, list):
                    reference = " ".join(str(x) for x in reference)
                reference = str(reference).strip()
                if not prediction or not reference:
                    prof_record_bert_scores[(dataset_name, idx)] = 0.0
                    if args.compute_sts:
                        prof_record_sts_scores[(dataset_name, idx)] = 0.0
                    continue
                prof_predictions.append(prediction)
                prof_references.append(reference)
                prof_sts_indices.append((dataset_name, idx))
                pred_segments = split_text(prediction, bert_tokenizer)
                ref_segments = split_text(reference, bert_tokenizer)
                for pred_segment in pred_segments:
                    pred_length = len(bert_tokenizer.tokenize(pred_segment))
                    for ref_segment in ref_segments:
                        ref_length = len(bert_tokenizer.tokenize(ref_segment))
                        prof_pair_predictions.append(pred_segment)
                        prof_pair_references.append(ref_segment)
                        prof_pair_weights.append((pred_length + ref_length) / 2)
                        prof_pair_indices.append((dataset_name, idx))
            # add later after BERTScore computation
        else:
            for record in dataset_records:
                out = dict(record)
                out["metric_type"] = "unknown"
                enriched_records.append(out)
            summary[dataset_name] = {
                "count": len(dataset_records),
                "metric_type": "unknown",
            }

    if prof_pair_predictions:
        _, _, f1 = bert_scorer.score(
            prof_pair_predictions,
            prof_pair_references,
            batch_size=args.bertscore_batch_size,
        )
        weighted_sum = defaultdict(float)
        total_weight = defaultdict(float)
        for key, score, weight in zip(prof_pair_indices, f1.detach().cpu().tolist(), prof_pair_weights):
            weighted_sum[key] += score * weight
            total_weight[key] += weight
        for key, value in weighted_sum.items():
            prof_record_bert_scores[key] = value / total_weight[key] if total_weight[key] > 0 else 0.0

    if sentence_model is not None and prof_predictions:
        sts_scores = compute_sts_scores(
            sentence_model,
            prof_predictions,
            prof_references,
            batch_size=args.sts_batch_size,
        )
        for key, value in zip(prof_sts_indices, sts_scores):
            prof_record_sts_scores[key] = value

    for dataset_name in PROFESSIONAL_DATASETS:
        dataset_records = per_dataset_records.get(dataset_name, [])
        bert_scores = []
        sts_scores = []
        for idx, record in enumerate(dataset_records):
            bert_score = prof_record_bert_scores.get((dataset_name, idx), 0.0)
            sts_score = prof_record_sts_scores.get((dataset_name, idx), 0.0)
            out = dict(record)
            out["metric_type"] = "professional"
            out["bertscore"] = bert_score
            if args.compute_sts:
                out["sts_score"] = sts_score
            enriched_records.append(out)
            bert_scores.append(bert_score)
            if args.compute_sts:
                sts_scores.append(sts_score)
        summary[dataset_name] = {
            "count": len(dataset_records),
            "metric_type": "professional",
            "BERTScore": float(np.mean(bert_scores)) if bert_scores else 0.0,
        }
        if args.compute_sts:
            summary[dataset_name]["STS_Score"] = float(np.mean(sts_scores)) if sts_scores else 0.0

    ordered_names = PROFESSIONAL_DATASETS + GENERAL_DATASETS + sorted(
        name for name in per_dataset_records.keys() if name not in {*PROFESSIONAL_DATASETS, *GENERAL_DATASETS}
    )

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for record in enriched_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump({name: summary[name] for name in ordered_names if name in summary}, f, ensure_ascii=False, indent=2)
        f.write("\n")

    Path(args.summary_tsv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_tsv, "w", encoding="utf-8") as f:
        f.write("dataset_name\tmetric_type\tcount\tvalue\n")
        for name in ordered_names:
            if name not in summary:
                continue
            stats = summary[name]
            if stats["metric_type"] == "EM":
                f.write(f"{name}\tEM\t{stats['count']}\t{stats['EM']}\n")
            elif stats["metric_type"] == "professional":
                f.write(f"{name}\tBERTScore\t{stats['count']}\t{stats['BERTScore']}\n")
                if args.compute_sts:
                    f.write(f"{name}\tSTS_Score\t{stats['count']}\t{stats['STS_Score']}\n")
            else:
                f.write(f"{name}\tunknown\t{stats['count']}\t\n")

    print(f"[Done] Saved mixed-domain metrics to {args.summary_json}")
    print(f"[Done] Saved mixed-domain TSV summary to {args.summary_tsv}")


if __name__ == "__main__":
    main()
