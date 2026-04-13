import argparse
import importlib
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

from src.language_modeling.utils import get_yaml_file, resolve_path


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--prediction_key", type=str, default=None)
    parser.add_argument("--reference_key", type=str, default=None)
    parser.add_argument("--compute_sts", type=eval, default=None)
    parser.add_argument("--bertscore_model_path", type=str, default=None)
    parser.add_argument("--sentence_model_path", type=str, default=None)
    parser.add_argument("--bertscore_num_layers", type=int, default=None)
    parser.add_argument("--bertscore_batch_size", type=int, default=None)
    parser.add_argument("--sts_batch_size", type=int, default=None)
    args = parser.parse_args()

    if args.config:
        yaml_config = get_yaml_file(args.config)
        for key, value in yaml_config.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)

    args.workdir = resolve_path(None, args.workdir or (os.path.dirname(os.path.abspath(args.config)) if args.config else os.getcwd()))
    args.input_file = resolve_path(args.workdir, args.input_file)
    args.output_file = resolve_path(args.workdir, args.output_file)
    args.bertscore_model_path = resolve_path(args.workdir, args.bertscore_model_path)
    args.sentence_model_path = resolve_path(args.workdir, args.sentence_model_path)
    if args.compute_sts is None:
        args.compute_sts = True
    if args.compute_sts and not args.sentence_model_path:
        raise ValueError("compute_sts=True requires sentence_model_path")
    if args.bertscore_batch_size is None:
        args.bertscore_batch_size = 64
    if args.sts_batch_size is None:
        args.sts_batch_size = 128
    return args


def split_text(text, tokenizer, max_length=500):
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


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch_bert_score_tokenizer_max_length()

    bert_tokenizer = AutoTokenizer.from_pretrained(args.bertscore_model_path)
    if getattr(bert_tokenizer, "model_max_length", None) is None or bert_tokenizer.model_max_length > 100000:
        # Some local scientific tokenizers inherit an "infinite" max length sentinel
        # that overflows downstream Rust tokenizers in bert-score. We only score short
        # chunks anyway, so clamping avoids the overflow without changing semantics.
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

    with open(args.input_file, "r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]

    valid_indices = []
    predictions = []
    references = []
    record_bert_scores = {}
    record_sts_scores = {}

    for index, record in enumerate(records):
        prediction = record.get(args.prediction_key, "").strip()
        reference = record.get(args.reference_key, "").strip()
        if not prediction or not reference:
            continue
        valid_indices.append(index)
        predictions.append(prediction)
        references.append(reference)

    pair_predictions = []
    pair_references = []
    pair_weights = []
    pair_record_indices = []

    for local_idx, (prediction, reference) in enumerate(zip(predictions, references)):
        pred_segments = split_text(prediction, bert_tokenizer)
        ref_segments = split_text(reference, bert_tokenizer)

        for pred_segment in pred_segments:
            pred_length = len(bert_tokenizer.tokenize(pred_segment))
            for ref_segment in ref_segments:
                ref_length = len(bert_tokenizer.tokenize(ref_segment))
                pair_predictions.append(pred_segment)
                pair_references.append(ref_segment)
                pair_weights.append((pred_length + ref_length) / 2)
                pair_record_indices.append(local_idx)

    bert_scores = []
    if pair_predictions:
        _, _, f1 = bert_scorer.score(
            pair_predictions,
            pair_references,
            batch_size=args.bertscore_batch_size,
        )
        weighted_sum = defaultdict(float)
        total_weight = defaultdict(float)
        for idx, score, weight in zip(pair_record_indices, f1.detach().cpu().tolist(), pair_weights):
            weighted_sum[idx] += score * weight
            total_weight[idx] += weight

        for idx in range(len(predictions)):
            bert_value = weighted_sum[idx] / total_weight[idx] if total_weight[idx] > 0 else 0.0
            record_bert_scores[idx] = bert_value
            bert_scores.append(bert_value)

    sts_scores = []
    if sentence_model is not None and predictions:
        sts_scores = compute_sts_scores(
            sentence_model,
            predictions,
            references,
            batch_size=args.sts_batch_size,
        )
        for idx, value in enumerate(sts_scores):
            record_sts_scores[idx] = value

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, "w", encoding="utf-8") as output:
        for local_idx, record_idx in enumerate(tqdm(valid_indices, desc="Writing metrics", ncols=100)):
            record = records[record_idx]
            record["bertscore"] = record_bert_scores.get(local_idx, 0.0)
            if sentence_model is not None:
                record["sts_score"] = record_sts_scores.get(local_idx, 0.0)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "BERTScore": float(np.mean(bert_scores)) if bert_scores else 0.0,
    }
    if sentence_model is not None:
        summary["STS_Score"] = float(np.mean(sts_scores)) if sts_scores else 0.0
    print(summary)


if __name__ == "__main__":
    main()
