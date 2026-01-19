# src/router/build_domain_prototypes.py
import os
import json
import argparse
import random
from typing import List

import torch
import numpy as np
from sklearn.cluster import KMeans

from .query_encoder import QueryEncoder


def load_questions_from_jsonl(path: str, question_field: str = "question") -> List[str]:
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            q = obj.get(question_field, "").strip()
            if q:
                qs.append(q)
    if not qs:
        raise RuntimeError(f"No questions loaded from {path}")
    return qs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_name_or_path", type=str, default="/home/jovyan/lirongji-2/models/Qwen3-1.7B")
    parser.add_argument("--input_jsonl", type=str, default="/home/jovyan/lirongji-2/projection/paper/acl_paper_for_supplementary_materials/GAG/data/build_prototypes/adjuvant_mini.jsonl")
    parser.add_argument("--output_path", type=str, default="/home/jovyan/lirongji-2/projection/paper/acl_paper_for_supplementary_materials/GAG/saves/prototypes/adjuvant_prototypes.pt")
    parser.add_argument("--domain_name", type=str, default="adjuvant")
    parser.add_argument("--question_field", type=str, default="question")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--num_prototypes", type=int, default=32)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=980406)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[Load] questions from {args.input_jsonl}")
    questions = load_questions_from_jsonl(args.input_jsonl, question_field=args.question_field)
    print(f"  total questions = {len(questions)}")

    if len(questions) > args.max_samples:
        print(f"  subsampling to {args.max_samples} for clustering")
        questions = random.sample(questions, args.max_samples)

    print(f"[Init] QueryEncoder from {args.encoder_name_or_path}")
    encoder = QueryEncoder(
        encoder_name_or_path=args.encoder_name_or_path,
        max_seq_length=args.max_seq_length,
    )

    print(f"[Encode] {len(questions)} queries...")
    reps = encoder.encode_batch(questions, batch_size=args.batch_size)   # [N, H]
    print(f"  reps.shape = {tuple(reps.shape)}")

    # L2-normalize
    reps_np = reps.numpy()
    norms = np.linalg.norm(reps_np, axis=1, keepdims=True) + 1e-6
    reps_np = reps_np / norms

    n_clusters = min(args.num_prototypes, reps_np.shape[0])
    print(f"[KMeans] n_clusters = {n_clusters}")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=args.seed,
        n_init=10,
    )
    kmeans.fit(reps_np)
    centers = kmeans.cluster_centers_  # [C, H]
    centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-6)

    prototypes = torch.tensor(centers, dtype=torch.float32)
    out_obj = {
        "domain_name": args.domain_name,
        "encoder_name_or_path": args.encoder_name_or_path,
        "hidden_size": int(prototypes.shape[1]),
        "num_prototypes": int(prototypes.shape[0]),
        "prototypes": prototypes,  # [C, H], L2-normalized
    }

    torch.save(out_obj, args.output_path)
    print(f"[OK] saved prototypes to {args.output_path}")
    print(f"  domain={args.domain_name}, num_prototypes={out_obj['num_prototypes']}, hidden_size={out_obj['hidden_size']}")


if __name__ == "__main__":
    main()


# # 通用领域
# python -m src.ppr.build_domain_prototypes \
#   --encoder_name_or_path /home/jovyan/lirongji-2/models/Qwen3-1.7B \
#   --input_jsonl /path/to/general_qa_mix.jsonl \
#   --domain_name general \
#   --output_path /path/to/prototypes/general_prototypes.pt

# # 佐剂
# python -m src.ppr.build_domain_prototypes \
#   --encoder_name_or_path /home/jovyan/lirongji-2/models/Qwen3-1.7B \
#   --input_jsonl /path/to/adjuvant_train.jsonl \
#   --domain_name adjuvant \
#   --output_path /path/to/prototypes/adjuvant_prototypes.pt

# # 材料
# python -m src.ppr.build_domain_prototypes \
#   --encoder_name_or_path /home/jovyan/lirongji-2/models/Qwen3-1.7B \
#   --input_jsonl /path/to/material_train.jsonl \
#   --domain_name material \
#   --output_path /path/to/prototypes/material_prototypes.pt


# python -m src.ppr.build_domain_prototypes --input_jsonl /home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/eval_PPR_6_domains/processed_dataset/aviationQA.jsonl --output_path /home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/gate_weight/qwen3_1_7B_version/aviation_prototypes.pt --domain_name aviation


# python -m src.ppr.build_domain_prototypes --input_jsonl /home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/eval_PPR_6_domains/processed_dataset/housingqa_train.jsonl --output_path /home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/gate_weight/qwen3_1_7B_version/law_prototypes.pt --domain_name law


# python -m src.ppr.build_domain_prototypes --input_jsonl /home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/eval_PPR_6_domains/processed_dataset/gsm8k.jsonl --output_path /home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/gate_weight/qwen3_1_7B_version/math_prototypes.pt --domain_name math