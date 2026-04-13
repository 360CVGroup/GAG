import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict

import numpy as np

from src.ppr.prototype_router import PrototypeRouter


DEFAULT_LABEL2DOMAIN = {
    0: "general",
    1: "materials",
    2: "adjuvant",
    3: "aviation",
    4: "law",
    5: "math",
}


def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if not data:
        raise RuntimeError(f"No data loaded from {path}")
    return data


def canonicalize_domain(domain_name: str) -> str:
    if domain_name == "material":
        return "materials"
    return domain_name


def eval_router(router: PrototypeRouter, data_path: str, label2domain: Dict[int, str]) -> Dict[str, Any]:
    data = load_jsonl(data_path)
    total = 0
    correct = 0
    per_domain_total = Counter()
    per_domain_correct = Counter()
    confusion = defaultdict(lambda: Counter())

    for example in data:
        question = example.get("question", "").strip()
        label = example.get("label", None)
        if not question or label not in label2domain:
            continue

        gold_domain = canonicalize_domain(label2domain[label])
        pred_domain, _, _ = router.route(question, return_scores=True)
        pred_domain = canonicalize_domain(pred_domain)

        total += 1
        per_domain_total[gold_domain] += 1
        confusion[gold_domain][pred_domain] += 1

        if pred_domain == gold_domain:
            correct += 1
            per_domain_correct[gold_domain] += 1

    if total == 0:
        raise RuntimeError("No valid samples were found for router evaluation.")

    per_domain_acc = {
        domain: per_domain_correct[domain] / per_domain_total[domain]
        for domain in per_domain_total
    }
    return {
        "total": total,
        "overall_acc": correct / total,
        "macro_acc": float(np.mean(list(per_domain_acc.values()))),
        "per_domain_acc": per_domain_acc,
        "confusion": {gold: dict(preds) for gold, preds in confusion.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPR routing accuracy on a mixed-domain JSONL file.")
    parser.add_argument("--encoder_name_or_path", type=str, required=True)
    parser.add_argument("--eval_jsonl", type=str, required=True)
    parser.add_argument("--general_proto", type=str, required=True)
    parser.add_argument("--adjuvant_proto", type=str, default=None)
    parser.add_argument("--materials_proto", type=str, default=None)
    parser.add_argument("--aviation_proto", type=str, default=None)
    parser.add_argument("--law_proto", type=str, default=None)
    parser.add_argument("--math_proto", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--output_report", type=str, default=None)
    args = parser.parse_args()

    prototype_files = {"general": args.general_proto}
    for domain_name, path in {
        "adjuvant": args.adjuvant_proto,
        "materials": args.materials_proto,
        "aviation": args.aviation_proto,
        "law": args.law_proto,
        "math": args.math_proto,
    }.items():
        if path and os.path.isfile(path):
            prototype_files[domain_name] = path

    router = PrototypeRouter(
        encoder_name_or_path=args.encoder_name_or_path,
        prototype_files=prototype_files,
        max_seq_length=args.max_seq_length,
    )

    label2domain = {
        label: domain
        for label, domain in DEFAULT_LABEL2DOMAIN.items()
        if domain in prototype_files
    }
    result = eval_router(router, args.eval_jsonl, label2domain)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_report:
        os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")


if __name__ == "__main__":
    main()
