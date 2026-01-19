# src/router/eval_router_accuracy.py
import os
import json
import argparse
from collections import Counter, defaultdict
from typing import Dict, Any

import torch
import numpy as np

from src.ppr.prototype_router import PrototypeRouter

"""
全局 label → 领域名 映射（最多 6 域）

目前的数据约定：
  0: general
  1: adjuvant
  2: material
  3: aviation
  4: law
  5: math
"""

GLOBAL_LABEL2DOMAIN = {
    0: "general",
    1: "adjuvant",
    2: "material",
    3: "aviation",
    4: "law",
    5: "math",
}


def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(obj)
    if not data:
        raise RuntimeError(f"No data loaded from {path}")
    return data


def eval_router(
    router: PrototypeRouter,
    data_path: str,
    label2domain: Dict[int, str],
) -> Dict[str, Any]:
    data = load_jsonl(data_path)

    total = 0
    correct = 0

    per_domain_total = Counter()
    per_domain_correct = Counter()
    confusion = defaultdict(lambda: Counter())

    for ex in data:
        q = ex.get("question", "").strip()
        if q == "":
            continue
        label = ex.get("label", None)
        if label is None:
            continue

        # 只评测那些在 label2domain 映射里的样本
        if label not in label2domain:
            continue

        gold_domain = label2domain[label]
        pred_domain, score, _ = router.route(q)  # 不带 margin 的主方法

        total += 1
        per_domain_total[gold_domain] += 1
        confusion[gold_domain][pred_domain] += 1

        if pred_domain == gold_domain:
            correct += 1
            per_domain_correct[gold_domain] += 1

    if total == 0:
        raise RuntimeError("No valid samples for evaluation.")

    overall_acc = correct / total

    per_domain_acc = {
        d: (per_domain_correct[d] / per_domain_total[d]) if per_domain_total[d] > 0 else 0.0
        for d in per_domain_total.keys()
    }

    # macro 平均
    macro_acc = np.mean(list(per_domain_acc.values()))

    # 把 confusion matrix 转成普通 dict
    confusion_dict = {
        gold: dict(pred_cnt)
        for gold, pred_cnt in confusion.items()
    }

    result = {
        "total": total,
        "overall_acc": overall_acc,
        "macro_acc": macro_acc,
        "per_domain_acc": per_domain_acc,
        "confusion": confusion_dict,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    # 路由用的 encoder（你现在是用 full_weight 版本）
    ap.add_argument(
        "--encoder_name_or_path",
        type=str,
        default="/home/jovyan/lirongji-2/models/Qwen3-1.7B",
        help="用于路由的 QueryEncoder backbone（建议用基座 full_weight）",
    )

    # 各领域 prototype 路径：存在就加载，不存在就不加入该领域
    ap.add_argument("--general_proto", type=str, default="/home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/gate_weight/qwen3_1_7B_version/general_prototypes.pt",
                    help="general 领域 prototype .pt 路径（必须提供）")
    ap.add_argument("--adjuvant_proto", type=str, default="/home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/gate_weight/qwen3_1_7B_version/adjuvant_prototypes.pt",
                    help="adjuvant 领域 prototype .pt 路径（必须提供）")
    ap.add_argument("--material_proto", type=str, default=None,
                    help="material 领域 prototype .pt 路径（可选）")
    ap.add_argument("--aviation_proto", type=str, default=None,
                    help="aviation 领域 prototype .pt 路径（可选）")
    ap.add_argument("--law_proto", type=str, default=None,
                    help="law 领域 prototype .pt 路径（可选）")
    ap.add_argument("--math_proto", type=str, default=None,
                    help="math 领域 prototype .pt 路径（可选）")

    # 评测集
    ap.add_argument(
        "--eval_jsonl",
        type=str,
        default="/home/jovyan/lirongji-2/projection/paper/acl_paper_for_supplementary_materials/GAG/data/gag_eval/general_adjuvant_materials_aviation_law_math_mini.jsonl",
    )
    ap.add_argument("--max_seq_length", type=int, default=1024)

    # 可选：带 margin 的路由
    ap.add_argument("--use_margin", action="store_true", help="是否用 route_with_margin 做 ablation")
    ap.add_argument("--margin", type=float, default=0.0)

    # 结果输出
    ap.add_argument(
        "--output_report",
        type=str,
        default="/home/jovyan/lirongji-2/projection/paper/acl_paper_for_supplementary_materials/GAG/data/gag_eval/general_adjuvant_materials_aviation_law_math_mini_router_eval_report.json",
        help="结果写入 json 的路径",
    )

    args = ap.parse_args()

    # 1) 构造 prototype_files 字典（可插拔领域）
    proto_files = {}

    # 必须包含 general 和 adjuvant
    if not os.path.isfile(args.general_proto):
        raise FileNotFoundError(f"general_proto not found: {args.general_proto}")
    if not os.path.isfile(args.adjuvant_proto):
        raise FileNotFoundError(f"adjuvant_proto not found: {args.adjuvant_proto}")

    proto_files["general"] = args.general_proto
    proto_files["adjuvant"] = args.adjuvant_proto

    # 其余领域是可选的，有路径且文件存在就加入
    if args.material_proto is not None and os.path.isfile(args.material_proto):
        proto_files["material"] = args.material_proto
    if args.aviation_proto is not None and os.path.isfile(args.aviation_proto):
        proto_files["aviation"] = args.aviation_proto
    if args.law_proto is not None and os.path.isfile(args.law_proto):
        proto_files["law"] = args.law_proto
    if args.math_proto is not None and os.path.isfile(args.math_proto):
        proto_files["math"] = args.math_proto

    print("[Router] loading prototypes for domains:")
    for d, p in proto_files.items():
        print(f"  - {d}: {p}")

    # 2) 初始化路由器（会把这些领域的 prototype 全部加载进来）
    router = PrototypeRouter(
        encoder_name_or_path=args.encoder_name_or_path,
        prototype_files=proto_files,
        max_seq_length=args.max_seq_length,
    )

    # 3) 根据“哪些领域加载了 prototype”，构造 label→domain 映射
    #    只保留那些在 proto_files 中出现的领域，对其它 label 的样本直接跳过。
    label2domain = {
        label: domain
        for label, domain in GLOBAL_LABEL2DOMAIN.items()
        if domain in proto_files
    }

    print("[Eval] label → domain 映射：")
    for lb, dm in sorted(label2domain.items()):
        print(f"  label {lb} -> {dm}")

    # 4) 选择是否带 margin
    if not args.use_margin:
        print("[Eval] using route() (no margin)")
        result = eval_router(router, args.eval_jsonl, label2domain)
    else:
        print(f"[Eval] using route_with_margin(), margin={args.margin}")

        def eval_with_margin(router, data_path, label2domain, margin):
            data = load_jsonl(data_path)
            total = 0
            correct = 0
            per_domain_total = Counter()
            per_domain_correct = Counter()

            for ex in data:
                q = ex.get("question", "").strip()
                label = ex.get("label", None)
                if q == "" or label not in label2domain:
                    continue
                gold_domain = label2domain[label]
                pred_domain, score, _ = router.route_with_margin(
                    q, general_domain_name="general", margin=margin
                )
                total += 1
                per_domain_total[gold_domain] += 1
                if pred_domain == gold_domain:
                    correct += 1
                    per_domain_correct[gold_domain] += 1

            if total == 0:
                raise RuntimeError("No valid samples for evaluation (margin).")

            overall_acc = correct / total
            per_domain_acc = {
                d: (per_domain_correct[d] / per_domain_total[d]) if per_domain_total[d] > 0 else 0.0
                for d in per_domain_total.keys()
            }
            macro_acc = np.mean(list(per_domain_acc.values()))
            return {
                "total": total,
                "overall_acc": overall_acc,
                "macro_acc": macro_acc,
                "per_domain_acc": per_domain_acc,
                "margin": margin,
            }

        result = eval_with_margin(router, args.eval_jsonl, label2domain, args.margin)

    print("\n===== Router Evaluation =====")
    for k, v in result.items():
        print(f"{k}: {v}")

    # 5) 写报告
    if args.output_report is not None:
        os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[OK] report written to {args.output_report}")


if __name__ == "__main__":
    main()


# python -m src.eval.gag.eval_ppr_accuracy_6domains \
#   --encoder_name_or_path /home/jovyan/lirongji-2/models/Qwen3-1.7B \
#   --general_proto /home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/gate_weight/qwen3_1_7B_version/general_prototypes.pt \
#   --adjuvant_proto /home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/gate_weight/qwen3_1_7B_version/adjuvant_prototypes.pt \
#   --material_proto /home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/gate_weight/qwen3_1_7B_version/material_prototypes.pt \
#   --aviation_proto /home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/gate_weight/qwen3_1_7B_version/aviation_prototypes.pt \
#   --law_proto /home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/gate_weight/qwen3_1_7B_version/law_prototypes.pt \
#   --math_proto /home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/prototype_based_plug_and_play_router_11_30/gate_weight/qwen3_1_7B_version/math_prototypes.pt \
#   --eval_jsonl data/gag_eval/general_adjuvant_materials_aviation_law_math_mini.jsonl \
#   --output_report data/gag_eval/general_adjuvant_materials_aviation_law_math_mini_router_eval_report.json
