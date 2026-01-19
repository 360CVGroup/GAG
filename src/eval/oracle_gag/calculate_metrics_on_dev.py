import argparse
import json
import numpy as np
import torch
from bert_score import score as bert_score
from transformers import AutoTokenizer
from tqdm import tqdm

# 检查是否有可用的GPU（保留，不影响逻辑）
device = "cuda" if torch.cuda.is_available() else "cpu"

# 初始化 tokenizer（与原脚本一致）
bert_tokenizer = AutoTokenizer.from_pretrained(
    "/home/jovyan/lirongji-2/models/allenai/scibert_scivocab_uncased"
)


def split_text(text, max_length=500):
    """拆分文本为多个片段，每个片段的标记数不超过 max_length"""
    tokens = bert_tokenizer.tokenize(text)  # 使用模型的 tokenizer

    if len(tokens) <= max_length:
        return [text]

    # 拆分 tokens 并确保每个片段是有效的字符串
    chunks = []
    for i in range(0, len(tokens), max_length):
        chunk_tokens = tokens[i : i + max_length]
        chunk = bert_tokenizer.convert_tokens_to_string(chunk_tokens).strip()
        if chunk:  # 确保不添加空字符串
            chunks.append(chunk)

    return chunks


def evaluate_responses(predictions, references):
    """计算BERTScore（逻辑保持与原脚本完全一致）"""
    weighted_bert_score_sum = 0.0
    total_weight = 0.0
    for pred, ref in zip(predictions, references):
        pred_segments = split_text(pred) if len(bert_tokenizer.tokenize(pred)) > 510 else [pred]
        ref_segments = split_text(ref) if len(bert_tokenizer.tokenize(ref)) > 510 else [ref]

        for pred_seg in pred_segments:
            for ref_seg in ref_segments:
                P, R, F1 = bert_score(
                    [pred_seg],
                    [ref_seg],
                    lang="en",
                    verbose=False,
                    model_type="/home/jovyan/lirongji-2/models/allenai/scibert_scivocab_uncased",
                    num_layers=8,
                )
                score = F1.item()

                pred_length = len(bert_tokenizer.tokenize(pred_seg))
                ref_length = len(bert_tokenizer.tokenize(ref_seg))

                weighted_bert_score_sum += score * (pred_length + ref_length) / 2
                total_weight += (pred_length + ref_length) / 2

    avg_bert_score = weighted_bert_score_sum / total_weight if total_weight > 0 else 0
    return avg_bert_score


def evaluate_scores(input_file, output_file):
    """逐行读取jsonl文件并计算BERTScore分数"""
    json_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                json_object = json.loads(line.strip())
                json_data.append(json_object)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"{line}")

    bert_score_list = []

    with open(output_file, "w", encoding="utf-8") as out_f:
        for data in tqdm(json_data, desc="Processing entries", unit="entry"):
            model_answer = data["Qwen3-8B_answer"]
            ground_truth = data["output"]

            if model_answer != "" and ground_truth != "":
                # 计算BERTScore
                bert_score_result = evaluate_responses([model_answer], [ground_truth])
                bert_score_list.append(bert_score_result)

                data["bertscore"] = bert_score_result
                out_f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # 计算指标平均值
    bert_score_avg = np.mean(bert_score_list) if bert_score_list else 0

    # 输出平均值
    print(f"Overall BERTScore Average: {bert_score_avg}")

    return {
        "BERTScore": bert_score_avg
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default="/home/jovyan/lirongji-2/projection/paper/acl_paper_for_supplementary_materials/GAG/saves/oracle_gag_dev_data/Adjuvant_benchmark_with_answer_embeddding_with_qwen3-8B_answer.jsonl",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/jovyan/lirongji-2/projection/paper/acl_paper_for_supplementary_materials/GAG/saves/oracle_gag_dev_data/Adjuvant_benchmark_with_answer_embeddding_with_qwen3-8B_answer_with_bertscore.jsonl",
    )
    args = parser.parse_args()

    scores = evaluate_scores(args.input_file, args.output_file)
    print(scores)
