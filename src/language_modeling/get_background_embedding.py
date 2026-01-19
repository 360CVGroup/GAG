import os
import json
import pickle
import torch
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------- 配置 ----------
MODEL_PATH = "/home/jovyan/lirongji/projection/llama_factory_finetune_projection/llamafactory6.29/LLaMA-Factory/LLaMA-Factory/saves/qwen3-1.7B/full/new_traindata_fullfinetune_qa_finetune_Adjuvant_QA_lr5e-6/checkpoint-1300"
INPUT_JSONL = "/home/jovyan/lirongji-2/projection/MinerU/adjuvant_code/7.20_pipeline_code/train_mlp_8_15/get_small_model_answer_tokens_embedding/original_train_jsonl_files/Adjuvant_QA_unimodal_withoutocr_pure_QA_shuffle_with_id_mini.jsonl"
OUTPUT_PKL = "/home/jovyan/lirongji-2/projection/paper/acl_paper_for_supplementary_materials/GAG/saves/train_dataset_background_embedding/Adjuvant_QA_with_answer_embeddding_different_layers.pkl"
MAX_NEW_TOKENS = 1004
TEMPERATURE = 0.7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

instruction = "You are an expert in immunology and adjuvant, with a strong background in vaccine development. Your research and practice in this field have equipped you with a deep understanding of the mechanisms of immune response and how to optimize vaccine efficacy through adjuvants. Please answer the following questions."
# ---------------------------


# 构建 Chat Prompt
def build_prompt_with_chat_template(query: str, tokenizer) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an expert in immunology and adjuvant, with a strong background in vaccine development. Your research and practice in this field have equipped you with a deep understanding of the mechanisms of immune response and how to optimize vaccine efficacy through adjuvants. You excel in providing relevant and professional background knowledge that can help answer the question. Please provide the background knowledge related to the following question.\n"
        },
        {
            "role": "user",
            "content": f"{query}\n"
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

def count_pickled_objects(path):
    count = 0
    if not os.path.exists(path):
        return count
    with open(path, "rb") as f:
        while True:
            try:
                pickle.load(f)
                count += 1
            except EOFError:
                break
    return count

with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

# 加载模型和 tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE)
model.eval()

# —— 断点恢复：读取已处理记录的 ID —— #
# processed_ids = set()
# if os.path.exists(OUTPUT_PKL):
#     print(f"▸ Found existing output file `{OUTPUT_PKL}`; loading processed IDs for resume...")
#     with open(OUTPUT_PKL, 'rb') as f:
#         while True:
#             try:
#                 rec = pickle.load(f)
#                 processed_ids.add(rec['id'])
#             except EOFError:
#                 break
#     print(f"▸ {len(processed_ids)} records already processed; will skip them.")

# —— 断点恢复：按条数恢复 —— #
processed_count = count_pickled_objects(OUTPUT_PKL)
if processed_count > 0:
    print(f"▸ Found {processed_count} processed records; will resume from line {processed_count+1}.")

# 打开输出文件，追加写入
output_f = open(OUTPUT_PKL, 'ab')

# 推理并提取 embedding

with open(INPUT_JSONL, 'r', encoding='utf-8') as infile:
    for idx, line in enumerate(tqdm(infile, total = total_lines, desc="Processing", ncols = 100, unit = "sample")):
        if idx < processed_count:
            continue  # 跳过已处理条数
        item = json.loads(line)
        rec_id = item["id"]
        # if rec_id in processed_ids:
        #     continue  # 已处理，跳过
        query = item["input"]
        prompt = build_prompt_with_chat_template(query, tokenizer)    # '<|im_start|>system\nYou are an expert in immunology and adjuvant, with a strong background in vaccine development. Your research and practice in this field have equipped you with a deep understanding of the mechanisms of immune response and how to optimize vaccine efficacy through adjuvants. You excel in providing relevant and professional background knowledge that can help answer the question. Please provide the background knowledge related to the following question.\n<|im_end|>\n<|im_start|>user\nHow does the HSV-1 amplicon vector promote the recruitment and activation of antigen-presenting cells?\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(DEVICE)
        input_token_count = inputs["input_ids"].shape[1]     # 117   # 120  # 120  说明它不会加eos_token这些

        with torch.no_grad():
            outputs = model.generate_to_get_answer_embedding_different_layers(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                return_dict_in_generate=True,
                output_hidden_states=True,
                temperature=TEMPERATURE,
                top_p=0.8,
                top_k=20,
                # do_sample=False,
                use_cache=False
            )

# model:
# Qwen3ForCausalLM(
#   (model): Qwen3Model(
#     (embed_tokens): Embedding(151936, 2048)
#     (layers): ModuleList(
#       (0-27): 28 x Qwen3DecoderLayer(
#         (self_attn): Qwen3Attention(
#           (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
#           (k_proj): Linear(in_features=2048, out_features=1024, bias=False)
#           (v_proj): Linear(in_features=2048, out_features=1024, bias=False)
#           (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
#           (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
#           (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
#         )
#         (mlp): Qwen3MLP(
#           (gate_proj): Linear(in_features=2048, out_features=6144, bias=False)
#           (up_proj): Linear(in_features=2048, out_features=6144, bias=False)
#           (down_proj): Linear(in_features=6144, out_features=2048, bias=False)
#           (act_fn): SiLU()
#         )
#         (input_layernorm): Qwen3RMSNorm((2048,), eps=1e-06)
#         (post_attention_layernorm): Qwen3RMSNorm((2048,), eps=1e-06)
#       )
#     )
#     (norm): Qwen3RMSNorm((2048,), eps=1e-06)
#     (rotary_emb): Qwen3RotaryEmbedding()
#   )
#   (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
# )


        # 解码回答文本
        full_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        answer_text = tokenizer.decode(outputs.sequences[0][input_token_count:], skip_special_tokens=True).strip()

        # 提取最后一层 hidden state
        # last_hidden = outputs.hidden_states[-1]    # shape: (1, seq_len, hidden_size)
        # last_layer = last_hidden[-1]
        # last_token_hidden = last_layer[:, input_token_count : , :].cpu()


        # 5 个层分别收集
        step_tensors_last  = []
        step_tensors_m2    = []
        step_tensors_m4    = []
        step_tensors_m6    = []
        step_tensors_m8    = []

        for step_item in outputs.hidden_states:
            # 对应上面 sample 里 selected_layers 的顺序
            h_last, h_m2, h_m4, h_m6, h_m8 = step_item   # 每个都是 [1, 1, 2048]

            step_tensors_last.append(h_last)
            step_tensors_m2.append(h_m2)
            step_tensors_m4.append(h_m4)
            step_tensors_m6.append(h_m6)
            step_tensors_m8.append(h_m8)

        # 按 dim=1 拼成 [B, answer_tokens_length, 2048]
        last_layer_answer_tokens_hidden   = torch.cat(step_tensors_last, dim=1).cpu()
        minus2_layer_answer_tokens_hidden = torch.cat(step_tensors_m2,   dim=1).cpu()
        minus4_layer_answer_tokens_hidden = torch.cat(step_tensors_m4,   dim=1).cpu()
        minus6_layer_answer_tokens_hidden = torch.cat(step_tensors_m6,   dim=1).cpu()
        minus8_layer_answer_tokens_hidden = torch.cat(step_tensors_m8,   dim=1).cpu()


        # step_tensors = []
        # for step_item in outputs.hidden_states:   # outputs.hidden_states是一个元组，包含若干个元组元素，每个元组元素内是一个[B,1,2048]的张量
        #     step_tensor = step_item[0]
        #     step_tensors.append(step_tensor)

        # answer_tokens_hidden = torch.cat(step_tensors, dim = 1).cpu()    # [B, answer_tokens_length, 2048]
        #print(answer_tokens_hidden.shape)

        full_ids_tensor = outputs.sequences[0]
        full_ids = outputs.sequences[0].cpu().tolist()            # full_ids=[prompt, answer]   它是torch.Tensor类型的
        prefix_len = int(input_token_count)                 # NEW
        has_eos = bool(full_ids[-1] == tokenizer.eos_token_id)  # 可选
        answer_ids = full_ids[prefix_len:]
        truncated = (len(answer_ids) >= MAX_NEW_TOKENS and not has_eos)

        # —— 强校验：生成序列的前缀必须与 prompt_ids 完全一致 —— #
        assert torch.equal(full_ids_tensor[:prefix_len], inputs["input_ids"][0]), \
            "prefix mismatch between inputs['input_ids'] and outputs.sequences prefix!"

        result = {
            'id': rec_id,
            'instruction': instruction,
            'input': item["input"],
            'output': item["output"],
            'qwen3-1.7B_answer_background': answer_text,
            'last_layer_answer_tokens_embedding': last_layer_answer_tokens_hidden,     # [1,307,2048]
            # 新增的 4 个层
            'layer_minus2_answer_tokens_embedding': minus2_layer_answer_tokens_hidden,
            'layer_minus4_answer_tokens_embedding': minus4_layer_answer_tokens_hidden,
            'layer_minus6_answer_tokens_embedding': minus6_layer_answer_tokens_hidden,
            'layer_minus8_answer_tokens_embedding': minus8_layer_answer_tokens_hidden,
            'full_ids': full_ids,                 # NEW      # [423]
            'prefix_len': prefix_len,             # NEW      # 116   数值对上了
            'has_eos': has_eos,                   # 可选
            'answer_ids': answer_ids,            # 便于直接取 label
            'truncated': truncated,
        }

        # 根据 KEEP_LAYER 只保留其中一层的 embedding
        # if KEEP_LAYER == "last":
        #     result['last_layer_answer_tokens_embedding'] = last_layer_answer_tokens_hidden
        # elif KEEP_LAYER == "minus2":
        #     result['layer_minus2_answer_tokens_embedding'] = minus2_layer_answer_tokens_hidden
        # elif KEEP_LAYER == "minus4":
        #     result['layer_minus4_answer_tokens_embedding'] = minus4_layer_answer_tokens_hidden
        # elif KEEP_LAYER == "minus6":
        #     result['layer_minus6_answer_tokens_embedding'] = minus6_layer_answer_tokens_hidden
        # elif KEEP_LAYER == "minus8":
        #     result['layer_minus8_answer_tokens_embedding'] = minus8_layer_answer_tokens_hidden
        # else:
        #     raise ValueError(f"Unsupported KEEP_LAYER: {KEEP_LAYER}")

        # 写入一条并立即刷盘
        pickle.dump(result, output_f)
        output_f.flush()
        # processed_ids.add(rec_id)

        # 显存清理
        torch.cuda.empty_cache()
        gc.collect()


# 保存结果
output_f.close()
print("✅ Done.")
