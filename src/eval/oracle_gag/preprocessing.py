from src.language_modeling.utils import GAG_TOKEN
from typing import List, Dict, Any
import pickle
import torch

def load_data(data_path: str) -> List[Dict[str, Any]]:
    """逐条读取 pickle 文件中的数据"""
    data = []
    with open(data_path, "rb") as f:
        while True:
            try:
                # 逐条加载数据
                item = pickle.load(f)
                data.append(item)
            except EOFError:
                break  # 读取完毕
    return data



def _concat_messages_qwen3(messages, tokenizer):
    ## Qwen3 Chat Format

    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    
    return text


def prepare_prompts(
    test_data,
    tokenizer,
    chat_format = 'qwen3',
    embed_key="last_layer_answer_tokens_embedding",
):
    if len(test_data) > 0:
        assert embed_key in test_data[0], f"embed_key={embed_key} not found in first record keys={list(test_data[0].keys())}"
        
    instruction = "You are an expert in immunology and adjuvant, with a strong background in vaccine development. Your research and practice in this field have equipped you with a deep understanding of the mechanisms of immune response and how to optimize vaccine efficacy through adjuvants. You excel in providing concise, precise, and professional responses to questions related to adjuvants and immunology.\n"

    prompts = []
    backgrounds = []

    for idx,sample in enumerate(test_data):

        # query, background_tokens_embedding = sample['input'], sample['last_layer_answer_tokens_embedding']

        query = sample["input"]
        background_tokens_embedding = sample[embed_key]     # ★改这里

        query = query.strip()

        # ---- 1) 取背景 token 长度 L（第二维）----
        if isinstance(background_tokens_embedding, torch.Tensor):
            # 形状约定为 [1, L, 2048]
            if background_tokens_embedding.dim() != 3 or background_tokens_embedding.size(0) != 1:
                raise ValueError(f"Expect background embedding shape [1, L, 2048], got {tuple(background_tokens_embedding.shape)}")

            full_L = int(background_tokens_embedding.size(1))
            assert full_L >= 1, "background embedding length must be >= 1"

            # 只保留最后一个 token 的 embedding，形状 [1, 1, 2048]
            background_tokens_embedding = background_tokens_embedding[:, -1:, :].contiguous()
            background_len = 1

            # background_len = int(background_tokens_embedding.size(1))  # L
        else:
            # 若不是 tensor（极少见），做个兜底
            raise TypeError(f"{embed_key} must be a 3D tensor-like of shape [1, L, D].")

        # 生成占位串（例如 "<GAG><GAG><GAG> ..."）
        gag_tokens = "".join([GAG_TOKEN] * background_len)

        prompt = (
            instruction
            + f"Please answer the following question based on the knowledge provided.\nQuestion: {query}\nKnowledge: {gag_tokens}\n "
        )    

        messages = [
            {"role": "user", "content": prompt},
        ]

        user_prompt = _concat_messages_qwen3(messages, tokenizer) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        
        prompts.append(user_prompt)
        backgrounds.append(background_tokens_embedding)

        if idx == 0:
            print(f"DEBUG eval sample[0] embed_key={embed_key}, compressed shape={tuple(background_tokens_embedding.shape)}")

    print("**"*20,"show one example","**"*20)
    print(prompts[0])
    print("**"*20,"show one example","**"*20)

    return prompts,backgrounds   # backgrounds : List[一个形状为1,length,2048的张量, 一个形状为1,length,2048的张量, 一个形状为1,length,2048的张量……]