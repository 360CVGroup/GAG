import pickle
import torch
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader
from .utils import (
    GAG_TOKEN,
)


def _concat_messages_qwen3(messages, tokenizer):
    ## Qwen3 Chat Format

    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    
    return text

def _encode_chat_format(
    messages,
    tokenizer,
    max_seq_length,
    chat_format = 'qwen3',
):
    '''
    messages = [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
      ]

    Return:
      {
        "input_ids":    Tensor[1, length],
        "input_labels": Tensor[1, length],  # user 段 = -100, assistant 段 = token id
      }
    '''
    example_text = _concat_messages_qwen3(messages, tokenizer).strip()
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids     # dtype maybe int64, i.e. torch.long  I'm not sure, check it!
    labels = input_ids.clone()
    #  only for debug
    assert tokenizer.eos_token_id in input_ids, (tokenizer("this is good."+ tokenizer.eos_token + '\n').input_ids, input_ids)

    # 简化场景：只有 user + assistant 两段
    if len(messages) == 2 and messages[0].get("role") == "user":
        # 计算 "只包含 user 段" 的 token 长度，作为 assistant 段起点
        user_text = _concat_messages_qwen3(messages[:1], tokenizer) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        user_end_idx = tokenizer(
            user_text,
            return_tensors='pt',
            max_length=max_seq_length,
            truncation=True,
        ).input_ids.shape[1]

        # 把 user 段的标签置为 -100（不参与 loss）
        labels[:, :user_end_idx] = -100

    else:
        raise ValueError("messages 必须是两个元素：[{'role':'user','content':...}, {'role':'assistant','content':...}]")

    return {
        "input_ids":    input_ids.flatten(),
        "input_labels": labels.flatten(),
    }



def encode_with_chat_format_finetune(
    example,
    tokenizer,
    max_seq_length,    # The maximum total sequence length (prompt+completion) of each training example.
    chat_format = 'qwen3',
    embed_key="last_layer_answer_tokens_embedding",   # ★新增
    ):
    '''
    Here we assume each example has three fields:
        1) "id",
        2) "instruction",
        3) "input",
        4) "output",
        5) "qwen3-1.7B_answer_backgrond",
        6) embedding field specified by embed_key (e.g., last_layer_answer_tokens_embedding / layer_minus2_answer_tokens_embedding / ...)
    '''

    instruction = "You are an expert in immunology and adjuvant, with a strong background in vaccine development. Your research and practice in this field have equipped you with a deep understanding of the mechanisms of immune response and how to optimize vaccine efficacy through adjuvants. You excel in providing concise, precise, and professional responses to questions related to adjuvants and immunology.\n"
    # query, background_tokens_embedding, ground_truth = example['input'], example['last_layer_answer_tokens_embedding'], example['output']

    query = example["input"]
    background_tokens_embedding = example[embed_key]      # ★改这里
    ground_truth = example["output"]

    query = query.strip()
    ground_truth = ground_truth.strip()

    # ---- 1) 取背景 token 长度 L（第二维）----
    if isinstance(background_tokens_embedding, torch.Tensor):
        # 形状约定为 [1, L, 2048]
        if background_tokens_embedding.dim() != 3 or background_tokens_embedding.size(0) != 1:
            raise ValueError(f"Expect background embedding shape [1, L, 2048], got {tuple(background_tokens_embedding.shape)}")

        full_background_len = int(background_tokens_embedding.size(1))
        assert full_background_len >= 1, "background embedding length must be >= 1"
        # 这里只是为了 sanity check，用不到 full_background_len
        background_len = 1   # ★★ 关键：始终只用 1 个 GAG token
        
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
        {"role": "assistant", "content": ground_truth},
    ]

    ret = {}

    encoded = _encode_chat_format(messages, tokenizer, max_seq_length, chat_format = chat_format)
    ret['gag_input_ids'] = encoded['input_ids']
    ret['gag_input_labels'] = encoded['input_labels']
    # ret['gag_background_tokens_embedding'] = background_tokens_embedding
    # ret['gag_background_tokens_embedding_labels'] = torch.ones_like(
    #     background_tokens_embedding[..., 0],    # [1, Lb]
    # )    # torch.ones_like(input) is equivalent to torch.ones(input.size(), dtype=input.dtype, layout=input.layout, device=input.device).
    # ret['gag_ground_truth_ids'] = encoded['ground_truth_ids']
    # ret['gag_ground_truth_labels'] = encoded['ground_truth_labels']
    
    return ret


class train_mlp_Dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,                     # 用于预编码
        max_seq_length: int,           # 用于预编码
        chat_format: str = "qwen3",     # 用于预编码
        check_presence: bool = True,     # 仅做“字段存在性”检查，不改动数据
        drop_bad_samples: bool = False,  # 预编码失败时是否跳过该样本
        embed_key: str = "last_layer_answer_tokens_embedding",
    ):
        self.data_path = data_path
        self.embed_key = embed_key          # ★★★ 必须加这一行（放在最前面）
        self.records: List[Dict[str, Any]] = self.load_data(data_path)

        if len(self.records) > 0:
            assert self.embed_key in self.records[0], (
                f"embed_key={self.embed_key} not found. available keys={list(self.records[0].keys())}"
            )

        if check_presence:
            need_keys = {
                "id",
                "instruction",
                "input",
                "output",
                "qwen3-1.7B_answer_background",
                self.embed_key,
            }
            for i,rec in enumerate(self.records):
                missing = [k for k in need_keys if k not in rec]
                if missing:
                    raise KeyError(f"Sample #{i} missing keys: {missing}. embed_key={self.embed_key}")


        # ---- 预编码：把每条样本转换成 gag_input_ids / gag_input_labels ----
        self.samples: List[Dict[str, Any]] = []
        self.num_skipped = 0


        for i, rec in enumerate(self.records):
            try:
                ret = encode_with_chat_format_finetune(
                    example=rec,
                    tokenizer=tokenizer,
                    max_seq_length=max_seq_length,
                    chat_format=chat_format,
                    embed_key=self.embed_key,      # ★新增
                )

                # ---- 只取最后一个答案 token 的 embedding，形状 [1, 1, 2048] ----
                full_emb = rec[self.embed_key]
                if isinstance(full_emb, torch.Tensor):
                    if full_emb.dim() != 3 or full_emb.size(0) != 1:
                        raise ValueError(f"Expect {self.embed_key} shape [1, L, 2048], got {tuple(full_emb.shape)}")
                    compressed_emb = full_emb[:, -1:, :].contiguous()   # [1, 1, 2048]
                else:
                    raise TypeError(f"{self.embed_key} must be a Tensor")

                # 只保留核心字段（可按需扩展）
                out = {
                    "id": rec["id"],    # int
                    "gag_input_ids":    ret["gag_input_ids"].to(dtype=torch.long),    # [Length]
                    "gag_input_labels": ret["gag_input_labels"].to(dtype=torch.long), # [Length]
                    "answer_tokens_embedding": compressed_emb,  # ★★ 只用最后一个 token
                }
                self.samples.append(out)
                if i == 0:
                    print(f"DEBUG train sample[0] embed_key={self.embed_key}, compressed shape={tuple(compressed_emb.shape)}")

            except Exception as e:
                if drop_bad_samples:
                    self.num_skipped += 1
                    print(f"Skip sample {i} due to error: {e}")
                    #logging.warning("Skip sample #%d due to error: %s", i, e, exc_info=True)   # exc_info=True 会把 traceback 一并写入日志。
                    # 也可以打印/记录 e 便于定位问题
                    continue
                else:
                    raise

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples encoded from: {data_path}")


    @staticmethod
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

    def __len__(self) -> int:
        return len(self.samples)

    
    def __getitem__(self, idx: int):
        return self.samples[idx]  # Dict[str, Any]

# def main():
#     # 训练集路径
#     data_path = "/home/jovyan/projection/MinerU/adjuvant_code/7.20_pipeline_code/train_mlp_7.26/7.26_pipeline_dataset/splited_Adjuvant_QA_unimodal_withoutocr_pure_QA_shuffle_with_id/output_files/Adjuvant_QA_unimodal_withoutocr_pure_QA_shuffle_with_id_6_with_answer_embeddding.pkl"
    
#     # 读取数据
#     data = load_data(data_path)
#     print(f"Loaded data from {data_path}. Number of samples: {len(data)}")

#     sample_data = data[:5]

#     for i, item in enumerate(sample_data, 1):
#         print(f"\n--- Sample {i} ---")
#         print(item)  # 原样输出整个字典
#         emb = item["last_layer_answer_tokens_embedding"]
#         if isinstance(emb, torch.Tensor):
#             print("last_layer_answer_tokens_embedding shape:", emb.shape)
#         else:
#             print("last_layer_answer_tokens_embedding is not a torch.Tensor")




def collator(        
        samples,
        llm_tokenizer,
    ):
    """
    collate tokenized input_ids and labels with left and right side padding supported
    
    Args:
        samples (dict): a dict contains input_ids, labels and maybe retrieval_text
        llm_tokenizer: tokenizer for llm
        retriever_tokenizer: tokenizer for retriever
        retrieval_context_length: max length for the retrieved passages
    
    Returns:
        xrag_input_ids: input_ids with xrag_token_id (xrag_labels,xrag_attention_mask)
        input_ids: input_ids for llm without xrag_token_id, vanilla rag (labels,attention_mask)
        retriever_input_ids: input_ids for retriever (retriever_attention_mask)

    """
    def padding(input_ids,labels=None,padding_side='right'):
        """
        batch padding
        """

        def _padding(ids,padding_value,padding_side='right'):
            if padding_side == 'right':
                return torch.nn.utils.rnn.pad_sequence(ids,batch_first=True,padding_value=padding_value)
            elif padding_side == 'left':
                flipped_ids = [torch.flip(x, dims=[0]) for x in ids]  
                return torch.flip(
                    torch.nn.utils.rnn.pad_sequence(flipped_ids,batch_first=True,padding_value=padding_value),
                    dims=[1],
                )
        input_ids = _padding(input_ids,padding_value=llm_tokenizer.pad_token_id,padding_side=padding_side)
        attention_mask = (input_ids != llm_tokenizer.pad_token_id).long()
        if labels is not None:
            labels = _padding(labels,padding_value=-100,padding_side=padding_side)
        return input_ids,attention_mask,labels

    gag_input_ids,gag_attention_mask,gag_input_labels = padding(
        input_ids=[x['gag_input_ids'] for x in samples],
        labels=[x['gag_input_labels'] for x in samples], # if 'gag_input_labels' in samples[0].keys() else None,
        padding_side=llm_tokenizer.padding_side,
    )

    ## add some noise to pretraining task TODO

    ret = {
        "gag_input_ids":gag_input_ids,
        "gag_attention_mask":gag_attention_mask,
        "gag_input_labels":gag_input_labels,
    }

    # 把每个样本的 [1, L_s, 2048] 压成 [L_s, 2048]，再按样本顺序 cat 成 [-1, 2048]
    embeds_flattened = []
    for s in samples:
        e = s["answer_tokens_embedding"]
        if e.dim() != 3 or e.size(0) != 1:
            raise ValueError(f"Expect [1, Ls, 2048], got {tuple(e.shape)}")
        embeds_flattened.append(e.squeeze(0))   # [Ls, 2048]
    if len(embeds_flattened) > 0:
        ret["retrieval_embeds"] = torch.cat(embeds_flattened, dim=0)  # [-1, 2048]
    else:
        # ret["retrieval_embeds"] = torch.empty(0, 2048)
        raise ValueError(
            "本批次所有样本的 'answer_tokens_embedding' 都为空（sum Ls == 0）。"
            "请检查上游生成逻辑或过滤条件。"
        )

    return ret    

    # if 'retriever_input_text' in samples[0].keys():
    #     retriever_input_text = [x['retriever_input_text'] for x in samples]
    #     assert isinstance(retriever_input_text[0],list)
    #     retriever_input_text = [x for y in retriever_input_text for x in y]
    #     ## handling different retriever tokenization problem
    #     if retriever_tokenizer.name_or_path == "intfloat/e5-large-v2":
    #         retriever_input_text = ["passage: "+x for x in retriever_input_text]
    #     elif retriever_tokenizer.name_or_path == 'intfloat/e5-mistral-7b-instruct':
    #         retriever_input_text = [x + retriever_tokenizer.eos_token for x in retriever_input_text]

    #     tokenized_retrieval_text = retriever_tokenizer(
    #         retriever_input_text, 
    #         max_length=retrieval_context_length,
    #         padding=True, truncation=True, return_tensors="pt"
    #     )
        
    #     ret['retriever_input_ids']      = tokenized_retrieval_text['input_ids']
    #     ret['retriever_attention_mask'] = tokenized_retrieval_text['attention_mask']
    
    # if 'input_ids' in samples[0].keys():
    #     input_ids = [x['input_ids'] for x in samples]
    #     labels =    [x['labels'] for x in samples]
     
    #     input_ids,attention_mask,labels = padding(input_ids,labels,padding_side=llm_tokenizer.padding_side)
        
    #     ret['input_ids'] = input_ids
    #     ret['attention_mask'] = attention_mask
    #     ret['labels'] = labels

    # return ret

# def get_retrieval_embeds(batch, key="answer_tokens_embedding"):
#     """
#     Extracts and processes the retrieval embeddings from the batch.
#     The batch contains a key that stores the embeddings in the form of [1, Ls, 2048],
#     where we remove the first dimension (size=1) and concatenate the embeddings for the batch.
    
#     Args:
#         batch: A dictionary containing the batch data from DataLoader.
#         key (str): The key to access the embeddings in the batch (default: "answer_tokens_embedding").
        
#     Returns:
#         torch.Tensor: A tensor of shape [Batch_size, 2048] containing the concatenated embeddings.
#     """
#     embeddings = []
    
#     for sample in batch:
#         # Extract the answer_tokens_embedding and remove the first dimension
#         answer_tokens_embedding = sample[key]  # Shape: [1, Ls, 2048]
        
#         if answer_tokens_embedding.dim() != 3 or answer_tokens_embedding.size(0) != 1:
#             raise ValueError(f"Expect answer_tokens_embedding shape [1, Ls, 2048], got {answer_tokens_embedding.shape}")
        
#         # Remove the first dimension (which is 1) to get [Ls, 2048]
#         embedding = answer_tokens_embedding.squeeze(0)  # Shape: [Ls, 2048]
        
#         embeddings.append(embedding)
    
#     # Stack the embeddings along the first dimension (Batch size)
#     retrieval_embeds = torch.stack(embeddings, dim=0)  # Shape: [Batch_size, 2048]
    
#     return retrieval_embeds



# def main():
#     pkl_path = "projection/MinerU/adjuvant_code/7.20_pipeline_code/train_mlp_8_15/get_small_model_answer_tokens_embedding/output_files/Adjuvant_QA_unimodal_withoutocr_pure_QA_shuffle_with_id_6_with_answer_embeddding.pkl"

#     # 1) 构建数据集
#     ds = train_mlp_Dataset(pkl_path, check_presence=True)     
#     print(f"Loaded {len(ds)} samples from {pkl_path}")

#     # 2) 看一条样本的关键信息
#     s0 = ds[0]        ## s0 Dict[str, Any]    
#                       # s0['last_layer_answer_tokens_embedding'].dtype : torch.float32   s0['last_layer_answer_tokens_embedding'].device : device(type='cpu')
#     print("keys:", list(s0.keys()))
#     emb = s0["last_layer_answer_tokens_embedding"]
#     try:
#         # 如果是 torch.Tensor
#         print("embedding type:", type(emb), "shape:", emb.shape)
#     except Exception:
#         # 如果不是 tensor（比如 numpy/list），也打印一点信息
#         print("embedding type:", type(emb))

#     # 3) 用 DataLoader 跑一遍（注意：没有 collate_fn 时只能 batch_size=1）
#     loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
#     # batch0 = next(iter(loader))
#     # # DataLoader 默认会把每个键收集成 list（因为样本是 dict），这一步只是验证能跑通
#     # print("batch0 keys:", batch0.keys())
#     # print("batch0 id:", batch0["id"][0])
#     for item in loader:     # type(item['instruction'])  <class 'list'>     
#                             # type(item['last_layer_answer_tokens_embedding'])  <class 'torch.Tensor'>
#         print(item.keys())
        

# if __name__ == "__main__":
#     main()