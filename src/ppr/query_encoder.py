# src/router/query_encoder.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


class MeanPool(nn.Module):
    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        # last_hidden_state: [B, L, H], attention_mask: [B, L]
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B, L, 1]
        s = (last_hidden_state * mask).sum(dim=1)                        # [B, H]
        d = mask.sum(dim=1).clamp_min(1e-6)                              # [B, 1]
        return s / d


class QueryEncoder(nn.Module):
    """
    冻结的文本 encoder，用于：
      - 离线构建各领域的 query embedding → 聚类得到 prototype
      - 在线路由时对输入 query 计算语义表示 e_x

    可以把 encoder_name_or_path 指到：
      - 基座大模型 (推荐): 例如 Qwen3-8B
      - 小模型 Qwen3-1.7B
      - 或任意 sentence embedding 模型（BGE 等）
    """
    def __init__(
        self,
        encoder_name_or_path: str,
        device: str = None,
        torch_dtype: str = "auto",
        max_seq_length: int = 256,
    ):
        super().__init__()
        self.encoder_name_or_path = encoder_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            encoder_name_or_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        # 先尝试 AutoModel（适配 encoder-only / seq2seq 等）
        try:
            self.base_model = AutoModel.from_pretrained(
                encoder_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            ).to(self.device)
        except Exception:
            # 回退到 CausalLM，再取 .model（类似你 Gate 的逻辑）
            lm = AutoModelForCausalLM.from_pretrained(
                encoder_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            ).to(self.device)
            # 大部分 HF CausalLM 都把 backbone 放在 .model 里
            self.base_model = lm.model if hasattr(lm, "model") else lm

        # 冻结参数
        for p in self.base_model.parameters():
            p.requires_grad = False

        # 记录 hidden_size
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "hidden_size"):
            self.hidden_size = int(self.base_model.config.hidden_size)
        else:
            # 兜底：第一次前向后再推断
            self.hidden_size = None

        self.pool = MeanPool()
        self.eval()

    @torch.no_grad()
    def encode_batch(self, texts, batch_size: int = 8) -> torch.Tensor:
        """
        texts: List[str]
        return: [N, H] 的 float32 向量（已经在 CPU 上，方便后续聚类/保存）
        """
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            sub = texts[i:i + batch_size]
            enc = self.tokenizer(
                sub,
                max_length=self.max_seq_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids = enc.input_ids.to(self.device)
            attention_mask = enc.attention_mask.to(self.device)
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )
            last_h = outputs.last_hidden_state  # [B, L, H]
            if self.hidden_size is None:
                self.hidden_size = last_h.size(-1)
            reps = self.pool(last_h, attention_mask)            # [B, H]
            reps = reps.to(dtype=torch.float32).cpu()
            all_vecs.append(reps)
        return torch.cat(all_vecs, dim=0)  # [N, H]
