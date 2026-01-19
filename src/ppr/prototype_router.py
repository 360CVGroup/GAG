# src/router/prototype_router.py
import os
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

from .query_encoder import QueryEncoder


class PrototypeRouter:
    """
    Prototype-based Plug-and-play Router (PPR)

    - 使用一个冻结的 QueryEncoder 定义语义空间 g(x)
    - 每个领域 i 有一组原型向量 P_i ∈ R^{C_i × H}（L2-normalized）
    - 路由规则：对每个领域 i，计算 s_i(x) = max_p∈P_i cos(e_x, p)，选 argmax

    通用领域（"general"）只是一个特殊的领域名而已，和专业领域平等竞争。
    """

    def __init__(
        self,
        encoder_name_or_path: str,
        prototype_files: Dict[str, str],
        device: Optional[str] = None,
        max_seq_length: int = 256,
        torch_dtype: str = "auto",
    ):
        """
        Args:
            encoder_name_or_path: 用于路由的 encoder（建议用 Qwen3-8B，与基座一致）
            prototype_files: {domain_name: path_to_prototype_pt}
                             其中必须包含 "general" 这个领域名
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = QueryEncoder(
            encoder_name_or_path=encoder_name_or_path,
            device=self.device,
            torch_dtype=torch_dtype,
            max_seq_length=max_seq_length,
        )

        self.domain_list: List[str] = []
        self.proto_bank: Dict[str, torch.Tensor] = {}
        self.encoder_name_or_path = encoder_name_or_path
        self.hidden_size = None

        for domain, path in prototype_files.items():
            self.add_domain_from_file(domain, path)

        assert "general" in self.domain_list, \
            "PrototypeRouter 要求 prototype_files 至少包含 'general' 领域"

    def add_domain_from_file(self, domain_name: str, prototype_pt_path: str):
        """
        可插拔式扩展：新增一个领域只需要：
          1) 用 build_domain_prototypes.py 离线生成该领域的 prototype .pt
          2) 在这里 load 进去，无需改动 encoder 和其他领域
        """
        if not os.path.isfile(prototype_pt_path):
            raise FileNotFoundError(f"{prototype_pt_path} not found for domain {domain_name}")

        obj = torch.load(prototype_pt_path, map_location="cpu")
        prototypes: torch.Tensor = obj["prototypes"]  # [C, H]
        if prototypes.ndim != 2:
            raise ValueError(f"prototypes for domain={domain_name} must be 2D, got {prototypes.shape}")

        # 确保 encoder 与 prototype 所在空间维度一致
        h = prototypes.size(1)
        if self.hidden_size is None:
            self.hidden_size = h
        else:
            assert h == self.hidden_size, \
                f"hidden_size mismatch: router={self.hidden_size}, domain={domain_name}, proto_dim={h}"

        # 再次 L2 normalize（以防万一）
        prototypes = prototypes.to(dtype=torch.float32)
        prototypes = F.normalize(prototypes, dim=1)  # [C, H]

        self.proto_bank[domain_name] = prototypes.to(self.device)
        if domain_name not in self.domain_list:
            self.domain_list.append(domain_name)

        print(f"[PrototypeRouter] add domain='{domain_name}', num_prototypes={prototypes.size(0)}, hidden_size={h}")

    @torch.no_grad()
    def _encode_single(self, question: str) -> torch.Tensor:
        reps = self.encoder.encode_batch([question], batch_size=1)  # [1, H] on CPU
        e = reps[0].to(self.device)                                 # [H]
        e = F.normalize(e, dim=0)
        return e  # [H]

    @torch.no_grad()
    def route(
        self,
        question: str,
        return_scores: bool = False,
    ) -> Tuple[str, float, Optional[Dict[str, float]]]:
        """
        核心路由函数（不使用任何阈值，通用领域与专业领域平等竞争）：

        Returns:
            best_domain: str
            best_score: float (对应 s_i(x) = max cosine)
            scores_dict: {domain: score}（仅当 return_scores=True 时返回）
        """
        e = self._encode_single(question)  # [H]
        scores = {}
        for domain in self.domain_list:
            P = self.proto_bank[domain]          # [C, H]
            # [C]
            sim = torch.matmul(P, e)             # cos 相似度，因为都已经 L2 归一化
            s_i = float(sim.max().item())
            scores[domain] = s_i

        # 取 argmax
        best_domain = max(scores.items(), key=lambda kv: kv[1])[0]
        best_score = scores[best_domain]

        if return_scores:
            return best_domain, best_score, scores
        else:
            return best_domain, best_score, None

    @torch.no_grad()
    def route_with_margin(
        self,
        question: str,
        general_domain_name: str = "general",
        margin: float = 0.0,
        return_scores: bool = False,
    ) -> Tuple[str, float, Optional[Dict[str, float]]]:
        """
        扩展版（可选）：带 margin 的“安全回退”策略

        - 先一样做 argmax
        - 若 best_domain != general 且 best_score - second_best_score < margin
          则回退为 general

        一般论文里可以把不带 margin 的版本当 main method，
        这个当 ablation / 附录扩展即可。
        """
        e = self._encode_single(question)
        scores = {}
        for domain in self.domain_list:
            P = self.proto_bank[domain]
            sim = torch.matmul(P, e)
            s_i = float(sim.max().item())
            scores[domain] = s_i

        sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_domain, best_score = sorted_items[0]
        if len(sorted_items) > 1:
            second_score = sorted_items[1][1]
        else:
            second_score = best_score

        if best_domain != general_domain_name:
            if (best_score - second_score) < margin:
                # 安全回退到 general
                best_domain = general_domain_name
                best_score = scores[general_domain_name]

        if return_scores:
            return best_domain, best_score, scores
        else:
            return best_domain, best_score, None
