import math
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F


LAYER_KEY_TO_FIELD = {
    "last": "last_layer_answer_tokens_embedding",
    "last_layer_answer_tokens_embedding": "last_layer_answer_tokens_embedding",
    "minus2": "layer_minus2_answer_tokens_embedding",
    "layer_minus2_answer_tokens_embedding": "layer_minus2_answer_tokens_embedding",
    "minus4": "layer_minus4_answer_tokens_embedding",
    "layer_minus4_answer_tokens_embedding": "layer_minus4_answer_tokens_embedding",
    "minus6": "layer_minus6_answer_tokens_embedding",
    "layer_minus6_answer_tokens_embedding": "layer_minus6_answer_tokens_embedding",
    "minus8": "layer_minus8_answer_tokens_embedding",
    "layer_minus8_answer_tokens_embedding": "layer_minus8_answer_tokens_embedding",
}


DEFAULT_LAYER_KEYS = ["last", "minus2", "minus4", "minus6"]
DEFAULT_MEMORY_NORMALIZATION = "layernorm"
DEFAULT_MEMORY_CLAMP_VALUE = 10.0


def normalize_layer_keys(layer_keys: Sequence[str] | str | None) -> List[str]:
    if layer_keys is None:
        layer_keys = DEFAULT_LAYER_KEYS
    if isinstance(layer_keys, str):
        layer_keys = [part.strip() for part in layer_keys.split(",") if part.strip()]

    normalized = []
    for key in layer_keys:
        if key not in LAYER_KEY_TO_FIELD:
            raise ValueError(
                f"Unsupported layer key: {key}. Supported keys: {sorted(LAYER_KEY_TO_FIELD)}"
            )
        normalized.append(key)
    return normalized


def _to_sequence_tensor(sequence_tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(sequence_tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(sequence_tensor)!r}")
    if sequence_tensor.dim() == 3:
        if sequence_tensor.size(0) != 1:
            raise ValueError(
                f"Expected shape [1, L, H] for stored embeddings, got {tuple(sequence_tensor.shape)}"
            )
        sequence_tensor = sequence_tensor.squeeze(0)
    if sequence_tensor.dim() != 2:
        raise ValueError(
            f"Expected shape [L, H] or [1, L, H], got {tuple(sequence_tensor.shape)}"
        )
    return sequence_tensor.float().contiguous()


def stabilize_memory_slots(
    memory_slots: torch.Tensor,
    normalization: str = DEFAULT_MEMORY_NORMALIZATION,
    clamp_value: float = DEFAULT_MEMORY_CLAMP_VALUE,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not isinstance(memory_slots, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(memory_slots)!r}")

    stabilized = torch.nan_to_num(
        memory_slots.float(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    normalization = (normalization or "none").lower()
    if normalization == "layernorm":
        stabilized = F.layer_norm(stabilized, (stabilized.size(-1),), eps=eps)
    elif normalization == "rmsnorm":
        variance = stabilized.pow(2).mean(dim=-1, keepdim=True)
        stabilized = stabilized * torch.rsqrt(variance + eps)
    elif normalization in {"none", ""}:
        pass
    else:
        raise ValueError(f"Unsupported memory normalization mode: {normalization}")

    if clamp_value is not None and clamp_value > 0:
        stabilized = stabilized.clamp(min=-float(clamp_value), max=float(clamp_value))

    return stabilized.contiguous()


def compress_sequence_to_memory_slots(
    sequence_tensor: torch.Tensor,
    num_memory_slots: int = 4,
    pooling: str = "segment_softmax",
    temperature: float = 1.0,
) -> torch.Tensor:
    sequence_tensor = _to_sequence_tensor(sequence_tensor)
    seq_len, hidden_size = sequence_tensor.shape

    if seq_len < 1:
        raise ValueError("Sequence length must be >= 1")
    if num_memory_slots < 1:
        raise ValueError("num_memory_slots must be >= 1")

    if num_memory_slots == 1:
        return sequence_tensor.mean(dim=0, keepdim=True)

    if pooling == "adaptive_avg":
        pooled = torch.nn.functional.adaptive_avg_pool1d(
            sequence_tensor.transpose(0, 1).unsqueeze(0),
            output_size=num_memory_slots,
        )
        return pooled.squeeze(0).transpose(0, 1).contiguous()

    if pooling != "segment_softmax":
        raise ValueError(f"Unsupported pooling mode: {pooling}")

    boundaries = torch.linspace(
        0, seq_len, steps=num_memory_slots + 1, dtype=torch.float32, device=sequence_tensor.device
    )
    boundaries = boundaries.round().long()

    slots = []
    for slot_idx in range(num_memory_slots):
        start = int(boundaries[slot_idx].item())
        end = int(boundaries[slot_idx + 1].item())
        if end <= start:
            center = min(seq_len - 1, max(0, start))
            segment = sequence_tensor[center : center + 1]
        else:
            segment = sequence_tensor[start:end]

        scores = segment.norm(dim=-1)
        if temperature and temperature > 0:
            scores = scores / temperature
        weights = torch.softmax(scores, dim=0).unsqueeze(-1)
        slots.append((segment * weights).sum(dim=0))

    return torch.stack(slots, dim=0).view(num_memory_slots, hidden_size)


def build_memory_slots_from_layer_dict(
    record: Dict,
    layer_keys: Sequence[str] | str | None = None,
    num_memory_slots: int = 4,
    pooling: str = "segment_softmax",
    temperature: float = 1.0,
) -> torch.Tensor:
    normalized_keys = normalize_layer_keys(layer_keys)
    per_layer_slots = []

    for layer_key in normalized_keys:
        field_name = LAYER_KEY_TO_FIELD[layer_key]
        if field_name not in record:
            raise KeyError(
                f"Missing field `{field_name}` in record. Available keys: {list(record.keys())}"
            )
        slots = compress_sequence_to_memory_slots(
            record[field_name],
            num_memory_slots=num_memory_slots,
            pooling=pooling,
            temperature=temperature,
        )
        per_layer_slots.append(slots)

    stacked = torch.stack(per_layer_slots, dim=0).contiguous()
    return stabilize_memory_slots(stacked)


def build_memory_slots_from_record(
    record: Dict,
    layer_keys: Sequence[str] | str | None = None,
    num_memory_slots: int = 4,
    pooling: str = "segment_softmax",
    temperature: float = 1.0,
) -> torch.Tensor:
    return build_memory_slots_from_layer_dict(
        record=record,
        layer_keys=layer_keys,
        num_memory_slots=num_memory_slots,
        pooling=pooling,
        temperature=temperature,
    )
