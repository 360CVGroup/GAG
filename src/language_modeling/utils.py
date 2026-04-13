import json
import os
import shutil
import tempfile
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer


GAG_TOKEN = "<GAG>"


def get_yaml_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_path(workdir: str | None, path: str | None) -> str | None:
    if path is None:
        return None
    if isinstance(path, str) and path.strip().lower() in {"", "none", "null"}:
        return None
    if os.path.isabs(path):
        return path
    if workdir is None:
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(workdir, path))


def _prepare_sanitized_tokenizer_path(model_name_or_path: str) -> str | None:
    if not os.path.isdir(model_name_or_path):
        return None

    tokenizer_config_path = os.path.join(model_name_or_path, "tokenizer_config.json")
    if not os.path.exists(tokenizer_config_path):
        return None

    with open(tokenizer_config_path, "r", encoding="utf-8") as file:
        tokenizer_config = json.load(file)

    if not isinstance(tokenizer_config.get("extra_special_tokens"), list):
        return None

    sanitized_dir = tempfile.mkdtemp(prefix="tokenizer_fix_")
    tokenizer_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "merges.txt",
        "vocab.json",
        "chat_template.jinja",
    ]

    for file_name in tokenizer_files:
        source_path = os.path.join(model_name_or_path, file_name)
        if not os.path.exists(source_path):
            continue

        target_path = os.path.join(sanitized_dir, file_name)
        if file_name == "tokenizer_config.json":
            patched_config = dict(tokenizer_config)
            patched_config["extra_special_tokens"] = {}
            with open(target_path, "w", encoding="utf-8") as file:
                json.dump(patched_config, file, ensure_ascii=False, indent=2)
                file.write("\n")
            continue

        try:
            os.symlink(source_path, target_path)
        except OSError:
            shutil.copy2(source_path, target_path)

    return sanitized_dir


def load_tokenizer_with_fast_fallback(model_name_or_path: str, **kwargs):
    try:
        return AutoTokenizer.from_pretrained(
            model_name_or_path,
            **kwargs,
        )
    except TypeError as error:
        error_text = str(error)
        if "NoneType" not in error_text and "os.PathLike" not in error_text:
            raise
    except AttributeError as error:
        error_text = str(error)
        if "keys" not in error_text and "extra_special_tokens" not in error_text:
            raise
    except ValueError as error:
        error_text = str(error)
        if "vocab_file" not in error_text and "tokenizer file" not in error_text.lower():
            raise

    sanitized_path = _prepare_sanitized_tokenizer_path(model_name_or_path) or model_name_or_path
    print(
        "[tokenizer] Falling back to a sanitized fast-tokenizer load because the tokenizer artifacts are incomplete for "
        f"{model_name_or_path}."
    )
    return AutoTokenizer.from_pretrained(
        sanitized_path,
        use_fast=True,
        **kwargs,
    )


def get_nll_loss(logits, labels, vocab_size):
    shift_logits = logits[..., :-1, :].contiguous().view(-1, vocab_size)
    shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits, shift_labels)


def masked_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.dim() == 2:
        attention_mask = attention_mask.unsqueeze(-1)
    attention_mask = attention_mask.to(dtype=hidden_states.dtype, device=hidden_states.device)
    numerator = (hidden_states * attention_mask).sum(dim=1)
    denominator = attention_mask.sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def mean_pool_encoder_outputs(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype, device=last_hidden_state.device)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def split_projected_memory_by_sample(
    projected_memory: torch.Tensor,
    memory_slots_per_sample: torch.Tensor | List[int],
) -> List[torch.Tensor]:
    if isinstance(memory_slots_per_sample, torch.Tensor):
        counts = memory_slots_per_sample.tolist()
    else:
        counts = list(memory_slots_per_sample)

    chunks = []
    start = 0
    for count in counts:
        end = start + int(count)
        chunks.append(projected_memory[start:end])
        start = end
    return chunks


def get_latent_semantic_loss(
    projected_answer_repr: torch.Tensor,
    target_semantic_repr: torch.Tensor,
) -> torch.Tensor:
    projected_answer_repr = F.normalize(projected_answer_repr.float(), dim=-1, eps=1e-6)
    target_semantic_repr = F.normalize(target_semantic_repr.float(), dim=-1, eps=1e-6)
    return 1.0 - (projected_answer_repr * target_semantic_repr).sum(dim=-1).mean()


def get_memory_diversity_loss(memory_chunks: Iterable[torch.Tensor]) -> torch.Tensor:
    losses = []
    for chunk in memory_chunks:
        if chunk is None or chunk.numel() == 0 or chunk.size(0) <= 1:
            continue
        normalized = F.normalize(chunk.float(), dim=-1, eps=1e-6)
        similarity = normalized @ normalized.transpose(0, 1)
        off_diagonal = similarity[~torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)]
        if off_diagonal.numel() > 0:
            losses.append(off_diagonal.pow(2).mean())
    if not losses:
        device = None
        for chunk in memory_chunks:
            if chunk is not None:
                device = chunk.device
                break
        return torch.tensor(0.0, device=device or "cpu")
    return torch.stack(losses).mean()


def save_with_accelerate(accelerator, model, tokenizer, output_dir, save_projector_only=False):
    unwrapped_model = accelerator.unwrap_model(model)
    os.makedirs(output_dir, exist_ok=True)

    if save_projector_only:
        params_to_save = {
            name: param.float().cpu()
            for name, param in unwrapped_model.named_parameters()
            if any(
                key in name
                for key in [
                    "embed_tokens",
                    "projector",
                    "layer_mixer_logits",
                    "semantic_head",
                    "lm_head",
                ]
            )
        }
        if accelerator.is_main_process:
            torch.save(params_to_save, os.path.join(output_dir, "ckpt.pth"))
            unwrapped_model.config.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        return

    state_dict = accelerator.get_state_dict(model)
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
        safe_serialization=False,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
