from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM


class XQwen3Config(Qwen3Config):
    def __init__(
        self,
        projector_type: str = "gated_residual",
        retriever_hidden_size: int = 2048,
        use_layer_mix: bool = True,
        layer_mix_num_layers: int = 4,
        num_memory_slots: int = 4,
        memory_slot_dropout: float = 0.0,
        layer_mix_logit_clamp: float = 10.0,
        semantic_hidden_size: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.retriever_hidden_size = int(retriever_hidden_size)
        self.use_layer_mix = bool(use_layer_mix)
        self.layer_mix_num_layers = int(layer_mix_num_layers)
        self.num_memory_slots = int(num_memory_slots)
        self.memory_slot_dropout = float(memory_slot_dropout)
        self.layer_mix_logit_clamp = float(layer_mix_logit_clamp)
        self.semantic_hidden_size = int(semantic_hidden_size)


class Projector(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.projector_type == "mlp2x_gelu":
            self.input_norm = nn.LayerNorm(config.retriever_hidden_size)
            self.projector = nn.Sequential(
                nn.Linear(config.retriever_hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            self.projector_type = "mlp2x_gelu"
            return

        if config.projector_type != "gated_residual":
            raise ValueError(f"Unsupported projector_type: {config.projector_type}")

        self.projector_type = "gated_residual"
        self.input_norm = nn.LayerNorm(config.retriever_hidden_size)
        self.input_proj = nn.Linear(config.retriever_hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.update = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Sigmoid(),
        )

    def forward(self, context_embedding):
        if self.projector_type == "mlp2x_gelu":
            input_dtype = context_embedding.dtype
            compute_dtype = self.projector[0].weight.dtype
            normalized_input = self.input_norm(context_embedding.to(dtype=compute_dtype))
            projected = self.projector(normalized_input)
            return projected.to(dtype=input_dtype)

        input_dtype = context_embedding.dtype
        compute_dtype = self.input_proj.weight.dtype
        context_embedding = self.input_norm(context_embedding.to(dtype=compute_dtype))
        base = self.input_proj(context_embedding)
        normalized = self.norm(base)
        update = self.update(normalized)
        gate = self.gate(normalized)
        return (base + gate * update).to(dtype=input_dtype)


class XQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.projector = Projector(config)
        self.retriever_hidden_size = config.retriever_hidden_size
        self.memory_slot_dropout = nn.Dropout(config.memory_slot_dropout)

        if config.use_layer_mix:
            self.layer_mixer_logits = nn.Parameter(
                torch.zeros(config.num_memory_slots, config.layer_mix_num_layers)
            )
        else:
            self.layer_mixer_logits = None

        if config.semantic_hidden_size > 0:
            self.semantic_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.semantic_hidden_size),
            )
        else:
            self.semantic_head = None

        self.post_init()

    def set_gag_token_id(self, token_id):
        self.gag_token_id = token_id

    def _get_layer_mix_weights(
        self,
        num_layers: int,
        device,
        retrieval_slot_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.layer_mixer_logits is None:
            if retrieval_slot_indices is None:
                return torch.full((num_layers,), 1.0 / num_layers, device=device)
            return torch.full(
                (retrieval_slot_indices.numel(), num_layers),
                1.0 / num_layers,
                device=device,
            )

        logits = self.layer_mixer_logits.float()
        clamp_value = getattr(self.config, "layer_mix_logit_clamp", 0.0)
        clamp_value = 0.0 if clamp_value is None else float(clamp_value)
        if clamp_value > 0:
            logits = logits.clamp(
                min=-clamp_value,
                max=clamp_value,
            )
        if logits.size(-1) < num_layers:
            logits = F.pad(logits, (0, num_layers - logits.size(-1)), value=0.0)
        elif logits.size(-1) > num_layers:
            logits = logits[:, :num_layers]

        if retrieval_slot_indices is None:
            return torch.softmax(logits.to(device=device), dim=-1)

        slot_indices = retrieval_slot_indices.to(device=device, dtype=torch.long)
        slot_logits = logits.to(device=device).index_select(0, slot_indices)
        return torch.softmax(slot_logits, dim=-1)

    def mix_retrieval_layers(
        self,
        retrieval_embeds: torch.Tensor,
        retrieval_slot_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if retrieval_embeds.dim() == 2:
            return retrieval_embeds
        if retrieval_embeds.dim() != 3:
            raise ValueError(
                "retrieval_embeds must have shape [num_slots, hidden] or [num_slots, num_layers, hidden], "
                f"got {tuple(retrieval_embeds.shape)}"
            )

        if retrieval_slot_indices is None:
            if self.layer_mixer_logits is None:
                retrieval_slot_indices = torch.arange(
                    retrieval_embeds.size(0),
                    device=retrieval_embeds.device,
                    dtype=torch.long,
                )
            else:
                retrieval_slot_indices = torch.arange(
                    retrieval_embeds.size(0),
                    device=retrieval_embeds.device,
                    dtype=torch.long,
                ) % self.layer_mixer_logits.size(0)

        weights = self._get_layer_mix_weights(
            retrieval_embeds.size(1),
            retrieval_embeds.device,
            retrieval_slot_indices=retrieval_slot_indices,
        )
        return (retrieval_embeds * weights.unsqueeze(-1)).sum(dim=1)

    def project_retrieval_embeds(self, retrieval_embeds, dtype, device, retrieval_slot_indices=None):
        retrieval_embeds = self.mix_retrieval_layers(
            retrieval_embeds,
            retrieval_slot_indices=retrieval_slot_indices,
        )
        projector_dtype = self.projector.input_proj.weight.dtype
        retrieval_embeds = retrieval_embeds.to(device=device, dtype=projector_dtype)
        retrieval_embeds = self.memory_slot_dropout(retrieval_embeds)
        return self.projector(retrieval_embeds).to(device=device, dtype=dtype)

    def prepare_inputs_embeds(self, input_ids, retrieval_embeds, retrieval_slot_indices=None):
        input_embeds = self.model.embed_tokens(input_ids)
        projected_retrieval_embeds = self.project_retrieval_embeds(
            retrieval_embeds=retrieval_embeds,
            dtype=input_embeds.dtype,
            device=input_embeds.device,
            retrieval_slot_indices=retrieval_slot_indices,
        )

        num_gag_tokens = torch.sum(input_ids == self.gag_token_id).item()
        num_retrieval_embeds = projected_retrieval_embeds.shape[0]
        if num_gag_tokens != num_retrieval_embeds:
            raise ValueError(
                f"Mismatch between <GAG> tokens ({num_gag_tokens}) and retrieval embeds ({num_retrieval_embeds})"
            )

        input_embeds[input_ids == self.gag_token_id] = projected_retrieval_embeds
        return input_embeds, projected_retrieval_embeds

    def project_semantic(self, hidden_repr: torch.Tensor) -> torch.Tensor:
        if self.semantic_head is None:
            raise RuntimeError("semantic_head is not initialized. Set semantic_hidden_size > 0.")
        compute_dtype = self.semantic_head[0].weight.dtype
        return self.semantic_head(hidden_repr.to(dtype=compute_dtype)).float()

    def forward(
        self,
        input_ids=None,
        retrieval_embeds=None,
        retrieval_slot_indices=None,
        attention_mask=None,
        **kwargs,
    ):
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        projected_retrieval_embeds = None
        at_the_beginning_of_generation = inputs_embeds is not None

        if not at_the_beginning_of_generation and retrieval_embeds is not None:
            inputs_embeds, projected_retrieval_embeds = self.prepare_inputs_embeds(
                input_ids=input_ids,
                retrieval_embeds=retrieval_embeds,
                retrieval_slot_indices=retrieval_slot_indices,
            )
            input_ids = None
            if attention_mask is not None and inputs_embeds.shape[1] != attention_mask.shape[1]:
                raise ValueError(
                    f"inputs_embeds and attention_mask length mismatch: {inputs_embeds.shape} vs {attention_mask.shape}"
                )

        outputs = super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        outputs.projected_retrieval_embeds = projected_retrieval_embeds
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        retrieval_embeds=None,
        retrieval_slot_indices=None,
        **kwargs,
    ):
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported for generate")

        if retrieval_embeds is not None:
            inputs_embeds, _ = self.prepare_inputs_embeds(
                input_ids=input_ids,
                retrieval_embeds=retrieval_embeds,
                retrieval_slot_indices=retrieval_slot_indices,
            )
            input_ids = None
            if attention_mask is not None and inputs_embeds.shape[1] != attention_mask.shape[1]:
                raise ValueError(
                    f"inputs_embeds and attention_mask length mismatch: {inputs_embeds.shape} vs {attention_mask.shape}"
                )
            return super().generate(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        return super().generate(
            attention_mask=attention_mask,
            input_ids=input_ids,
            **kwargs,
        )
