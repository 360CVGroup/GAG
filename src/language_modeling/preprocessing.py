import pickle
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from .memory_utils import build_memory_slots_from_record, normalize_layer_keys
from .utils import GAG_TOKEN


DEFAULT_DOMAIN_INSTRUCTION = (
    "You are a professional domain expert. Please produce concise, accurate, and evidence-aware answers."
)
DEFAULT_GAG_REQUEST_TEXT = "Please answer the following question based on the knowledge provided."


def _concat_messages_qwen3(messages, tokenizer):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def _encode_chat_format(messages, tokenizer, max_seq_length, chat_format="qwen3"):
    if chat_format != "qwen3":
        raise ValueError(f"Unsupported chat format: {chat_format}")

    example_text = _concat_messages_qwen3(messages, tokenizer).strip()
    tokenized_example = tokenizer(
        example_text,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    if len(messages) == 2 and messages[0].get("role") == "user":
        user_text = _concat_messages_qwen3(messages[:1], tokenizer) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        user_end_idx = tokenizer(
            user_text,
            return_tensors="pt",
            max_length=max_seq_length,
            truncation=True,
        ).input_ids.shape[1]
        labels[:, :user_end_idx] = -100
    else:
        raise ValueError("messages must contain [user, assistant]")

    return {
        "input_ids": input_ids.flatten(),
        "input_labels": labels.flatten(),
    }


def build_gag_prompt(
    instruction_text: str,
    query: str,
    num_memory_slots: int,
    request_text: str | None = DEFAULT_GAG_REQUEST_TEXT,
) -> str:
    gag_tokens = "".join([GAG_TOKEN] * num_memory_slots)
    request_text = DEFAULT_GAG_REQUEST_TEXT if request_text is None else request_text
    prompt_lines = [instruction_text.strip()]
    if request_text.strip():
        prompt_lines.append(request_text.strip())
    prompt_lines.append(f"Question: {query.strip()}")
    prompt_lines.append(f"Knowledge: {gag_tokens}")
    return "\n".join(prompt_lines) + "\n"


def encode_with_chat_format_finetune(
    example,
    tokenizer,
    max_seq_length,
    chat_format="qwen3",
    num_memory_slots=4,
    instruction_text=None,
    request_text: str = DEFAULT_GAG_REQUEST_TEXT,
):
    instruction = instruction_text or example.get("instruction") or DEFAULT_DOMAIN_INSTRUCTION
    query = example["input"].strip()
    ground_truth = example["output"].strip()

    prompt = build_gag_prompt(
        instruction_text=instruction,
        query=query,
        num_memory_slots=num_memory_slots,
        request_text=request_text,
    )
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ground_truth},
    ]

    encoded = _encode_chat_format(
        messages=messages,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        chat_format=chat_format,
    )
    return {
        "gag_input_ids": encoded["input_ids"],
        "gag_input_labels": encoded["input_labels"],
    }


class train_mlp_Dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int,
        chat_format: str = "qwen3",
        check_presence: bool = True,
        drop_bad_samples: bool = False,
        layer_keys=None,
        num_memory_slots: int = 4,
        slot_pooling: str = "segment_softmax",
        slot_pooling_temperature: float = 1.0,
        instruction_text: str | None = None,
        request_text: str = DEFAULT_GAG_REQUEST_TEXT,
    ):
        self.data_path = data_path
        self.layer_keys = normalize_layer_keys(layer_keys)
        self.num_memory_slots = int(num_memory_slots)
        self.slot_pooling = slot_pooling
        self.slot_pooling_temperature = float(slot_pooling_temperature)
        self.instruction_text = instruction_text
        self.request_text = request_text
        self.records: List[Dict[str, Any]] = self.load_data(data_path)

        if check_presence:
            for index, record in enumerate(self.records):
                missing = [key for key in ["input", "output"] if key not in record]
                if missing:
                    raise KeyError(f"Sample #{index} missing keys: {missing}")

        self.samples: List[Dict[str, Any]] = []
        self.num_skipped = 0

        for index, record in enumerate(self.records):
            try:
                encoded = encode_with_chat_format_finetune(
                    example=record,
                    tokenizer=tokenizer,
                    max_seq_length=max_seq_length,
                    chat_format=chat_format,
                    num_memory_slots=self.num_memory_slots,
                    instruction_text=self.instruction_text,
                    request_text=self.request_text,
                )
                if not (encoded["gag_input_labels"] != -100).any():
                    raise ValueError(
                        f"No supervised answer tokens remain after truncation for sample {record.get('id', index)}"
                    )
                compressed_memory = build_memory_slots_from_record(
                    record=record,
                    layer_keys=self.layer_keys,
                    num_memory_slots=self.num_memory_slots,
                    pooling=self.slot_pooling,
                    temperature=self.slot_pooling_temperature,
                )
                if not torch.isfinite(compressed_memory).all():
                    raise ValueError(
                        f"Non-finite compressed memory detected for sample {record.get('id', index)}"
                    )
                self.samples.append(
                    {
                        "id": record.get("id", index),
                        "gag_input_ids": encoded["gag_input_ids"].to(dtype=torch.long),
                        "gag_input_labels": encoded["gag_input_labels"].to(dtype=torch.long),
                        "answer_tokens_embedding": compressed_memory,
                        "answer_text": record["output"].strip(),
                    }
                )
            except Exception as error:
                if drop_bad_samples:
                    self.num_skipped += 1
                    print(f"Skip sample {index} due to error: {error}")
                    continue
                raise

        if not self.samples:
            raise RuntimeError(f"No valid samples encoded from {data_path}")

        if self.num_skipped > 0:
            print(f"Loaded {len(self.samples)} samples from {data_path}; skipped {self.num_skipped} bad samples.")

    @staticmethod
    def load_data(data_path: str) -> List[Dict[str, Any]]:
        data = []
        with open(data_path, "rb") as file:
            while True:
                try:
                    data.append(pickle.load(file))
                except EOFError:
                    break
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collator(samples, llm_tokenizer):
    def padding(input_ids, labels=None, padding_side="right"):
        def _padding(ids, padding_value, padding_side="right"):
            if padding_side == "right":
                return torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=padding_value)
            flipped_ids = [torch.flip(x, dims=[0]) for x in ids]
            return torch.flip(
                torch.nn.utils.rnn.pad_sequence(flipped_ids, batch_first=True, padding_value=padding_value),
                dims=[1],
            )

        input_ids = _padding(
            input_ids,
            padding_value=llm_tokenizer.pad_token_id,
            padding_side=padding_side,
        )
        attention_mask = (input_ids != llm_tokenizer.pad_token_id).long()
        if labels is not None:
            labels = _padding(labels, padding_value=-100, padding_side=padding_side)
        return input_ids, attention_mask, labels

    gag_input_ids, gag_attention_mask, gag_input_labels = padding(
        input_ids=[sample["gag_input_ids"] for sample in samples],
        labels=[sample["gag_input_labels"] for sample in samples],
        padding_side=llm_tokenizer.padding_side,
    )

    embeds_flattened = []
    memory_slots_per_sample = []
    retrieval_slot_indices = []
    for sample in samples:
        embedding = sample["answer_tokens_embedding"]
        if embedding.dim() != 3:
            raise ValueError(
                f"Expected [num_layers, num_slots, hidden_size], got {tuple(embedding.shape)}"
            )
        embeds_flattened.append(embedding.permute(1, 0, 2).contiguous())
        num_slots = embedding.size(1)
        memory_slots_per_sample.append(num_slots)
        retrieval_slot_indices.append(torch.arange(num_slots, dtype=torch.long))

    retrieval_embeds = torch.cat(embeds_flattened, dim=0)
    return {
        "sample_ids": [sample["id"] for sample in samples],
        "gag_input_ids": gag_input_ids,
        "gag_attention_mask": gag_attention_mask,
        "gag_input_labels": gag_input_labels,
        "retrieval_embeds": retrieval_embeds,
        "retrieval_slot_indices": torch.cat(retrieval_slot_indices, dim=0),
        "memory_slots_per_sample": torch.tensor(memory_slots_per_sample, dtype=torch.long),
        "answer_texts": [sample["answer_text"] for sample in samples],
    }
