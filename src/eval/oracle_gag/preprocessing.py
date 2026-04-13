import pickle
from typing import Any, Dict, List

from src.language_modeling.memory_utils import build_memory_slots_from_record, normalize_layer_keys
from src.language_modeling.preprocessing import _concat_messages_qwen3, build_gag_prompt


def load_data(data_path: str) -> List[Dict[str, Any]]:
    data = []
    with open(data_path, "rb") as file:
        while True:
            try:
                data.append(pickle.load(file))
            except EOFError:
                break
    return data


def prepare_prompts(
    test_data,
    tokenizer,
    chat_format="qwen3",
    layer_keys=None,
    num_memory_slots=4,
    slot_pooling="segment_softmax",
    slot_pooling_temperature=1.0,
    instruction_text=None,
    request_text=None,
):
    if chat_format != "qwen3":
        raise ValueError(f"Unsupported chat format: {chat_format}")

    normalized_layer_keys = normalize_layer_keys(layer_keys)
    prompts = []
    backgrounds = []

    for sample in test_data:
        instruction = instruction_text or sample.get("instruction")
        prompt = build_gag_prompt(
            instruction_text=instruction,
            query=sample["input"],
            num_memory_slots=num_memory_slots,
            request_text=request_text if request_text is not None else None,
        )
        messages = [{"role": "user", "content": prompt}]
        user_prompt = _concat_messages_qwen3(messages, tokenizer) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"

        memory_slots = build_memory_slots_from_record(
            record=sample,
            layer_keys=normalized_layer_keys,
            num_memory_slots=num_memory_slots,
            pooling=slot_pooling,
            temperature=slot_pooling_temperature,
        )

        prompts.append(user_prompt)
        backgrounds.append(memory_slots.permute(1, 0, 2).contiguous())

    return prompts, backgrounds
