import json
import os

import torch
from tqdm import tqdm


@torch.no_grad()
def llm_for_open_generation(
    llm,
    llm_tokenizer,
    prompts,
    retrieval_embeds,
    batch_size=4,
    enable_progress_bar=True,
    do_sample=False,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.05,
    no_repeat_ngram_size=3,
):
    generated_answers = []
    device = llm.device

    batched_prompts = [prompts[idx : idx + batch_size] for idx in range(0, len(prompts), batch_size)]
    batched_retrieval = None
    if retrieval_embeds is not None:
        batched_retrieval = [
            retrieval_embeds[idx : idx + batch_size] for idx in range(0, len(retrieval_embeds), batch_size)
        ]

    progress_bar = tqdm(range(len(prompts)), ncols=80, disable=not enable_progress_bar)
    for batch_idx, prompt_batch in enumerate(batched_prompts):
        tokenized = llm_tokenizer(prompt_batch, padding="longest", return_tensors="pt").to(device)
        retrieval_kwargs = {}
        if batched_retrieval is not None:
            memory_tensors = batched_retrieval[batch_idx]
            memory_batch = torch.cat(memory_tensors, dim=0).to(device)
            retrieval_kwargs["retrieval_embeds"] = memory_batch
            retrieval_kwargs["retrieval_slot_indices"] = torch.cat(
                [
                    torch.arange(sample_memory.size(0), dtype=torch.long)
                    for sample_memory in memory_tensors
                ],
                dim=0,
            ).to(device)

        generation_kwargs = {
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "pad_token_id": llm_tokenizer.pad_token_id,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            **retrieval_kwargs,
        }
        if do_sample:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                }
            )
        else:
            generation_kwargs["do_sample"] = False

        generated = llm.generate(**generation_kwargs)
        input_length = 0 if retrieval_kwargs else tokenized.input_ids.shape[1]
        generated_answers.extend(
            answer.strip()
            for answer in llm_tokenizer.batch_decode(generated[:, input_length:], skip_special_tokens=True)
        )
        progress_bar.update(len(prompt_batch))

    return generated_answers


def save_with_answers(test_data, generated_results, output_file_path):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for index, item in enumerate(test_data):
            result = {
                "id": item.get("id", index),
                "instruction": item.get("instruction"),
                "input": item.get("input"),
                "output": item.get("output"),
                "qwen3-1.7B_answer_background": item.get("qwen3-1.7B_answer_background"),
                "Qwen3-8B_answer": generated_results[index],
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
