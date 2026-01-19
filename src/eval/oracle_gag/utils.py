import os,json
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from tqdm import tqdm


@torch.no_grad()
def llm_for_open_generation(
    llm,llm_tokenizer,
    prompts,
    retrieval_embeds,
    batch_size = 4,
    enable_progress_bar = True,
):

    # gen_cfg = llm.generation_config
    # gen_cfg.do_sample = False
    # gen_cfg.temperature = None
    # gen_cfg.top_p = None
    # gen_cfg.top_k = None

    generated_answers = []
    total_test_number = len(prompts)    # my own: 3559
    device = llm.device   # device(type='cuda', index=0)
    batched_prompts = [prompts[idx:idx+batch_size] for idx in range(0,len(prompts),batch_size)]      # my own: len(batched_prompts) = 890
    if retrieval_embeds is not None:    # my own: [[一个形状为1,length,2048的张量],[],[]……]  len(retrieval_embeds) = 3559
        batched_retrieval_embeds = [retrieval_embeds[idx:idx+batch_size] for idx in range(0,len(retrieval_embeds),batch_size)]     
        # my own: batched_retrieval_embeds  [[ [一个形状为1,length,2048的张量],[一个形状为1,length,2048的张量],[],[] ], ……]   
        # len(batched_retrieval_embeds) = 890
        # len(batched_retrieval_embeds[0]) = 4
        # batched_retrieval_embeds[0][0][0].shape : [1,65,2048]  
        # batched_retrieval_embeds[0][1][0].shape : [1,199,2048]
        # batched_retrieval_embeds[2][1][0].shape : [1,200,2048]

        assert len(batched_prompts) == len(batched_retrieval_embeds)
    
    progress_bar = tqdm(range(total_test_number),ncols=60,disable= not enable_progress_bar)
    for batch_idx in range(len(batched_prompts)):
        prompt = batched_prompts[batch_idx]    # my own : List[string,string,string,string,]
        tokenized_propmt = llm_tokenizer(prompt,padding='longest',return_tensors='pt')  
        #  my own: {'input_ids': tensor([[151643, 151643, 151643,  ...,    271, 151668,    271],
        # [151644,    872,    198,  ...,    271, 151668,    271],
        # [151643, 151643, 151643,  ...,    271, 151668,    271],
        # [151643, 151643, 151643,  ...,    271, 151668,    271]]), 'attention_mask': tensor([[0, 0, 0,  ..., 1, 1, 1],
        # [1, 1, 1,  ..., 1, 1, 1],
        # [0, 0, 0,  ..., 1, 1, 1],
        # [0, 0, 0,  ..., 1, 1, 1]])}
        
        # tokenized_prompt.input_ids.shape : [4,330]
        # tokenized_prompt.attention_mask.shape: [4,330]
        input_ids = tokenized_propmt.input_ids.to(device)    # my own: [4,330]
        attention_mask = tokenized_propmt.attention_mask.to(device)  # my own: [4,330]
        # stopping_criteria = stop_sequences_criteria(llm_tokenizer, input_ids.shape[1], input_ids.shape[0])     # 这一句代码报Keyword arguments {'add_special_tokens': False} not recognized.
        #stopping_criteria = stop_sequences_criteria(llm_tokenizer, 512, input_ids.shape[0])
        retrieval_kwargs = {}
        embeds = []
        if retrieval_embeds is not None:
            embeds = batched_retrieval_embeds[batch_idx]                        # my own: retrieval_embeds: [[一个形状为1,length,2048的张量],[],[]……]
                                                                                # my own: batched_retrieval_embeds : [[ [一个形状为1,length,2048的张量],[],[],[] ], ……]
                                                                                # embeds: [[一个形状为1,length,2048的张量],[一个形状为1,length,2048的张量],[一个形状为1,length,2048的张量],[一个形状为1,length,2048的张量]]
                                                                                # embeds[0][0].shape: [1,65,2048]
                                                                                # embeds[1][0].shape: [1,199,2048]
            embeds = [x for y in embeds for x in y]   
            # my own: [ 一个形状为1,length,2048的张量, 一个形状为1,length,2048的张量,一个形状为1,length,2048的张量 , 一个形状为1,length,2048的张量, ]
            # len(embeds) = 4
            # embeds[0].shape: [1,65,2048]  embeds[1].shape: [1,199,2048]  embeds[2].shape: [1,34,2048]  embeds[3].shape: [1,25,2048]
            embeds = torch.cat(embeds, dim=1).to(device)   
            # my own: embeds: [1,length_1+length_2+……+length_batch_size,2048]
            # embeds.shape: [1,323,2048]

            embeds = embeds.view(-1,2048)  # my own: embeds: [length_1+length_2+……+length_batch_size,2048]  embeds.shape: [323,2048]
            retrieval_kwargs['retrieval_embeds'] = embeds
            # stopping_criteria = stop_sequences_criteria(llm_tokenizer, 0, input_ids.shape[0])      # 这一句代码报Keyword arguments {'add_special_tokens': False} not recognized.

        ## actual computation
        generated_output = llm.generate(     
            input_ids = input_ids,    # my own: [4,330]
            attention_mask = attention_mask,   # my own: [4,330]
            # stopping_criteria=stopping_criteria,
            do_sample=True,   # my own: 先设置为false，好debug，后面记得修改
            max_new_tokens=3000,
            pad_token_id=llm_tokenizer.pad_token_id,  # my own: 151643
            use_cache=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            **retrieval_kwargs,     # my own: [length_1+length_2+……+length_batch_size,2048]   retrieval_kwargs['retrieval_embeds'].shape: [323,2048]
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
        )
        ## because HF generate with inputs_embeds would not return prompt
        input_length = 0 if retrieval_kwargs else input_ids.shape[1]    # 0
        results = llm_tokenizer.batch_decode(generated_output[:,input_length:],skip_special_tokens=True)  # len(results) = 4
        # my own: generated_output.shape: torch.Size([4,100])
        # type(results): <class 'list'>
        generated_answers.extend(results)
        progress_bar.update(batch_size)

    generated_answers = [x.strip() for x in generated_answers]
    return generated_answers
    


def save_with_answers(test_data, generated_results, output_file_path):
    # 打开输出文件进行写入
    with open(output_file_path, 'w') as output_f:
        for idx, item in enumerate(test_data):
            # 获取当前项的 ID 和其他字段
            rec_id = item["id"]
            answer_text = generated_results[idx]  # 对应生成的回答

            # 可以从 `test_data` 中获取相应的字段
            instruction = item["instruction"]
            input_text = item["input"]
            output_text = item["output"]
            small_model_answer_background = item["qwen3-1.7B_answer_background"]
            # answer_tokens_hidden = item.get("last_layer_answer_tokens_embedding")

            # 创建要保存的字典，包含新的字段 'Qwen3-8B_answer'
            result = {
                'id': rec_id,
                'instruction': instruction,
                'input': input_text,
                'output': output_text,
                'qwen3-1.7B_answer_background': small_model_answer_background,
                # 'last_layer_answer_tokens_embedding': answer_tokens_hidden,
                'Qwen3-8B_answer': answer_text,
            }

            # 写入一条数据并立即刷新到文件
            output_f.write(json.dumps(result) + '\n')

    print(f"Results successfully written to {output_file_path}")

# 调用这个函数，传入生成的结果和 `test_data`
# output_file_path = '/home/jovyan/projection/MinerU/adjuvant_code/7.20_pipeline_code/train_mlp_8_15/src/eval/results/Adjuvant_QA_unimodal_withoutocr_pure_QA_shuffle_with_id_1_with_answer_embeddding_with_qwen3_answer.pkl'
# save_with_answers(test_data, generated_results, output_file_path)
