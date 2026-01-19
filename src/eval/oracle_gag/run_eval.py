## built-in
import argparse,json,os
import time
## third party
from transformers import (
    Qwen3ForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
import torch
import datasets
from tqdm import tqdm
import pandas as pd

## own
from src.model import (
    XQwen3ForCausalLM,
)
from src.language_modeling.utils import (
    GAG_TOKEN,
    get_yaml_file,
)
from src.eval.oracle_gag.utils import (
    llm_for_open_generation,
    save_with_answers,
)
from src.eval.oracle_gag.preprocessing import(
    load_data,
    prepare_prompts,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="config file to launch the eval"
    )
    parser.add_argument(
        "--data_path",
    )
    parser.add_argument(
        "--output_file_path",
        type=str,
        help="jsonl file with Qwen3-8B answer based on small model's background"
    )
    parser.add_argument(
        "--enable_progress_bar",
        type=eval,
        default=True,
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="large language model(with finetuned projector)'s path"
    )
    parser.add_argument(
        "--retrieval_embed_length",
        type=int,default=1,
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        help="for debug",
    )
    # parser.add_argument(
    #     "--save_dir",
    # )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--chat_format",
        default='qwen3',
    )
    parser.add_argument(
        "--embed_key",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    yaml_config = get_yaml_file(args.config)

    ## priority: CLI > YAML (with all default value set to None in argument parser)
    for k,v in yaml_config.items():
        assert hasattr(args,k), f"{k} not in parsed arguments"
        if getattr(args,k) is None:
            setattr(args,k,v)

    if args.embed_key is None:
        args.embed_key = "last_layer_answer_tokens_embedding"

    return args


def main():
    args = parse_args()

    ## load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side = 'left',
        add_eos_token=False, ## import to include this!
        use_fast=False,
    )
    if tokenizer.pad_token:   # my own: '<|endoftext|>'
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    ## load retriever and retriever_tokenizer
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    retriever_hidden_size = 2048    
    retriever,retriever_tokenizer = None,None

    test_data = load_data(      # List[Dict[str, Any]]
        args.data_path,
    )

    if args.max_test_samples is not None:
        test_data = test_data[:args.max_test_samples]

    prompts,backgrounds = prepare_prompts(    # my own： prompts: List[string]   backgrounds: List[torch.tensor]   # backgrounds : List[一个形状为1,length,2048的张量, 一个形状为1,length,2048的张量, 一个形状为1,length,2048的张量……]
        test_data = test_data,
        tokenizer = tokenizer,
        chat_format = args.chat_format, 
        embed_key=args.embed_key,    # ★新增
    )        
    assert len(prompts) == len(backgrounds)

    retrieval_embeds = None

    # backgrounds List[List[String]]
    num_samples = len(backgrounds)

    retrieval_embeds = [[] for _ in range(num_samples)]
    for id,embeds in enumerate(backgrounds, start=0):
        retrieval_embeds[id].append(embeds)

    avg_prompt_length = tokenizer(prompts,return_length=True).length      # len(prompts) = 200  len(avg_prompt_length) = 200
    avg_prompt_length = sum(avg_prompt_length)/len(avg_prompt_length)     # 48.15

    ## load llm
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    MODEL_CLASS = eval(config.architectures[0])
    model = MODEL_CLASS.from_pretrained(
        args.model_name_or_path,
        torch_dtype = torch.bfloat16,
        low_cpu_mem_usage = True,
    ).to(device)
    
    model.eval()
    # model = model.to(device)
    assert GAG_TOKEN in tokenizer.get_vocab() 
    model.set_gag_token_id(tokenizer.convert_tokens_to_ids(GAG_TOKEN))

    generated_results = llm_for_open_generation(    # len(generated_results) = 200
        llm = model,
        llm_tokenizer = tokenizer,
        prompts = prompts,    # my own: len(prompts): 3559
        retrieval_embeds = retrieval_embeds,       # len(retrieval_embeds) = 200        retrieval_embeds[0][0].shape = torch.Size([4096])
                                                   # my own: [[一个形状为1,length,2048的张量],[],[]……]  len(retrieval_embeds) = 3559
        batch_size = args.eval_batch_size,     # 4 
        enable_progress_bar= args.enable_progress_bar,    # true
    )
    
    assert len(generated_results) == num_samples

    save_with_answers(
        test_data = test_data,
        generated_results = generated_results,
        output_file_path = args.output_file_path,
    )

    
if __name__ == "__main__":
    main()