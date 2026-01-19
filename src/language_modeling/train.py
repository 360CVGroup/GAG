## built-in
import argparse
import logging
import math
import os
import random
import types
import pickle,json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_IGNORE_GLOBS"]='*.pth' ## not upload ckpt to wandb cloud

## third-party
import datasets
import torch
import torch.distributed as dist
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import copy
import transformers
from transformers import (
    AutoTokenizer,
    Qwen2Tokenizer,
    Qwen2TokenizerFast,
    SchedulerType,
    get_scheduler,
)
import deepspeed
from tokenizers import AddedToken
import wandb

# own
from src.model import (
    XQwen3Config, 
    XQwen3ForCausalLM,
)  # src.model 在这里是一个 Python 包（文件夹下有 __init__.py，才能被当成包）。
   # import ... 时，Python 会先加载 src/model/__init__.py 文件。
   # 然后，它会在 src/model/__init__.py 的命名空间里去找 XQwen3Config, XQwen3ForCausalLM 这些符号。

from src.language_modeling.utils import (
    GAG_TOKEN,
    get_yaml_file,
    get_nll_loss,
    save_with_accelerate,
)

from src.language_modeling.preprocessing import (
    encode_with_chat_format_finetune,
    train_mlp_Dataset,
    collator,
)

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chat_format",
        choices=['mistral','tulu','mixtral','qwen3','yi','gemma']
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
    )
    parser.add_argument(
        "--update_projector_only",
        type=eval,
    )
    parser.add_argument(
        "--workdir",
        type=str,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="config file to launch the training"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        help="pretrain or finetune"
    )
    # parser.add_argument(
    #     "--retrieval_context_length",
    #     type=int,
    #     help="max token number for document encoder in dense retrieval",
    # )
    parser.add_argument(
        "--embed_key",
        type=str,
        default=None,
        help="Which embedding field in pkl to use, e.g. last_layer_answer_tokens_embedding or layer_minus2_answer_tokens_embedding",
    )
    parser.add_argument(
        "--alpha_nll",
        type=float,
        help="coefficient for multi-task learning",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A pkl file containing the training data."
    )
    parser.add_argument(
        "--dev_file", type=str, default=None, help="A pkl file containing the dev data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--use_flash_attn",
        type=eval,
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--exp_note", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=eval, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=eval,
        help=(
            "Turn on gradient checkpointing. Saves memory but slows training."
        ),
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
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

    args.train_file = os.path.join(args.workdir,args.train_file)
    if args.dev_file is not None:args.dev_file = os.path.join(args.workdir,args.dev_file)
    if os.path.isdir(os.path.join(args.workdir,args.model_name_or_path)):
        args.model_name_or_path = os.path.join(args.workdir,args.model_name_or_path)

    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    retriever_hidden_size = 2048

    # === 指定 wandb 的保存根目录 ===
    wandb_dir = "/home/jovyan/lirongji/wandb_save"
    os.makedirs(wandb_dir, exist_ok=True)

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with="wandb")
    accelerator.init_trackers(
        project_name=args.project_name, 
        config=args,
        init_kwargs={
            "wandb": {
                "dir": wandb_dir, 
                "name": args.exp_name if args.exp_name is not None else None,
                "notes": args.exp_note if args.exp_note is not None else None,
                "save_code": True,
            },
        }
    )
    accelerator.print(json.dumps(vars(args),indent=4))
    checkpoint_dir = [None]
    if accelerator.is_local_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        checkpoint_dir = [os.path.join(wandb_tracker.run.dir,'checkpoint')]
    if accelerator.use_distributed:
        dist.broadcast_object_list(checkpoint_dir,src=0)
    args.output_dir = checkpoint_dir[0]

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()


    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    if args.chat_format == 'qwen3':
        MODEL_CLASS,CONFIG_CLASS = XQwen3ForCausalLM,XQwen3Config
        tokenizer.padding_side = 'left'
    config = CONFIG_CLASS.from_pretrained(args.model_name_or_path,retriever_hidden_size=retriever_hidden_size)

    # 开启/关闭 FA2 用 config.attn_implementation，而不是传 use_flash_attention_2
    if args.use_flash_attn:
        config.attn_implementation = "flash_attention_2"   # 备选: "sdpa" 或 "eager"

    model = MODEL_CLASS.from_pretrained(
        args.model_name_or_path,
        config=config,
        # use_flash_attention_2=args.use_flash_attn,
        torch_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' else 'auto',
    )

    num_added_tokens = 0
    ## qwen3 tokenizer is a Qwen2Tokenizer
    # if isinstance(tokenizer, Qwen2Tokenizer) or isinstance(tokenizer, Qwen2TokenizerFast):
    #     num_added_tokens = tokenizer.add_special_tokens({
    #         "pad_token": "<pad>",
    #     })
    #     assert num_added_tokens in [0, 1], "Qwen2Tokenizer should only add one special token - the pad_token, or no tokens if pad token present."


    ## GAG_TOKEN simply functions as a placeholder, would not be trained
    num_added_tokens += tokenizer.add_tokens([AddedToken(GAG_TOKEN,lstrip=False,rstrip=False)])   # original: special_tokens={'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>',
    # after: 
    gag_token_id = tokenizer.convert_tokens_to_ids(GAG_TOKEN)  # 151669
    model.set_gag_token_id(gag_token_id)
    if num_added_tokens > 0:
        #model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
        model.resize_token_embeddings(len(tokenizer))
    vocab_size = len(tokenizer)

    # load datasets and preprocess datasets
    dev_dataset = None  

    if args.train_file is not None:
        train_dataset = train_mlp_Dataset(
            data_path=args.train_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            chat_format="qwen3",
            check_presence=True,
            drop_bad_samples=False,   # 想跳过坏样本就改 True
            embed_key=args.embed_key,          # ★新增
        )

    if args.dev_file is not None:
        dev_dataset = train_mlp_Dataset(
            data_path=args.dev_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            chat_format="qwen3",
            check_presence=True,
            drop_bad_samples=False,   # 想跳过坏样本就改 True
            embed_key=args.embed_key,          # ★新增
        )



    ## select N samples, mainly for debug
    # if args.max_train_samples is not None and len(raw_datasets['train']) > args.max_train_samples:
    #     selected_indices = random.sample(range(len(raw_datasets['train'])),args.max_train_samples)
    #     raw_datasets['train'] = raw_datasets['train'].select(selected_indices)
    
    # if args.exclude_dataset_type is not None:
    #     for d_type in args.exclude_dataset_type:
    #         raw_datasets['train'] = raw_datasets['train'].filter(lambda  example:example['task_type']!=d_type)
    
    collate_fn = partial(
        collator,
        llm_tokenizer=tokenizer,
    )

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size
    )

    dev_dataloader = None
    if dev_dataset is not None:
        dev_dataloader = DataLoader(
            dev_dataset,
            shuffle=False, 
            collate_fn=collate_fn,
            batch_size=args.per_device_train_batch_size
        )
    
    if args.update_projector_only:
        for n,p in model.named_parameters():
            if 'projector' not in n:p.requires_grad = False
            else:p.requires_grad = True
                
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume 
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total 
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the 
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )
    
    # # https://github.com/microsoft/DeepSpeed/pull/4966
    # if args.chat_format == 'mixtral':
    #     deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    # Prepare everything with `accelerator`.
    if dev_dataset is not None:
        model, optimizer, train_dataloader, lr_scheduler, dev_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, dev_dataloader)

    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)


    

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)



    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Max Sequence Length = {args.max_seq_length}")
    logger.info(f"  Trainable Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)/(10**6):.2f} M") ## not applicable for deepspeed

    completed_steps = 0
    starting_epoch = 0

    # logging_interval_grad_norm = 0
    logging_interval_loss = 0
    logging_interval_nll_loss = 0
    
    total_loss = 0
    total_nll_loss = 0

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    # progress_bar = tqdm(range(args.max_train_steps), disable=True)

    # update the progress_bar if load from checkpoint
    save_one_sample = True

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        active_dataloader = train_dataloader

        for batch in active_dataloader:
            if save_one_sample:
                if accelerator.is_local_main_process:
                    pickle.dump(
                        batch,
                        open(os.path.join(os.path.dirname(args.output_dir),"sample_data.pkl"),'wb'),
                    )
                accelerator.print("**"*20,"show one example","**"*20)
                accelerator.print(batch.keys())
                accelerator.print(tokenizer.decode(batch['gag_input_ids'][0]))
                accelerator.print(batch['gag_input_ids'][0])
                # if "retriever_input_text" in batch:
                #     accelerator.print(batch['retriever_input_text'][0])
                # if 'input_ids' in batch:
                #     for input_id,label_id,attention_mask in zip(batch['input_ids'][0],batch['labels'][0],batch['attention_mask'][0]):
                #         accelerator.print(f"{tokenizer.convert_ids_to_tokens([input_id])[0]}({label_id.item()})({attention_mask})",end=" ")
                accelerator.print()    
                for input_id,label_id,attention_mask in zip(batch['gag_input_ids'][0],batch['gag_input_labels'][0],batch['gag_attention_mask'][0]):
                    accelerator.print(f"{tokenizer.convert_ids_to_tokens([input_id])[0]}({label_id.item()})({attention_mask})",end=" ")
                accelerator.print('\n'+"**"*20,"show one example","**"*20)
                save_one_sample=False

            with accelerator.accumulate(model):
                ## forward with retrieval embeds
                # retrieval_kwargs = {}
                # retrieval_kwargs['retrieval_embeds'] = get_retrieval_embeds(batch)     #[2,2048]

                outputs = model(
                    input_ids = batch['gag_input_ids'],     #[2,70]
                    attention_mask = batch['gag_attention_mask'],    #[2,70]
                    # **retrieval_kwargs,   #[2,2048]
                    retrieval_embeds = batch['retrieval_embeds']
                )    # outputs.keys() include: 'logits','past_key_values'     outputs.logits.shape: [2,70,32002]
                loss = None
                if args.alpha_nll is not None and args.alpha_nll > 0.0:
                    
                    nll_loss = get_nll_loss(
                        labels = batch['gag_input_labels'],    # [2,70]
                        logits = outputs.logits,     # [2,70,32002]
                        vocab_size = vocab_size,
                    )

                    logging_interval_nll_loss += nll_loss.detach().float()

                    loss = args.alpha_nll * nll_loss

                logging_interval_loss += loss.detach().float()
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()       

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(logging_interval_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    total_loss += accelerator.gather(logging_interval_loss).mean().item() / args.gradient_accumulation_steps 

                    to_be_logged = {
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "train_loss": avg_loss,
                        "rolling_loss":total_loss / completed_steps,
                    }
                    if args.alpha_nll is not None and args.alpha_nll > 0.0:
                        total_nll_loss += accelerator.gather(logging_interval_nll_loss).mean().item() / args.gradient_accumulation_steps
                        to_be_logged["rolling_nll_loss"] = total_nll_loss  / completed_steps

                    # if args.alpha_kl is not None and args.alpha_kl > 0.0:
                    #     total_kl_loss  += accelerator.gather(logging_interval_kl_loss ).mean().item() / args.gradient_accumulation_steps
                    #     to_be_logged["rolling_kl_loss"] = total_kl_loss  / completed_steps

                    accelerator.log(to_be_logged,step=completed_steps)
                    
                    # logging_interval_grad_norm = 0
                    logging_interval_loss = 0
                    logging_interval_nll_loss = 0
                    
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        # output_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
                        # save_with_accelerate(accelerator, model, tokenizer, output_dir,save_projector_only=args.update_projector_only)
                        base_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
                        # 保存 full
                        save_with_accelerate(
                            accelerator, model, tokenizer,
                            os.path.join(base_dir, "full_weight"),
                            save_projector_only=False
                        )
                        # 保存 only projector
                        save_with_accelerate(
                            accelerator, model, tokenizer,
                            os.path.join(base_dir, "only_projector_weight"),
                            save_projector_only=True
                        )

                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            base_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            # 保存 full
            save_with_accelerate(
                accelerator, model, tokenizer,
                os.path.join(base_dir, "full_weight"),
                save_projector_only=False
            )
            # 保存 only projector
            save_with_accelerate(
                accelerator, model, tokenizer,
                os.path.join(base_dir, "only_projector_weight"),
                save_projector_only=True
            )

    accelerator.end_training()

    ## save the last one
    base_dir = os.path.join(args.output_dir,"last")
    save_with_accelerate(
        accelerator, model, tokenizer,
        os.path.join(base_dir, "full_weight"),
        save_projector_only=False
    )
    save_with_accelerate(
        accelerator, model, tokenizer,
        os.path.join(base_dir, "only_projector_weight"),
        save_projector_only=True
    )
    # save_with_accelerate(accelerator, model, tokenizer, output_dir,save_projector_only=False)
    # output_dir = os.path.join(args.output_dir,"last_only_projector")
    # save_with_accelerate(accelerator, model, tokenizer, output_dir,save_projector_only=True)


if __name__ == "__main__":
    main()