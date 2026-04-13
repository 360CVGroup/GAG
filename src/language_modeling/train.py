import argparse
import json
import logging
import math
import os
from functools import partial

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tokenizers import AddedToken
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)

from src.language_modeling.memory_utils import normalize_layer_keys
from src.language_modeling.preprocessing import collator, train_mlp_Dataset
from src.language_modeling.utils import (
    GAG_TOKEN,
    get_latent_semantic_loss,
    get_memory_diversity_loss,
    get_nll_loss,
    get_yaml_file,
    masked_mean_pool,
    mean_pool_encoder_outputs,
    resolve_path,
    save_with_accelerate,
    split_projected_memory_by_sample,
)
from src.model import XQwen3Config, XQwen3ForCausalLM


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--workdir", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--dev_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--chat_format", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=str, default=None)
    parser.add_argument("--clip_grad_norm", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_flash_attn", type=eval, default=None)
    parser.add_argument("--gradient_checkpointing", type=eval, default=None)
    parser.add_argument("--update_projector_only", type=eval, default=None)
    parser.add_argument("--projector_type", type=str, default=None)
    parser.add_argument("--retriever_hidden_size", type=int, default=None)
    parser.add_argument("--layer_keys", nargs="*", default=None)
    parser.add_argument("--embed_key", type=str, default=None)
    parser.add_argument("--num_memory_slots", type=int, default=None)
    parser.add_argument("--slot_pooling", type=str, default=None)
    parser.add_argument("--slot_pooling_temperature", type=float, default=None)
    parser.add_argument("--instruction_text", type=str, default=None)
    parser.add_argument("--request_text", type=str, default=None)
    parser.add_argument("--alpha_nll", type=float, default=None)
    parser.add_argument("--alpha_semantic", type=float, default=None)
    parser.add_argument("--alpha_diversity", type=float, default=None)
    parser.add_argument("--use_layer_mix", type=eval, default=None)
    parser.add_argument("--memory_slot_dropout", type=float, default=None)
    parser.add_argument("--layer_mix_logit_clamp", type=float, default=None)
    parser.add_argument("--semantic_model_name_or_path", type=str, default=None)
    parser.add_argument("--semantic_max_length", type=int, default=None)
    parser.add_argument("--report_to_wandb", type=eval, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    yaml_config = get_yaml_file(args.config)
    for key, value in yaml_config.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    if args.embed_key and not args.layer_keys:
        args.layer_keys = [args.embed_key]
    args.layer_keys = normalize_layer_keys(args.layer_keys)

    args.workdir = resolve_path(None, args.workdir or os.path.dirname(os.path.abspath(args.config)))
    args.train_file = resolve_path(args.workdir, args.train_file)
    args.dev_file = resolve_path(args.workdir, args.dev_file)
    args.model_name_or_path = resolve_path(args.workdir, args.model_name_or_path)
    args.output_dir = resolve_path(args.workdir, args.output_dir)
    args.semantic_model_name_or_path = resolve_path(args.workdir, args.semantic_model_name_or_path)
    return args


def load_semantic_encoder(args, accelerator):
    if not args.alpha_semantic or args.alpha_semantic <= 0.0:
        return None, None, 0
    if not args.semantic_model_name_or_path:
        raise ValueError("alpha_semantic > 0 requires semantic_model_name_or_path")

    semantic_tokenizer = AutoTokenizer.from_pretrained(args.semantic_model_name_or_path, trust_remote_code=True)
    semantic_model = AutoModel.from_pretrained(
        args.semantic_model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if accelerator.mixed_precision == "bf16" else None,
    )
    semantic_model.eval()
    for parameter in semantic_model.parameters():
        parameter.requires_grad = False

    semantic_hidden_size = getattr(semantic_model.config, "hidden_size", None)
    if semantic_hidden_size is None:
        raise ValueError("Semantic encoder config must expose hidden_size")
    return semantic_tokenizer, semantic_model, int(semantic_hidden_size)


def build_optimizer(model, args):
    if args.update_projector_only:
        for name, parameter in model.named_parameters():
            trainable = any(
                key in name for key in ["projector", "layer_mixer_logits", "semantic_head"]
            )
            parameter.requires_grad = trainable
            if trainable and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check update_projector_only and model config.")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    return optimizer, trainable_params


def _format_log_value(value):
    if torch.is_tensor(value):
        value = value.detach().float().item()
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _collect_non_finite_param_names(parameters, prefix):
    bad = []
    for name, parameter in parameters:
        if not torch.isfinite(parameter).all():
            bad.append(f"{prefix}:{name}")
    return bad


def sanitize_projector_side_params(model):
    sanitized = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not any(key in name for key in ["projector", "layer_mixer_logits", "semantic_head"]):
                continue

            target = parameter.data.float()
            if "layer_mixer_logits" in name:
                if not torch.isfinite(target).all():
                    sanitized.append(f"{name}:reset_zero_from_nonfinite")
                target.zero_()
            elif not torch.isfinite(target).all():
                target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
                sanitized.append(f"{name}:nan_to_num")

            if parameter.dtype != torch.float32 or target.data_ptr() != parameter.data.data_ptr():
                parameter.data = target.contiguous()

    return sanitized


def main():
    args = parse_args()
    set_seed(args.seed)

    report_to = "wandb" if args.report_to_wandb else None
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=report_to,
    )
    if args.report_to_wandb:
        accelerator.init_trackers(
            project_name=args.wandb_project or "GAG",
            config=vars(args),
            init_kwargs={"wandb": {"name": args.wandb_run_name}},
        )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(json.dumps(vars(args), indent=2), main_process_only=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    semantic_tokenizer, semantic_model, semantic_hidden_size = load_semantic_encoder(args, accelerator)

    config = XQwen3Config.from_pretrained(
        args.model_name_or_path,
        retriever_hidden_size=args.retriever_hidden_size,
        projector_type=args.projector_type,
        use_layer_mix=args.use_layer_mix,
        layer_mix_num_layers=len(args.layer_keys),
        num_memory_slots=args.num_memory_slots,
        memory_slot_dropout=args.memory_slot_dropout,
        layer_mix_logit_clamp=args.layer_mix_logit_clamp,
        semantic_hidden_size=semantic_hidden_size,
    )
    if args.use_flash_attn:
        config.attn_implementation = "flash_attention_2"

    model = XQwen3ForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16 if accelerator.mixed_precision == "bf16" else "auto",
    )
    sanitized_params = sanitize_projector_side_params(model)
    if sanitized_params:
        logger.warning(
            "Sanitized projector-side parameters after load: %s",
            ", ".join(sanitized_params[:20]),
            main_process_only=True,
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    num_added_tokens = tokenizer.add_tokens([AddedToken(GAG_TOKEN, lstrip=False, rstrip=False)])
    model.set_gag_token_id(tokenizer.convert_tokens_to_ids(GAG_TOKEN))
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    vocab_size = len(tokenizer)

    train_dataset = train_mlp_Dataset(
        data_path=args.train_file,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        chat_format=args.chat_format,
        check_presence=True,
        drop_bad_samples=True,
        layer_keys=args.layer_keys,
        num_memory_slots=args.num_memory_slots,
        slot_pooling=args.slot_pooling,
        slot_pooling_temperature=args.slot_pooling_temperature,
        instruction_text=args.instruction_text,
        request_text=args.request_text,
    )

    dev_dataset = None
    if args.dev_file:
        dev_dataset = train_mlp_Dataset(
            data_path=args.dev_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            chat_format=args.chat_format,
            check_presence=True,
            drop_bad_samples=True,
            layer_keys=args.layer_keys,
            num_memory_slots=args.num_memory_slots,
            slot_pooling=args.slot_pooling,
            slot_pooling_temperature=args.slot_pooling_temperature,
            instruction_text=args.instruction_text,
            request_text=args.request_text,
        )

    collate_fn = partial(collator, llm_tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size,
    )
    dev_dataloader = None
    if dev_dataset is not None:
        dev_dataloader = DataLoader(
            dev_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=args.per_device_train_batch_size,
        )

    optimizer, trainable_params = build_optimizer(model, args)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_training_steps_for_scheduler = (
        args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    if dev_dataloader is not None:
        model, optimizer, train_dataloader, lr_scheduler, dev_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler, dev_dataloader
        )
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

    if semantic_model is not None:
        semantic_model = semantic_model.to(accelerator.device)
    base_model = accelerator.unwrap_model(model)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    completed_steps = 0
    rolling = {"loss": 0.0, "nll": 0.0, "semantic": 0.0, "diversity": 0.0}
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(args.num_train_epochs):
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["gag_input_ids"],
                    attention_mask=batch["gag_attention_mask"],
                    retrieval_embeds=batch["retrieval_embeds"],
                    retrieval_slot_indices=batch["retrieval_slot_indices"],
                    output_hidden_states=bool(args.alpha_semantic and args.alpha_semantic > 0.0),
                )

                loss = torch.zeros((), device=outputs.logits.device, dtype=torch.float32)
                nll_loss = get_nll_loss(
                    labels=batch["gag_input_labels"],
                    logits=outputs.logits,
                    vocab_size=vocab_size,
                )
                loss = loss + args.alpha_nll * nll_loss
                rolling["nll"] += nll_loss.detach().float().item()

                if args.alpha_semantic and args.alpha_semantic > 0.0:
                    answer_mask = (batch["gag_input_labels"] != -100).long()
                    answer_hidden = masked_mean_pool(outputs.hidden_states[-1], answer_mask)
                    predicted_semantic = base_model.project_semantic(answer_hidden)

                    with torch.no_grad():
                        tokenized_answers = semantic_tokenizer(
                            batch["answer_texts"],
                            padding=True,
                            truncation=True,
                            max_length=args.semantic_max_length,
                            return_tensors="pt",
                        ).to(accelerator.device)
                        semantic_outputs = semantic_model(**tokenized_answers)
                        target_semantic = mean_pool_encoder_outputs(
                            semantic_outputs.last_hidden_state,
                            tokenized_answers["attention_mask"],
                        ).to(dtype=predicted_semantic.dtype)

                    semantic_loss = get_latent_semantic_loss(predicted_semantic, target_semantic)
                    loss = loss + args.alpha_semantic * semantic_loss
                    rolling["semantic"] += semantic_loss.detach().float().item()

                if args.alpha_diversity and args.alpha_diversity > 0.0 and outputs.projected_retrieval_embeds is not None:
                    projected_chunks = split_projected_memory_by_sample(
                        outputs.projected_retrieval_embeds,
                        batch["memory_slots_per_sample"],
                    )
                    diversity_loss = get_memory_diversity_loss(projected_chunks)
                    loss = loss + args.alpha_diversity * diversity_loss
                    rolling["diversity"] += diversity_loss.detach().float().item()

                if not torch.isfinite(loss):
                    retrieval_tensor = batch.get("retrieval_embeds")
                    retrieval_absmax = None
                    retrieval_std = None
                    if retrieval_tensor is not None:
                        retrieval_absmax = retrieval_tensor.detach().float().abs().max()
                        retrieval_std = retrieval_tensor.detach().float().std()

                    logits_absmax = outputs.logits.detach().float().abs().max()
                    logits_finite = torch.isfinite(outputs.logits).all()
                    projected_absmax = None
                    projected_finite = None
                    if outputs.projected_retrieval_embeds is not None:
                        projected_absmax = outputs.projected_retrieval_embeds.detach().float().abs().max()
                        projected_finite = torch.isfinite(outputs.projected_retrieval_embeds).all()

                    details = {
                        "step": completed_steps + 1,
                        "sample_ids": batch.get("sample_ids", []),
                        "nll_loss": _format_log_value(nll_loss),
                        "semantic_loss": _format_log_value(
                            semantic_loss if args.alpha_semantic and args.alpha_semantic > 0.0 else 0.0
                        ),
                        "diversity_loss": _format_log_value(
                            diversity_loss if args.alpha_diversity and args.alpha_diversity > 0.0 else 0.0
                        ),
                        "retrieval_absmax": _format_log_value(retrieval_absmax),
                        "retrieval_std": _format_log_value(retrieval_std),
                        "logits_absmax": _format_log_value(logits_absmax),
                        "logits_finite": bool(logits_finite.item()),
                        "projected_absmax": _format_log_value(projected_absmax),
                        "projected_finite": None if projected_finite is None else bool(projected_finite.item()),
                    }
                    raise FloatingPointError(f"Non-finite loss encountered: {json.dumps(details)}")

                rolling["loss"] += loss.detach().float().item()
                accelerator.backward(loss)
                grad_norm = None
                if accelerator.sync_gradients and args.clip_grad_norm and args.clip_grad_norm > 0:
                    grad_norm = accelerator.clip_grad_norm_(trainable_params, args.clip_grad_norm)
                if accelerator.sync_gradients:
                    bad_grad_names = []
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            bad_grad_names.append(f"grad:{name}")
                    if bad_grad_names:
                        raise FloatingPointError(
                            "Non-finite gradients encountered: "
                            + ", ".join(bad_grad_names[:10])
                        )
                optimizer.step()
                if getattr(base_model, "layer_mixer_logits", None) is not None and args.layer_mix_logit_clamp:
                    with torch.no_grad():
                        base_model.layer_mixer_logits.clamp_(
                            min=-float(args.layer_mix_logit_clamp),
                            max=float(args.layer_mix_logit_clamp),
                        )
                bad_param_names = _collect_non_finite_param_names(
                    ((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad),
                    prefix="param",
                )
                if bad_param_names:
                    raise FloatingPointError(
                        "Non-finite trainable parameters encountered after optimizer step: "
                        + ", ".join(bad_param_names[:10])
                    )
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    scale = args.logging_steps
                    logs = {
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "train_loss": rolling["loss"] / scale,
                        "train_nll_loss": rolling["nll"] / scale,
                    }
                    if grad_norm is not None:
                        logs["grad_norm"] = _format_log_value(grad_norm)
                    if args.alpha_semantic and args.alpha_semantic > 0.0:
                        logs["train_semantic_loss"] = rolling["semantic"] / scale
                    if args.alpha_diversity and args.alpha_diversity > 0.0:
                        logs["train_diversity_loss"] = rolling["diversity"] / scale
                    logs = {key: _format_log_value(value) for key, value in logs.items()}
                    accelerator.log(logs, step=completed_steps)
                    logger.info(json.dumps({"step": completed_steps, **logs}), main_process_only=True)
                    rolling = {"loss": 0.0, "nll": 0.0, "semantic": 0.0, "diversity": 0.0}

                if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                    step_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
                    save_with_accelerate(
                        accelerator,
                        model,
                        tokenizer,
                        os.path.join(step_dir, "full_weight"),
                        save_projector_only=False,
                    )
                    save_with_accelerate(
                        accelerator,
                        model,
                        tokenizer,
                        os.path.join(step_dir, "only_projector_weight"),
                        save_projector_only=True,
                    )

                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            save_with_accelerate(
                accelerator,
                model,
                tokenizer,
                os.path.join(epoch_dir, "full_weight"),
                save_projector_only=False,
            )
            save_with_accelerate(
                accelerator,
                model,
                tokenizer,
                os.path.join(epoch_dir, "only_projector_weight"),
                save_projector_only=True,
            )

        if completed_steps >= args.max_train_steps:
            break

    accelerator.end_training()
    last_dir = os.path.join(args.output_dir, "last")
    save_with_accelerate(
        accelerator,
        model,
        tokenizer,
        os.path.join(last_dir, "full_weight"),
        save_projector_only=False,
    )
    save_with_accelerate(
        accelerator,
        model,
        tokenizer,
        os.path.join(last_dir, "only_projector_weight"),
        save_projector_only=True,
    )


if __name__ == "__main__":
    main()
