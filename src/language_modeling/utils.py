import torch
import torch.nn as nn
import os
import yaml
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM


GAG_TOKEN = "<GAG>"


def get_yaml_file(file_path):  
    import yaml
    try:  
        with open(file_path, 'r') as file:  
            return yaml.safe_load(file)  
    except FileNotFoundError:  
        print(f"YAML configuration file {file_path} not found.")  
        return {}  


def get_nll_loss(logits,labels,vocab_size):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def save_with_accelerate(accelerator, model, tokenizer, output_dir, save_projector_only=False):
    
    unwrapped_model = accelerator.unwrap_model(model)

    if save_projector_only:    
            params_to_save = {
                n:p.float() for n,p in unwrapped_model.named_parameters() 
                if any(
                    sub_string in n 
                    for sub_string in ['embed_tokens','projector','lm_head']
                    )
                }
            if accelerator.is_main_process:
                os.makedirs(output_dir)
                torch.save(params_to_save, os.path.join(output_dir,'ckpt.pth'))
                unwrapped_model.config.save_pretrained(output_dir)

    else:    
        # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
        # Otherwise, sometimes the model will be saved with only part of the parameters.
        # Also, accelerator needs to use the wrapped model to get the state_dict.
        state_dict = accelerator.get_state_dict(model)

        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict,
            safe_serialization=False, ## safetensors is buggy for now
        )

    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)

        
# def test():
#     # 生成占位串（例如 "[XRAG] [XRAG] [XRAG] ..."）
#     background_len = 5
#     gag_tokens = " ".join([GAG_TOKEN] * background_len)
#     print(gag_tokens)

# if __name__ == "__main__":
#     test()