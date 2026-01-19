import torch
import torch.nn as nn
from transformers import Qwen3ForCausalLM, Qwen3Config
from typing import Optional, Union


class XQwen3Config(Qwen3Config):

    def __init__(
        self,
        projector_type: str = "mlp2x_gelu",
        retriever_hidden_size: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.retriever_hidden_size = int(retriever_hidden_size)


class Projector(nn.Module):

    def __init__(self, config):
        super().__init__()
        projector_type = config.projector_type
        self.projector = nn.Sequential(
            nn.Linear(config.retriever_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

    def forward(self, context_embedding):
        return self.projector(context_embedding)


class XQwen3ForCausalLM(Qwen3ForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.projector = Projector(config)
        self.retriever_hidden_size = config.retriever_hidden_size
        self.post_init()

    def set_gag_token_id(self, token_id):
        self.gag_token_id = token_id

    # def prepare_inputs_embeds(
    #     self,
    #     prefix_ids,  # [B, Lp]
    #     retrieval_embeds,  # [B, Ls, retriever_hidden_size]
    #     answer_ids,  # [B, La]
    # ):

    #     prefix_embeds = self.model.embed_tokens(prefix_ids)  # [B, Lp, config.hidden_size]   # [1,14,4096]
    #     answer_embeds = self.model.embed_tokens(answer_ids)  # [B, La, config.hidden_size]    # [1,30,4096]
    #     #retrieval_embeds = retrieval_embeds.view(-1, self.retriever_hidden_size)   # [B, Ls, self.retriever_hidden_size]    # debug: [1,39,2048]

    #     retrieval_embeds = self.projector(retrieval_embeds.to(prefix_embeds.dtype))    # [B, Ls, config.hidden_size]   # [1,39,4096]

    #     inputs_embeds = torch.cat([prefix_embeds, retrieval_embeds, answer_embeds], dim = 1)   # [B, Lp+Ls+La,config.hidden_size]   # [1,83,4096]

    #     return inputs_embeds

    def prepare_inputs_embeds(
        self,
        input_ids,  # [B, Lp + Ls + La ]
        retrieval_embeds,  # [-1, retriever_hidden_size]
    ):

        input_embeds = self.model.embed_tokens(input_ids)  # [B, Lp + Ls + La, config.hidden_size]   # [1,14,4096]   # my own: [4,330,4096]
        retrieval_embeds = retrieval_embeds.view(-1, self.retriever_hidden_size)   # [-1, self.retriever_hidden_size]    
        # my own: original:[323,2048]
        # after: [323,2048]
        

        ## sanity check
        num_gag_tokens = torch.sum(input_ids==self.gag_token_id).item()  # 323
        num_retrieval_embeds = retrieval_embeds.shape[0]  # 323
        assert num_gag_tokens == num_retrieval_embeds,(num_gag_tokens,num_retrieval_embeds)

        retrieval_embeds = self.projector(retrieval_embeds.to(device=input_embeds.device, dtype=input_embeds.dtype))    
        # my own: input_embeds.device : device(type='cuda', index=0)
        # input_embeds.dtype: torch.bfloat16
        # retrieval_embeds.shape: [323,4096]
        input_embeds[input_ids==self.gag_token_id] = retrieval_embeds
        # my own: input_embeds.shape: [4,330,3096]
        # input_ids.shape: [4,330]
        # retrieval_embeds.shape: [323,4096]
        # (input_ids==self.gag_token_id).sum().item() : 323
        # 执行完之后：input_embeds.shape: [4,330,4096]



        return input_embeds    # [B, Lp+ Ls+ La, 4096]

    
    def forward(
        self,
        input_ids = None,  # [B, Lp + Ls + La]
        retrieval_embeds = None,  # [-1, retriever_hidden_size]
        attention_mask = None,
        **kwargs,
    ):

        inputs_embeds = kwargs.pop("inputs_embeds", None)  # my own eval: [4,330,4096]
        at_the_begining_of_generation = False
        if inputs_embeds is not None:
            assert not self.training
            assert retrieval_embeds is None
            at_the_begining_of_generation = True
        
        if not at_the_begining_of_generation:

            if retrieval_embeds is not None:
                inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds)    # [B, Lp+Ls+La, config.hidden_size]    
                input_ids = None
                assert inputs_embeds.shape[1] == attention_mask.shape[1], (inputs_embeds.shape, attention_mask.shape)

        #     return super().forward(
        #         inputs_embeds = inputs_embeds,
        #         attention_mask = attention_mask,
        #         **kwargs,
        #     )
        
        # else:
        #     return super().forward(
        #         inputs_embeds = inputs_embeds,   # my own eval: [4,330,4096]
        #         attention_mask = attention_mask,   # my own eval: [4,330]
        #         **kwargs,   # my own eval: kwargs.keys(): dict_keys(['cache_position', 'past_key_values', 'use_cache', 'return_dict'])
        #     )
        return super().forward(
            input_ids = input_ids,
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask,
            **kwargs,
        )


    @torch.no_grad()
    def generate(
        self,
        input_ids = None,
        retrieval_embeds = None,
        **kwargs,
    ):
        attention_mask = kwargs.pop("attention_mask", None)   # my own: [4,330]
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported for generate")
        
        inputs_embeds=None
        if retrieval_embeds is not None:   # my own: [323,2048]
            inputs_embeds = self.prepare_inputs_embeds(input_ids,retrieval_embeds)   # my own: input_ids.shape: [4,330]   retrieval_embeds.shape: [323,2048]   input_embeds.shape: [4,330,4096]
            input_ids = None
            if attention_mask is not None:  # [4,330]
                assert inputs_embeds.shape[1] == attention_mask.shape[1],(inputs_embeds.shape,attention_mask.shape)
            return super().generate(
                attention_mask=attention_mask,  # [4,330]
                inputs_embeds=inputs_embeds,  # [4,330,4096]
                **kwargs      # 原版xRAG是这样的：{'stopping_criteria': [<src.eval.utils.MultiTokenEOSCriteria object at 0x7f6ed58e36d0>, <src.eval.utils.MultiTokenEOSCriteria object at 0x7f6ed58e3670>, <src.eval.utils.MultiTokenEOSCriteria object at 0x7f6ed58e3640>], 'do_sample': False, 'max_new_tokens': 100, 'pad_token_id': 32000, 'use_cache': True}
                # my own: kwargs: {'do_sample': False, 'max_new_tokens': 100, 'pad_token_id': 151643, 'use_cache': True}
            )
        
        else:
            return super().generate(
                attention_mask=attention_mask,
                input_ids=input_ids,
                **kwargs
            )
    