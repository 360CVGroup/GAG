# sitecustomize.py
def _patch_transformers_generation():
    import transformers.generation.utils as gen_utils
    from transformers.generation.utils import GenerationMixin

    # 已经打过补丁就不重复
    if hasattr(GenerationMixin, "generate_to_get_answer_embedding_different_layers"):
        return

    # 你的实现放在你仓库里（下面这个模块你自己创建）
    from gag_patches.transformers_generation_patch import (
        generate_to_get_answer_embedding_different_layers,
        _sample_to_get_answer_embedding_different_layers,
    )

    GenerationMixin.generate_to_get_answer_embedding_different_layers = generate_to_get_answer_embedding_different_layers
    GenerationMixin._sample_to_get_answer_embedding_different_layers = _sample_to_get_answer_embedding_different_layers

_patch_transformers_generation()
