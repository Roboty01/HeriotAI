"""Install the team's RMSNorm/RoPE and the explicitly excluded starter SwiGLU."""


def apply():
    from transformers.models.qwen2 import modeling_qwen2 as model
    from .kernels import install
    from .swiglu import patch_qwen2_mlp
    install()
    if not getattr(model.Qwen2MLP.forward, '_soup_starter_swiglu', False):
        patch_qwen2_mlp()
        model.Qwen2MLP.forward._soup_starter_swiglu = True
