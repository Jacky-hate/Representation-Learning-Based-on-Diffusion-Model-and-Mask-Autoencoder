from .vision_transformer import build_vit
from .dmim import build_dmim


def build_model(config, is_pretrain=True):
    if is_pretrain:
        model = build_dmim(config)
    else:
        model_type = config.MODEL.TYPE
        if model_type == 'vit':
            model = build_vit(config)
        else:
            raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model
