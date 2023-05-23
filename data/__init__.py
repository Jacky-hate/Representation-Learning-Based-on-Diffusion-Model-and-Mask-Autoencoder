from .data_dmim import build_loader_dmim
from .data_finetune import build_loader_finetune
from .data_sample import _build_loader_sample

def build_loader(config, logger, is_pretrain):
    if is_pretrain:
        return build_loader_dmim(config, logger)
    else:
        return build_loader_finetune(config, logger)

def build_loader_sample(config, logger, shuffle=False):
    return _build_loader_sample(config, logger, shuffle)