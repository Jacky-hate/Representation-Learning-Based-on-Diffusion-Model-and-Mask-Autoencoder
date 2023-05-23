import os
import time
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.utils import AverageMeter

from config import get_config
from models import build_model
from data import build_loader_sample
from logger import create_logger
from utils import load_pretrained_sample, calculate_psnr, calculate_ssim
from einops import rearrange
from torchvision.utils import make_grid

from PIL import Image



def parse_option():
    parser = argparse.ArgumentParser('MIM training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--pretrained', type=str, help='path to pre-trained model')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    
    # distributed training
    parser.add_argument("--local_rank", type=int, default='0', help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    
    data_loader_val = build_loader_sample(config, logger, shuffle=False)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=True)
    model.cuda()
    logger.info(str(model))

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")


    load_pretrained_sample(config, model_without_ddp, logger)
    sample(config, data_loader_val, model)

@torch.no_grad()
def sample(config, data_loader, model):
    model.eval()

    batch_time = AverageMeter()
    
    num_batches = len(data_loader)
    
    
    path = os.path.join(config.OUTPUT, f'val_ep100_1024')
    
    os.makedirs(path,exist_ok=True)
    
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    
    mean=torch.tensor(IMAGENET_DEFAULT_MEAN).cuda(non_blocking=True).unsqueeze(1).unsqueeze(2)
    std=torch.tensor(IMAGENET_DEFAULT_STD).cuda(non_blocking=True).unsqueeze(1).unsqueeze(2)

    end = time.time()
    for idx, (images, masks, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        masks = masks.cuda(non_blocking=True)

        # compute output
        masked_imgs, denoised_imgs, denoised_imgs_process = model.sample(images, masks)
        del denoised_imgs_process
        
        
        denoised_imgs = denoised_imgs.mul(std).add(mean).clamp(0., 1.)

        # to image
        denoised_imgs = 255. * rearrange(denoised_imgs, 'b c h w -> b h w c').cpu().numpy()
        
        for i in range(denoised_imgs.shape[0]):
            img = Image.fromarray(denoised_imgs[i].astype(np.uint8), mode="RGB")
            img.save(os.path.join(path, f'{idx*config.DATA.BATCH_SIZE+i}.png'), quality=95)
         
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.info(f'{idx}/{num_batches}')
        if idx == 0:
            break

    logger.info(f'Sampling succeed')



if __name__ == '__main__':
    _, config = parse_option()


    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)