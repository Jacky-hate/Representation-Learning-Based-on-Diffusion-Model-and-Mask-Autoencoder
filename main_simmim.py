import warnings
warnings.simplefilter('ignore')
import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.cuda.amp as amp
from timm.utils import AverageMeter

from config import get_config
from models import build_model
from data import build_loader, build_loader_sample
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, calculate_psnr, calculate_ssim
from contextlib import nullcontext
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange
from PIL import Image
from torchvision.utils import make_grid


def parse_option():
    parser = argparse.ArgumentParser('MIM pre-training script', add_help=False)
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
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--contrastive', action='store_true')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    writer = SummaryWriter(config.OUTPUT)
    data_loader_train = build_loader(config, logger, is_pretrain=True)
    if dist.get_rank() == 0:
        data_loader_val = build_loader_sample(config, logger)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, is_pretrain=True)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model, logger, is_pretrain=True)
    
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[config.LOCAL_RANK], 
        broadcast_buffers=False, 
        # find_unused_parameters=True
        )
    model_without_ddp = model.module


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    scaler = amp.GradScaler()

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT, logger)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, scaler, logger)

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler, scaler, writer)
        if dist.get_rank() == 0:
            if config.MODEL.TYPE == 'vit_ldm':
                psnr, ssim = evaluate(config, model_without_ddp, data_loader_val, epoch, writer)
                writer.add_scalar('sample_psnr', psnr, global_step=epoch)
                writer.add_scalar('sample_ssim', ssim, global_step=epoch)
                psnr, ssim = evaluate(config, model_without_ddp, data_loader_val, epoch, writer, False)
                writer.add_scalar('denoise_psnr', psnr, global_step=epoch)
                writer.add_scalar('denoise_ssim', ssim, global_step=epoch)
            if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_without_ddp, 0., optimizer, lr_scheduler, scaler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    
    
@torch.no_grad()
def evaluate(config, model, data_loader, epoch, writer, is_sample = True):
    
    path = os.path.join(config.OUTPUT, f'figs_epoch_{epoch}' if is_sample else f'figs_denoised_epoch_{epoch}')
    
    os.makedirs(path,exist_ok=True)
    
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    
    mean=torch.tensor(IMAGENET_DEFAULT_MEAN).cuda(non_blocking=True).unsqueeze(1).unsqueeze(2)
    std=torch.tensor(IMAGENET_DEFAULT_STD).cuda(non_blocking=True).unsqueeze(1).unsqueeze(2)
    
    for idx, (img, mask, _) in enumerate(data_loader):
        imgs = img.cuda(non_blocking=True)
        masks = mask.cuda(non_blocking=True)
        
        masked_imgs, denoised_imgs, denoised_imgs_process = model.sample(imgs, masks) if is_sample else model.predict_denoised(imgs, masks)
        torch.cuda.synchronize()
        
        denoised_imgs_process = torch.tensor(np.array([x.cpu().numpy() for x in denoised_imgs_process])).permute(1, 0, 2, 3, 4).cuda(non_blocking=True)
        
        psnrs=0.0
        ssims=0.0
        
        for index,(masked_img, image,denoised_process,denoised_img) in enumerate(zip(masked_imgs,imgs,denoised_imgs_process,denoised_imgs)):

            all_figs = [image, masked_img]+[denoised_process[index] for index in range(0,len(denoised_process),len(denoised_process)//20)]+[denoised_img]
            
            psnr = calculate_psnr(255. * rearrange(image, 'c h w -> h w c').cpu().numpy(),255. * rearrange(denoised_img, 'c h w -> h w c').cpu().numpy())
            ssim = calculate_ssim(255. * rearrange(image, 'c h w -> h w c').cpu().numpy(),255. * rearrange(denoised_img, 'c h w -> h w c').cpu().numpy())

            psnrs = psnrs + psnr
            ssims = ssims + ssim
            
            all_figs = [x.mul_(std).add_(mean).clamp_(0., 1.) for x in all_figs]
            # display as grid
            grid = torch.stack(all_figs, 0)
            #grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=5)

            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            
            
            id = index
            save_path = os.path.join(path, f'img_{id}_ssim_{round(ssim, 5)}_psnr_{round(psnr, 5)}.png')
            img_to_save = Image.fromarray(grid.astype(np.uint8), mode="RGB")
            img_to_save.save(save_path, quality=95)  # max 95
        
        return psnrs/min(config.DATA.BATCH_SIZE,12), ssims/min(config.DATA.BATCH_SIZE,12)


def save_img(config, name, imgs):
    
    path = os.path.join(config.OUTPUT)
    
    os.makedirs(path,exist_ok=True)
    
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    
    mean=torch.tensor(IMAGENET_DEFAULT_MEAN).cuda(non_blocking=True).unsqueeze(1).unsqueeze(2)
    std=torch.tensor(IMAGENET_DEFAULT_STD).cuda(non_blocking=True).unsqueeze(1).unsqueeze(2)
    
    all_figs = [x[0].mul_(std).add_(mean).clamp_(0., 1.) for x in imgs]
    
    grid = torch.stack(all_figs, 0)
    #grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=1)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()

    save_path = os.path.join(path, f'{name}.png')
    img_to_save = Image.fromarray(grid.astype(np.uint8), mode="RGB")
    img_to_save.save(save_path, quality=95)  # max 95

def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler, scaler, writer):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    loss_scale_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (img, mask, augmented_pair, _) in enumerate(data_loader):
        img = img.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        augmented_pair = [aug.cuda(non_blocking=True) for aug in augmented_pair]
        
        
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            
            mcontext = model.no_sync if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS != 0 else nullcontext
            
            with mcontext():
                with amp.autocast(enabled=False):
                    loss = model(img, mask, augmented_pair)
            loss = loss / config.TRAIN.ACCUMULATION_STEPS
            scaler.scale(loss).backward()
            grad_norm = get_grad_norm(model.parameters())
                
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                if config.TRAIN.CLIP_GRAD:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step_update(epoch * num_steps + idx)
                optimizer.zero_grad()
        else:
            with amp.autocast(enabled=True):
                loss = model(img, mask, augmented_pair)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if config.TRAIN.CLIP_GRAD:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), img.size(0))
        norm_meter.update(grad_norm)
        loss_scale_meter.update(scaler.get_scale())
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {loss_scale_meter.val:.4f} ({loss_scale_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            writer.add_scalar('loss', loss_meter.avg, global_step=epoch*num_steps+idx)
            writer.add_scalar('lr', lr, global_step=epoch*num_steps+idx)
            writer.add_scalar('grad_norm', norm_meter.val, global_step=epoch*num_steps+idx)
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    _, config = parse_option()


    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # # linear scale the learning rate according to total batch size, may not be optimal
    # linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    # if config.TRAIN.ACCUMULATION_STEPS > 1:
    #     linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
    #     linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
    #     linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    # config.defrost()
    # config.TRAIN.BASE_LR = linear_scaled_lr
    # config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    # config.TRAIN.MIN_LR = linear_scaled_min_lr
    # config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)
