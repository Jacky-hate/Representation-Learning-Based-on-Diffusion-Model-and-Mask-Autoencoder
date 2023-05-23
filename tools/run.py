import os
for ep in range(0,81,10):
    cmd=" \
CUDA_VISIBLE_DEVICES=7 python sample.py \
--cfg ./configs/vit_base_ldm/dmim_pretrain__vit_base_ldm__img192__100ep.yaml \
--data-path /mnt/urchin/kzou/data/ILSVRC/Data/CLS-LOC \
--batch-size 12 \
--pretrained /mnt/urchin/kzou/code/dmim/output_pt_ldm_192_ep800_wo_amp_all_loss/dmim_pretrain/dmim_pretrain__vit_base__img192__100ep/ckpt_epoch_{}.pth \
--output /mnt/urchin/kzou/code/dmim/output_sample \
--opts MODEL.DIFFUSION.TIMESTEPS 1000".format(ep)
    os.system(cmd)