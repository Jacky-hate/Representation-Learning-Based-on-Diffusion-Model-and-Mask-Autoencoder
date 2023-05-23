CUDA_VISIBLE_DEVICES=6 python sample_uncond.py \
--cfg ./configs/vit_base_ldm/dmim_pretrain__vit_base_ldm__img192__100ep.yaml \
--data-path /mnt/urchin/kzou/data/ILSVRC/Data/CLS-LOC \
--batch-size 10 \
--pretrained /mnt/urchin/kzou/code/transformer/output_pt_ldm_x0/output_pt_ldm_192_ep800_x0_bs1024/dmim_pretrain/dmim_pretrain__vit_base__img192__100ep/ckpt_epoch_500.pth \
--output /mnt/urchin/kzou/code/transformer/output_sample \
--no-sp-checkpoint \
--opts MODEL.DIFFUSION.TIMESTEPS 1000 MODEL.DIFFUSION.LOG_EVERY_T 1000
