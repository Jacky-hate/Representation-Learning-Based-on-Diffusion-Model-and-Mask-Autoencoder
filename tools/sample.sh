CUDA_VISIBLE_DEVICES=4 python sample.py \
--cfg ./configs/vit_base_ldm/dmim_pretrain__vit_base_ldm__img192__100ep.yaml \
--data-path /mnt/urchin/kzou/data/ILSVRC/Data/CLS-LOC \
--batch-size 1024 \
--pretrained /mnt/urchin/kzou/code/transformer/output_pt_ldm_x0/output_pt_ldm_192_ep800_x0_bs1024/dmim_pretrain/dmim_pretrain__vit_base__img192__100ep/ckpt_epoch_100.pth \
--output /mnt/urchin/kzou/code/transformer/output_sample \
--opts MODEL.DIFFUSION.TIMESTEPS 1000 MODEL.DIFFUSION.LOG_EVERY_T 1000
