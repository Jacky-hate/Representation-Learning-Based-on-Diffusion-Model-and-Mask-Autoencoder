CUDA_VISIBLE_DEVICES=4 python sample_one.py \
--cfg ./configs/vit_base/dmim_pretrain__vit_base__img192__100ep.yaml \
--data-path /mnt/urchin/kzou/data/ILSVRC/Data/CLS-LOC \
--batch-size 1 \
--pretrained /mnt/urchin/kzou/code/transformer/output_vit/output_pt_vit_192_ep800/dmim_pretrain/dmim_pretrain__vit_base__img192__100ep/ckpt_epoch_100.pth \
--output /mnt/urchin/kzou/code/transformer/output_sample \
