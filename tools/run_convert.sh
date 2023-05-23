CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node 8 \
convert_pth.py \
--cfg ./configs/vit_base/dmim_finetune__vit_base__img192__100ep.yaml \
--data-path /mnt/urchin/kzou/data/ILSVRC/Data/CLS-LOC \
--batch-size 192 \
--pretrained /mnt/urchin/kzou/code/transformer/output_pt_vit_192_ep800/dmim_pretrain/dmim_pretrain__vit_base__img192__100ep/ckpt_epoch_300.pth \
--output ./output_converted/vit_ep300 \
--accumulation-steps 1 \
--opts SAVE_FREQ 5
