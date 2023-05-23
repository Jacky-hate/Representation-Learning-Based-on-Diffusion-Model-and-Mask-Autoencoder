CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node 8 \
main_finetune.py \
--cfg ./configs/vit_base/dmim_finetune__vit_base__img192__100ep.yaml \
--data-path /mnt/urchin/kzou/data/ILSVRC/Data/CLS-LOC \
--batch-size 192 \
--pretrained /mnt/urchin/kzou/code/transformer/output_pt_ldm_constrastive/output_pt_ldm_contrasive_loss_noised/dmim_pretrain/dmim_pretrain__vit_base__img192__100ep/ckpt_epoch_50.pth \
--output ./output_ft_ldm_contrastive/output_ft_ldm_ep50_noised \
--accumulation-steps 1 \
--opts SAVE_FREQ 5
