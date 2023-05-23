CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nproc_per_node 8 \
main_dmim.py \
--cfg ./configs/vit_base_ldm/dmim_pretrain__vit_base_ldm__img192__100ep.yaml \
--data-path /mnt/urchin/kzou/data/ILSVRC/Data/CLS-LOC \
--batch-size 43 \
--output ./output_pt_ldm_constrastive/output_pt_ldm_contrasive_loss_neg \
--accumulation-steps 3 \
--contrastive \
--opts TRAIN.EPOCHS 800 SAVE_FREQ 5
