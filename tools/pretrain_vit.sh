python -m torch.distributed.launch \
--nproc_per_node 8 \
main_dmim.py \
--cfg ./configs/vit_base/dmim_pretrain__vit_base__img192__100ep.yaml \
--data-path /mnt/urchin/kzou/data/ILSVRC/Data/CLS-LOC \
--batch-size 128 \
--output ./output_pt_vit_192_ep800  \
--accumulation-steps 1 \
--opts TRAIN.EPOCHS 800 SAVE_FREQ 5
