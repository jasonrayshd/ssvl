OMP_NUM_THREADS=100 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 51225 --nnodes=1  --node_rank=$1 --master_addr=$2 \
    ../run_class_finetuning.py \
    --enable_deepspeed \
    --dist_eval \
    --config $3 \
    --overwrite command-line \
    --project pt-epic55-ft-ego4dsc \
    --debug \
    --name temp
    # --name preepic55ftego4d_multicae_A0_wregressor_lrdecay \
    # --wandb_id 1xv1mgg5 \
    # --config /data/shared/ssvl/videomae/config/temp/finetune_ego4d0.yml \
    # --project pt-epic55-ft-ego4dsc \
    # --name preepic55ftego4d_A0 \
    # --wandb_id 3ccex8ya \