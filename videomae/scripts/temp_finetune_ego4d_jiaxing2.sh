OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 51225 --nnodes=4  --node_rank=$1 --master_addr=$2 \
    ../run_class_finetuning.py \
    --enable_deepspeed \
    --dist_eval \
    --overwrite command-line \
    --config /data/shared/ssvl/videomae/config/temp/2finetune_ts_ego4d.yml \
    --project pt-k400-ft-ego4dsc \
    --name prek400ftego4d_A6 \
    # --debug \
    # --project pt-epic55-ft-ego4dsc \
    # --name preepic55ftego4d_A0 \
    # --wandb_id 3ccex8ya \