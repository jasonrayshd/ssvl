python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 51225 --nnodes=4  --node_rank=$1 --master_addr=$2 \
    ../run_class_finetuning.py \
    --enable_deepspeed \
    --dist_eval \
    --overwrite command-line \
    --config /data/shared/ssvl/videomae/config/temp/finetune_basic_ego4d.yml \
    --project pt-epic55-ft-ego4dsc \
    --name preepic55ftego4d_A5 \
    # --wandb_id 3ccex8ya \
    # --project temp \
    # --name temp \
    # --debug \