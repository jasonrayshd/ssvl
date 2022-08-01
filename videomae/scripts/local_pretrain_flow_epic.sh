
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 \
        --master_port 12320 --nnodes=1 --node_rank=$1 --master_addr=$2 \
        ../run_mae_pretraining.py \
        --overwrite command-line \
        --config /mnt/code/ssvl/videomae/config/local/pretrain_flow_epic55.yml \
        --project pretrain_single_precision_epic55 \
        --name pre_single_precision_epic55_A2 \
        --seed 413 \
        --train_wo_amp \
        --debug \
        # --wandb_id 32gziijm # resume training