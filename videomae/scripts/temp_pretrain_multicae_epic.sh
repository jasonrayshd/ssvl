OMP_NUM_THREADS=100 python -m torch.distributed.launch --nproc_per_node=2 \
        --master_port 12320 --nnodes=2 --node_rank=$1 --master_addr=$2 \
        ../run_mae_pretraining.py \
        --overwrite command-line \
        --config $3 \
        --project pretrain_multimodal_epic55 \
        --name multicae_preepic55_A0 \
        # --debug
        # --resume /data/shared/output/multimodal_preepic55_A0/checkpoint-39.pth