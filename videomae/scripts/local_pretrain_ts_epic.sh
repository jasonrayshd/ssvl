OMP_NUM_THREADS=40 python -m torch.distributed.launch --nproc_per_node=1 \
        --master_port 12320 --nnodes=1 --node_rank=$1 --master_addr=$2 \
        ../run_mae_pretraining.py \
        --overwrite command-line \
        --config ../config/local/pretrain_ts_epic55.yml \
        --project pretrain_ts_epic55 \
        --name temp \
        --debug