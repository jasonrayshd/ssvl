OMP_NUM_THREADS=100 python -m torch.distributed.launch --nproc_per_node=1 \
        --master_port 12320 --nnodes=1 --node_rank=$1 --master_addr=$2 \
        ../run_mae_pretraining.py \
        --overwrite command-line \
        --config $3 \
        --project pretrain_multimodal_epic55 \
        --name temp \
        --seed 1234 \
        --debug