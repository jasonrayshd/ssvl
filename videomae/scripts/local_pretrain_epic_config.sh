
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 \
        --master_port 12320 --nnodes=1 --node_rank=$1 --master_addr=$2 \
        ../run_mae_pretraining.py \
        --overwrite command-line \
        --config /mnt/code/videomae/config/local/pretrain_flow_epic55.yml \
        --project temp \
        --name preepic55_A0
        # --debug \