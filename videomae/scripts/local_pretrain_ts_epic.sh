
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 \
        --master_port 12320 --nnodes=1 --node_rank=$1 --master_addr=$2 \
        ../run_mae_pretraining.py \
        --overwrite command-line \
        --config /mnt/code/ssvl/videomae/config/local/pretrain_ts_epic55.yml \
        --project pretrain_ts_epic55 \
        --name ts_preepic55_A0 \
        --local_world_size 4 \
        --debug \