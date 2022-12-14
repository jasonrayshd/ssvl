OMP_NUM_THREADS=40 python -m torch.distributed.launch --nproc_per_node=2 \
        --master_port 12320 --nnodes=2 --node_rank=$1 --master_addr=$2 \
        ../run_mae_pretraining.py \
        --overwrite command-line \
        --config /data/shared/ssvl/videomae/config/temp/tuning_ts2.yml \
        --project tuning_ts_epic55 \
        --name ts_tuning_A2 \
        # --debug \
