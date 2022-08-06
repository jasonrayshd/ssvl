
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 \
        --master_port 12320 --nnodes=2 --node_rank=$1 --master_addr=$2 \
        ../run_mae_pretraining.py \
        --overwrite command-line \
        --config /data/shared/ssvl/videomae/config/local/pretrain_ts_epic55.yml \
        --project pretrain_ts_epic55 \
        --name ts_preepic55_A3 \

        # --project temp \
        # --name temp \
        # --debug \
   