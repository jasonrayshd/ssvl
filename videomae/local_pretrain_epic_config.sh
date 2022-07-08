
CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 \
        --master_port 12320 --nnodes=1 --node_rank=$1 --master_addr=$2 \
        run_mae_pretraining.py \
        --overwrite command-line \
        --config /mnt/code/videomae/config/pretrain_flow_epic.yml \
        # --debug \