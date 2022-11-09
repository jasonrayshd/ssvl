# # Set the path to save checkpoints
# OUTPUT_DIR='/mnt/code/ssvl/videomae/output/k400'
# # Set the path to Kinetics train set. 
# DATA_PATH='/data/shared/ssvl/kinetics/kinetics-dataset/k400/annotations/resized320/train.csv'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 \
        --master_port 12320 --nnodes=1 --node_rank=$1 --master_addr=$2 \
        ../run_mae_pretraining.py \
        --name pre_k400_A0 \
        --config /mnt/code/ssvl/videomae/config/local/pretrain_rgb_k400.yml \
        --project pretrain_k400 \
        --debug \
        --seed 0 \