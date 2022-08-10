OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port 51225 --nnodes=1  --node_rank=$1 --master_addr=$2 \
    ../test_on_ego4d.py \
    --dist_eval \
    --overwrite command-line \
    --name temp \
    --config /mnt/code/ssvl/videomae/config/local/test_basic_ego4d.yml \
    --ckpt /mnt/code/ssvl/videomae/output/preepic55ftego4d_A0/checkpoint-99/mp_rank_00_model_states.pt