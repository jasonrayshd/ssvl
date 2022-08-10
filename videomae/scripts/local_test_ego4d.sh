OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port 51225 --nnodes=1  --node_rank=$1 --master_addr=$2 \
    ../test_on_ego4d.py \
    --overwrite command-line \
    --name preepic55ftego4d_A2 \
    --config /data/shared/ssvl/videomae/config/temp/test_basic_ego4d.yml \
    --ckpt /data/shared/output/preepic55ftego4d_A2/mp_rank_00_model_states.pt
    # --dist_eval \