OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 51225 --nnodes=1  --node_rank=$1 --master_addr=$2 \
    ../test_on_ego4d.py \
    --overwrite command-line \
    --name prek400ftego4d_A7 \
    --config /code/videomae/config/temp/test_basic_ego4d.yml \
    --ckpt /data/jiachen/ssvl_output/prek400ftego4d_A8/checkpoint-best_state_cls/mp_rank_00_model_states.pt \
    --dist_eval \
