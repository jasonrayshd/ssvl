
CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port 51225 --nnodes=2  --node_rank=$1 --master_addr=$2 \
    ../run_class_finetuning.py \
    --enable_deepspeed \
    --dist_eval \
    --overwrite command-line \
    --config /mnt/code/videomae/config/finetune_basic_ego4d.yml \
    --debug \