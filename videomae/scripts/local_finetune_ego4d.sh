
OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 51225 --nnodes=1  --node_rank=$1 --master_addr=$2 \
    ../run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set Ego4d-statechange-classification-localization \
    --nb_classes -1 \
    --log_dir /mnt/output \
    --output_dir /mnt/output \
    --data_path /mnt/ego4d/v1/clips \
    --anno_path /mnt/ego4d/v1/partial_anno \
    --pos_clip_save_path /mnt/pos \
    --neg_clip_save_path /mnt/neg \
    --batch_size 2 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --clip_len 8 \
    --sampling_rate 2 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 75 \
    --enable_deepspeed \
    --dist_eval \
    --debug \
    --config /mnt/code/videomae/config/finetune_basic_ego4d.yml