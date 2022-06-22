#!/bin/bash

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nnodes=$1 --nproc_per_node=$2 --node_rank=$3 --master_addr=$4 --master_port $5 \
    run_class_finetuning.py \
    --dist_on_itp \
    --model vit_base_patch16_224 \
    --data_set Ego4d-statechange-classification-localization \
    --nb_classes -1 \
    --finetune GoogleDrive://k400_videomae_pretrain_base_patch16_224_tubemasking_ratio_0.9_e800 \
    --log_dir /mnt/shuang/Output/output_ego4d \
    --output_dir /mnt/shuang/Output/output_ego4d \
    --data_path /mnt/shuang/Data/ego4d/data/v1/full_scale \
    --anno_path /mnt/shuang/Data/ego4d/data/v1/annotations \
    --pos_clip_save_path /mnt/shuang/Data/ego4d/preprocessed_data/pos \
    --neg_clip_save_path /mnt/shuang/Data/ego4d/preprocessed_data/neg \
    --name runvprelim \
    --batch_size 8 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --save_ckpt \
    --clip_len 8 \
    --sampling_rate 2 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 75 \
    --enable_deepspeed \
    --dist_eval \
    # --test_num_segment 5 \
    # --test_num_crop 3 \