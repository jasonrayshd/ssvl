
# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port 51225 --nnodes=1  --node_rank=$1 --master_addr=$2 \
    run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set Ego4d-statechange-classification-localization \
    --nb_classes -1 \
    --finetune GoogleDrive://k400_videomae_pretrain_base_patch16_224_tubemasking_ratio_0.9_e800 \
    --log_dir /mnt/shuang/Output/output_ego4d \
    --output_dir /mnt/shuang/Output/output_ego4d \
    --batch_size 2 \
    --num_sample 1 \
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
    --data_path /mnt/ego4d/v1/
    # --data_path /mnt/shuang/Data/ego4d/data/v1/
    # --test_num_segment 5 \
    # --test_num_crop 3 \
    # --data_path ${DATA_PATH} \