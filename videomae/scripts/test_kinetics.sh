# Set the path to save checkpoints
OUTPUT_DIR='YOUR_PATH/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/eval_lr_5e-4_repeated_aug_epoch_75'
# path to Kinetics set (train.csv/val.csv/test.csv)
DATA_PATH='YOUR_PATH/list_kinetics-400'
# path to pretrain model
MODEL_PATH='YOUR_PATH/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/checkpoint-1599.pth'

python run_class_finetuning.py \
    --model vit_base_patch16_224 \
    --data_set Kinetics-400 \
    --nb_classes 400 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 6 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --num_frames 16 \
    --sampling_rate 4 \
    --eval \
    --test_num_segment 5 \
    --test_num_crop 3 \