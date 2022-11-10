rank=$1
python ms_extract_flow.py \
    --logfile /mnt/shuang/Data/ego4d/preprocessed_data/processed_flow_$rank.data \
    --anno_path /mnt/shuang/Data/ego4d/preprocessed_data/new_egoclip_$rank.csv \
    --source /mnt/shuang/Data/ego4d/preprocessed_data/egoclip \
    --gpus 0 1 2 3 \
    --nprocess 48 \
    --max_num_threads 4 \
