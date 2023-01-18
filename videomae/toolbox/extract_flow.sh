rank=$1
# python ms_extract_flow.py \
#     --logfile processed_flow_$rank.data \
#     --anno_path /mnt/shuang/Data/ego4d/preprocessed_data/new_egoclip_$rank.csv \
#     --source /mnt/shuang/Data/ego4d/preprocessed_data/egoclip/ \
#     --gpus 0 \
#     --nprocess 1 \
#     --max_num_threads 1 \

python ms_extract_flow.py \
    --anno_path /mnt/shuang/Data/ego4d/preprocessed_data/new_egoclip_$rank.csv \
    --logfile /mnt/shuang/Data/ego4d/preprocessed_data/processed_flow_$rank.data \
    --source /mnt/shuang/Data/ego4d/preprocessed_data/egoclip \
    --nprocess 32 \
    --max_num_threads 2 \
    --gpus 0