rank=$1

python ms_clip2zip.py \
        --logfile "/mnt/shuang/Data/ego4d/preprocessed_data/epic55_rank$rank.data" \
        --nprocess 8 \
        --max_num_threads 8 \