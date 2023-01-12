rank=$1
python ms_extract_flow.py \
    --logfile extract_flow_$rank.data \
    --anno_path /data/shared/ssvl/ego4d/v1/annotations/egoclip.csv \
    --source /data/shared/random/temp/ \
    --gpus 0 \
    --nprocess 1 \
    --max_num_threads 1 \