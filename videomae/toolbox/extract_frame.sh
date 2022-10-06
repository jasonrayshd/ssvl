# path to egoclip annotation file 
# e.g. on vm: /home/COMPASS-user/workspace/blobfuse/shuang/Data/ego4d/data/v1/annotations/egoclip.csv
ANNOTATION_FILE=""

# path to log file. videos that have been successfully processed or errors will be record in the log file
# recommended value: processed_video.data
LOGFILE=""

# path to source videos
# e.g. on vm: /home/COMPASS-user/workspace/blobfuse/shuang/Data/ego4d/data/v1/full_scale
SOURCE=""

# path to save processed videos (frames)
# e.g. on vm: /home/COMPASS-user/workspace/blobfuse/shuang/Data/ego4d/preprocessed_data/egoclip
DEST=""

# number of processes to use
NPROCESS=2

# maximum number of threads that each process is allowed to own
MAX_NUM_THREADS=32

# shorter side size of saved frame
# e.g. the shape of original frame is 1920x1080
# after resizing, the shape of frame will become 455x256
DESIRED_SHORTER_SIDE=256


python process_egoclip.py \
    --logfile $LOGFILE \
    --anno_path $ANNOTATION_FILE \
    --source $SOURCE \
    --dest $DEST \
    --nprocess $NPROCESS \
    --max_num_threads $MAX_NUM_THREADS \
    --desired_shorter_side $DESIRED_SHORTER_SIDE