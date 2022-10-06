# path to log file
LOGFILE="extract_flow.data"

# path to processed videos 
# e.g. on vm: /home/COMPASS-user/workspace/blobfuse/shuang/Data/ego4d/preprocessed_data/egoclip
SOURCE="/data/shared/ssvl/ego4d/v1/egoclip"

# path to new egoclip annotation file
# e.g. on vm: /home/COMPASS-user/workspace/blobfuse/shuang/Data/ego4d/data/v1/annotations/new_egoclip.csv
NEW_ANNOTATION_FILE="/home/leijiachen/ssvl/videomae/new_egoclip.csv"

# maximum number of threads on each gpu to extract flow
# e.g. if the cuda memory of gpu is 32G then 64 is recommended
#                   ...             24G then 32 is recommended
MAX_NUM_THREADS=32

# available gpu ids
# $MAX_NUM_THREADS*${#GPUS[@]} number of threads will be started
GPUS=(0 1 5 3)

# total threads
thread_num=$(( ${#GPUS[@]}*$MAX_NUM_THREADS ))


function run_docker(){

    local gpu_clip_id=$1
    local source=$2
    local logfile=$3

    
    local gpu=$(echo $gpu_clip_id | cut -d ":" -f 1)
    local clip_id=$(echo $gpu_clip_id | cut -d ":" -f 2)
    local uid=$(echo $clip_id | cut -d "_" -f 1)

    local in="$source/$uid/$clip_id/rgb"
    local out="$source/$uid/$clip_id/flow"

    # mkdir if not exist and change directory permission
    [ -d $out ] || mkdir $out
    chmod 777 $in
    chmod 777 $out

    # echo $gpu $clip_id $in $out done

    rgb_frames=$(ls $in | sort)
    echo ${rgb_frames[@]}

    # rename
    # num=0
    # for frame in ${rgb_frames[@]};do
    #     new_name=$(printf "%05d.jpg" $num)
    #     mv "$in/$frame" "$in/$new_name"
    #     num=$((num+1))
    #     # echo $new_name, $num
    # done


    # docker run --gpus "device=$gpu" \
    #         --rm \
    #         -v "$in:/input" \
    #         -v "$out:/output" \
    #         willprice/furnari-flow \
    #         %05d.jpeg -g 0 -s 1 -d 1 -b 8 > /dev/null 2>&1

    # if [[ $? -ne 0  ]];then
    #     echo "$clip_id,fail" >> $logfile
    # else
    #     echo "$clip_id,success" >> $logfile
    # fi

}

export -f run_docker

# echo "Reading $NEW_ANNOTATION_FILE ..."
# clips=()
# idx=0
# num=0
# while IFS= read -r line; do
#     fields=($(echo $line | rev | cut -d " " -f 2-4 | rev))
#     clip_id=${fields[0]}
#     clips+=("${GPUS[$idx]}:$clip_id")

#     # echo ${GPUS[$idx]}, $idx    

#     idx=$((idx+1))
#     idx=$((idx % ${#GPUS[@]}))

#     below codes are for debugging
#     num=$((num+1))
#     if [[ $num -eq 10 ]];then
#         break
#     fi

# done < "$NEW_ANNOTATION_FILE"


echo "Start extracting flows"

# echo ${clips[@]} | xargs -n 1 -P $thread_num -i bash -c "run_docker {} $SOURCE $LOGFILE"

# for debugging
echo 0:865733f5-97b6-4380-a418-1fd6510e0f5e_00001 | xargs -n 1 -i bash -c "run_docker {} $SOURCE $LOGFILE" done
