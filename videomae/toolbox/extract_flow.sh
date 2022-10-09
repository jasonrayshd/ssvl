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
MAX_NUM_THREADS=2

# available gpu ids
# $MAX_NUM_THREADS*${#GPUS[@]} number of threads will be started
GPUS=(3 4)

# total threads
thread_num=$(( ${#GPUS[@]}*$MAX_NUM_THREADS ))
echo number of threads: $thread_num

function run_docker(){

    local gpu_clip_id=$1
    local source=$2
    local logfile=$3

    local gpu=$(echo $gpu_clip_id | cut -d ":" -f 1)
    local clip_id=$(echo $gpu_clip_id | cut -d ":" -f 2)
    local uid=$(echo $clip_id | cut -d "_" -f 1)

    local in="$source/$uid/$clip_id/rgb"
    local zjuin="$source/$uid/$clip_id"
    local out="$source/$uid/$clip_id/flow"

    # for local server
    # mkdir if $in not exist and change directory permission
    [ -d $in ] || mkdir -p $in
    local localf=($(ls $zjuin | grep jpeg))
    if [[ ${#localf[@]} -ne 0 ]];then
        for f in ${localf[@]};do
            mv "$zjuin/$f" "$in"
        done
    fi

    # mkdir if $out not exist and change directory permission
    [ -d $out ] || mkdir $out

    chmod 777 $in
    chmod 777 $out

    # rename image file to %05d.jpg and start from 00000.jpg
    rgb_frames=$(ls $in | sort)
    # echo ${rgb_frames[@]}

    if [[ ${#rgb_frames[@]} -eq 0 ]];then
        echo "$clip_id,fail" >> $logfile
        return 1
    fi

    num=0
    for frame in ${rgb_frames[@]};do
        new_name=$(printf "%05d.jpg" $num)
        if [[ "$in/$frame" == "$in/$new_name" ]];then
            continue
        fi
        mv "$in/$frame" "$in/$new_name"
        num=$((num+1))
        # echo $new_name, $num
    done

    docker run --gpus "device=$gpu" \
            --rm \
            -v "$in:/input" \
            -v "$out:/output" \
            willprice/furnari-flow \
            %05d.jpg -g 0 -s 1 -d 1 -b 8 > /dev/null 2>&1

    if [[ $? -ne 0  ]];then
        echo "$clip_id,fail" >> $logfile
    else
        echo "$clip_id,success" >> $logfile
    fi

}

export -f run_docker

# determine clips to be processed
echo "Preparing information of clips to be processed..."
py_ret=($(python filter_processed_flow.py --logfile $LOGFILE --anno_path $NEW_ANNOTATION_FILE))

processed_num="${py_ret[0]}"
clips_raw=("${py_ret[@]:1}")
echo "Done. $processed_num clips have been processed. A total of ${#clips_raw[@]} clips will be processed"

# determine gpu id for each clip
clips=()
idx=0
num=0
for clip in ${clips_raw[@]};do

    # echo ${GPUS[$idx]}, $idx    
    clips+=("${GPUS[$idx]}:$clip")
    idx=$((idx+1))
    idx=$((idx % ${#GPUS[@]}))

    # below codes are for debugging
    # num=$((num+1))
    # if [[ $num -eq 10 ]];then
    #     break
    # fi
done

echo "Start extracting flows... Please check $LOGFILE for current progress"

printf "%s\n" "${clips[@]}" | xargs -n 1 -P $thread_num -i bash -c "run_docker {} $SOURCE $LOGFILE;echo \r processing..." done
