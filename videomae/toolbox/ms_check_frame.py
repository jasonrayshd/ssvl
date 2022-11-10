import os
import csv
import zipfile
import argparse
from tqdm im    port tqdm
import wandb
wandb.init(project="preprocess_egoclip")

import threading
import multiprocessing as mlp

def parse_terminal_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--logfile", type=str, default="check_processed_video.data", help="Path to log file")
    parser.add_argument("--anno_path", type=str, default="/data/shared/ssvl/ego4d/v1/annotations/egoclip.csv", help="Path to egoclip annotation file")

    parser.add_argument("--source", type=str, default="/data/shared/ssvl/ego4d/v1/egoclip/", help="Path to source videos")

    parser.add_argument("--nprocess", type=int, default=1, help="Total number of processes used")
    parser.add_argument("--max_num_threads", type=int, default=1, help="Number of threads used by each process")

    return parser.parse_args()


def read_egoclip_csv(anno_path):
    """
        process egoclip csv annotation file

        Parameters:
            anno_path: str, path of egoclip csv annotation file

        Return:
            data_dict: dict, whose key is video uid and value is a list of list which contains frames for each clip

    """
    reader = csv.reader(open(anno_path, "r", encoding='utf-8'))
    next(reader)
    rows = list(reader)
    progress_bar = tqdm(total = len(rows))
    data_dict = {}
    video2index = {}
    frame_num = 0
    for row in rows:

        meta = row[0].split("\t")

        if meta[0] not in data_dict:
            data_dict[meta[0]] = []

        start_f = max(0, int( float(meta[5]) * 30) )
        end_f = max(0, int( float(meta[6]) * 30) )

        frame_num += end_f - start_f
        if meta[0] not in video2index:
            video2index[meta[0]] = 0
        else:
            video2index[meta[0]] += 1

        data_dict[meta[0]].append({
            "video_uid":meta[0],
            "video_dur": meta[1],
            "narration_source": meta[2],
            "narration_ind":meta[3],
            "narration_time": meta[4],
            "clip_start": meta[5],
            "clip_end": meta[6],
            "arration_info": "\t".join(meta[7:]),
        
            "frames": list(range(start_f, end_f+1)),
            "clip_idx": video2index[meta[0]],
            "clip_name": meta[0] + "_{:05d}".format( video2index[meta[0]] ),
            "start_frame": start_f,
            "end_frame": end_f,
        })

        # data_dict[meta[0]].append(list(range(start_f, end_f+1))) # [[], [], [], ...]

        progress_bar.update(1)

    return data_dict


def split_data(data_dict, chunk_num=10):

    """
        split data dictionary for each process

        Parameters:
            data_dict: dict, whose key is video uid and value is a list of all frame indexes of the video 
            chunk_num: int, number that data_dict will be separated into

        Return:

            chunked_data_dict: list, a list that contains {chunk} numbers of data_dict

    """

    chunked_data_dict = []
    idx_lst = [i for i in range(len(data_dict.keys()))]
    chunk_size = len(idx_lst) // chunk_num

    chunk_size_lst =  [i * chunk_size for i in range(chunk_num)]

    chunk_size_lst.append(len(idx_lst))
    print(f"Chunksize: {chunk_size_lst}")

    data_keys = list(data_dict.keys())
    for i in range(chunk_num):
        chunked_data_dict.append([[ key, *data_dict[key] ] for key in data_keys[chunk_size_lst[i]:chunk_size_lst[i+1]] ])

    return chunked_data_dict


def thread_worker(video_meta, source, thread_queue):
    # print(video_meta)
    uid, clip_metas = video_meta[0], video_meta[1:]

    for c_meta in clip_metas:
        uid = c_meta["video_uid"]
        clip_name = c_meta["clip_name"]
        st_f, end_f = c_meta["start_frame"], c_meta["end_frame"]

        if not os.path.exists(os.path.join(source, uid, clip_name, "frames.zip")):
            thread_queue.put({
                "thread_name": threading.current_thread().name,
                "clip_name": clip_name,
                "thread_end": False,
                "state": "cannot find frames.zip",
            })

            continue

        missed_frame = []
        with zipfile.ZipFile(os.path.join(source, uid, clip_name, "frames.zip")) as zf:
            try:
                frame_idx_lst = [name.split(".")[0].split("_")[-1] for name in zf.namelist()]
            except:
                thread_queue.put({
                    "thread_name": threading.current_thread().name,
                    "thread_end": False,
                    "clip_name": clip_name,
                    "state": "frames.zip has corrupted",
                })
                continue

            for i in range(st_f, end_f):
                frame_idx = "{:010d}".format(i)
                if frame_idx not in frame_idx_lst:
                    missed_frame.append(str(frame_idx))

        thread_queue.put({
                "thread_name": threading.current_thread().name,
                "thread_end": False,
                "clip_name": clip_name,
                "state": f"missing frames: {' '.join(missed_frame)}" if len(missed_frame) != 0 else "success",
        })

    thread_queue.put({
        "thread_name": threading.current_thread().name,
        "thread_end": True,
        "clip_name": "",
        "state": "",
    })
    return 0


def process_worker(data_list, source, max_num_threads, queue):

    def comm_with_main_proces(pack, queue):
        pack.update({
            "process_end": False,
            "process_name": mlp.current_process().name,
        })
        queue.put(pack)

    active_thread_num = 0
    thread_pool = {}

    thread_queue = mlp.Queue(2*max_num_threads)
    i = 0
    while i < len(data_list):

        if active_thread_num < max_num_threads:
            thread_name = f"Thread_{i}"
            thread = threading.Thread(target=thread_worker, args=(data_list[i], source, thread_queue), name=thread_name)
            thread.start()

            thread_pool[thread_name] = thread
            active_thread_num += 1
            i += 1

        else:
            pack = thread_queue.get()

            thread_name = pack["thread_name"]
            if pack["thread_end"]:
                thread_pool[thread_name].join()
                active_thread_num -= 1
                continue

            comm_with_main_proces(pack, queue)

    done_thread_num = 0
    print("emptying thread queue...")
    while not thread_queue.empty() or done_thread_num != active_thread_num:
        pack = thread_queue.get()
        thread_name = pack["thread_name"]
        if pack["thread_end"]:
            thread_pool[thread_name].join()
            done_thread_num += 1
            continue

        comm_with_main_proces(pack, queue)

    # inform main process that this process ended
    queue.put({
        "process_end": True,
        "process_name": mlp.current_process().name,

        "clip_name": "",
        "state": "",
    })

    return 0


def write2log(pack, logger):
    clip_name = pack["clip_name"]
    state = pack["state"]

    logger.write(f"{clip_name},{state}\n")


def main(args):

    tqdm.write("reading egoclip annotation file")
    data_dict = read_egoclip_csv(args.anno_path)

    open(args.logfile, "a+").close()


    checked_clips = []
    with open(args.logfile, "r") as logger:
        for line in logger.readlines():
            clip_uid = line.split(",")[0]
            checked_clips.append(clip_uid)

    tqdm.write("filtering data dict")
    total_video_num = 0
    total_clip_num = 0
    skip_clip_num = 0
    for video_uid, clip_metas in tqdm(data_dict.items()):
        new_clip_metas = []
        for c_meta in clip_metas:
            if c_meta["clip_name"] in checked_clips:
                skip_clip_num += 1
                continue
            new_clip_metas.append(c_meta)

        if len(new_clip_metas) != 0:
            total_video_num += 1
        total_clip_num += len(new_clip_metas)
        data_dict[video_uid] = new_clip_metas

    tqdm.write(f"skipped {skip_clip_num} clips")
    source = args.source
    nprocess = args.nprocess
    max_num_threads = args.max_num_threads

    chunked_data_dict = split_data(data_dict, nprocess) # list[ list[str, dict, dict, ...], list[], ... ]

    process_pool = {}
    queue = mlp.Queue(nprocess*max_num_threads*2)
    for i in range(nprocess):
        process_name = f"Process-{i}"
        process = mlp.Process(target=process_worker, args=(chunked_data_dict[i], source, max_num_threads, queue), name=process_name)
        process.start()
        process_pool[process_name] = process

    iter = 0
    done_process = 0
    progress_bar = tqdm(total=total_clip_num)
    logger = open(args.logfile, "a+")
    logger.write("\n")

    tqdm.write(f"recording logs to file: {args.logfile}")
    while iter < total_clip_num:

        pack = queue.get()

        if not pack["process_end"]:
            # if process not end

            write2log(pack, logger)
            iter += 1

            progress_bar.update(1)
            progress_bar.set_postfix_str(f"current process number:{nprocess-done_process}")
            wandb.log({"video": iter})
        else:
            done_process += 1

        # tqdm.set_postfixstr


if __name__ == "__main__":

    args = parse_terminal_args()

    main(args)