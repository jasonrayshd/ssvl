"""
split processed video to clip

"""

import io
import os
import shutil
import csv
import argparse
from tqdm import tqdm
import threading
import multiprocessing as mlp

import pandas as pd
from datetime import timedelta
import time

import signal

import tarfile
import zipfile

import wandb
wandb.init(project="preprocess_egoclip")


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def segment_name(self):
        return NotImplementedError()

    @property
    def participant(self):
        return NotImplementedError()

    @property
    def untrimmed_video_name(self):
        return NotImplementedError()

    @property
    def start_frame(self):
        return NotImplementedError()

    @property
    def end_frame(self):
        return NotImplementedError()

    @property
    def num_frames(self):
        return NotImplementedError()

    @property
    def label(self):
        return NotImplementedError()


def timestamp_to_sec(timestamp):
    x = time.strptime(timestamp, '%H:%M:%S.%f')
    sec = float(timedelta(hours=x.tm_hour,
                          minutes=x.tm_min,
                          seconds=x.tm_sec).total_seconds()) + float(
        timestamp.split('.')[-1]) / 100
    return sec



class EpicKitchensVideoRecord(VideoRecord):
    def __init__(self, tup):
        self._index = str(tup[0])
        self._series = tup[1]
        # print(self._series.keys())
    @property
    def participant(self):
        return self._series['participant_id']

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    # has problem when reading P01_102_47 from csv annotation file
    @property
    def start_frame(self):
        return self._series['start_frame']
    # has problem when reading P01_102_47
    @property
    def end_frame(self):
        return self._series['stop_frame']

    @property
    def fps(self):
        is_100 = len(self.untrimmed_video_name.split('_')[1]) == 3
        return 50 if is_100 else 60

    @property
    def num_frames(self):
        return self.end_frame - self.start_frame

    @property
    def label(self):
        return {'verb': self._series['verb_class'] if 'verb_class' in self._series else -1,
                'noun': self._series['noun_class'] if 'noun_class' in self._series else -1}

    @property
    def metadata(self):
        return {'narration_id': self._index}


class DelayedKeyboardInterrupt:

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)
                
    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


def parse_terminal_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--logfile", type=str, default="clip2zip.data", help="Path to log file")

    parser.add_argument("--nprocess", type=int, default=1, help="Total number of processes used")
    parser.add_argument("--max_num_threads", type=int, default=1, help="Number of threads used by each process")

    return parser.parse_args()


def read_epic_csv(anno_path):
    """
        process epic-kitchens55 csv annotation file

        Parameters:
            anno_path: str, path of epic-kitchens55 csv annotation file

        Return:
            data_dict: a dict[list] object whose key is video uid and value is the list of frame index list of each **clip** (not video) 
            e.g.
            "uid": [[frame0, frame1, ...], [frame100, frame101, ...], ...]

    """

    data_dict = {}
    for tup in pd.read_pickle(anno_path).iterrows():
        epic_video = EpicKitchensVideoRecord(tup)

        clip_uid = epic_video._index
        person = epic_video.participant
        video = epic_video.untrimmed_video_name
        st_f = epic_video.start_frame
        end_f = epic_video.end_frame

        if st_f == 0:
            print(f"0 frame detected!: {video}_{clip_uid}")

        name = f"{video}_{clip_uid}"

        if video not in data_dict.keys():
            data_dict[video] = []

        data_dict[video].append(
            {
                "name": name,
                "clip_uid": clip_uid,
                "person": person,
                "video": video,
                "st_f": st_f,
                "end_f": end_f,
            } 
        )

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
        chunked_data_dict.append([ [*data_dict[key] ] for key in data_keys[chunk_size_lst[i]:chunk_size_lst[i+1]] ])

    return chunked_data_dict


def thread_worker(path, clip_packs, tmp_dir, queue):

    video = clip_packs[0]["video"]
    person = clip_packs[0]["person"]

    # print( len(clip_packs) )

    # read video tar file from remote server
    rgb_frame_path = os.path.join(path, "rgb", "train", person, video)
    flow_frame_path = os.path.join(path, "flow", "train", person, video)

    exist_rgb_tar_files = [file.split(".")[0] for file in os.listdir(rgb_frame_path) if file.endswith(".tar")]
    exist_flow_tar_files = [file.split(".")[0] for file in os.listdir(flow_frame_path) if file.endswith(".tar")]
    # exist_rgb_zip_files = [file.split(".")[0] for file in os.listdir(rgb_frame_path) if file.endswith(".zip")]
    # exist_flow_zip_files = [file.split(".")[0] for file in os.listdir(flow_frame_path) if file.endswith(".zip")]

    os.makedirs(os.path.join(tmp_dir, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(tmp_dir, "flow"), exist_ok=True)

    # cache to local
    shutil.copy(rgb_frame_path+".tar", tmp_dir+"/rgb")
    shutil.copy(flow_frame_path+".tar", tmp_dir+"/flow")

    rgb_video_tf = tarfile.open(tmp_dir+"/rgb/"+ video+".tar", "r")
    flow_video_tf = tarfile.open(tmp_dir+"/flow/"+ video+".tar", "r")

    frame_idx_list = []
    frame_idx_to_clip = {} # dict[k:list]
    clip_to_frame_num = {}
    to_transfer_clip = [] # clip that is not transferred
    for pack in clip_packs:

        name = pack["name"]

        st_f = pack["st_f"]
        end_f = pack["end_f"]

        for i in range(st_f, end_f+1):
            if i not in frame_idx_to_clip.keys():
                frame_idx_to_clip[i] = []

            frame_idx_to_clip[i].append(name)

        # collect all frames in the video that are used for training
        frame_idx_list.extend(list(range(st_f, end_f+1)))
        # collect total frame number of each clip
        clip_to_frame_num[name] = end_f - st_f + 1
        to_transfer_clip.append(name)

    # deduplicate and sort the frame index list
    frame_idx_list = sorted(list(set(frame_idx_list)))

    for frame_idx in frame_idx_list:
        rgb_idx = frame_idx
        frame_name = "frame_{:010d}.jpg".format(rgb_idx)
        frame_bytes = rgb_video_tf.extractfile("./" + frame_name).read()

        clip_list = frame_idx_to_clip[frame_idx]

        for clip_name in clip_list:
            # targeted clip tar file
            current_frame_num = 0
            with zipfile.ZipFile(os.path.join(tmp_dir, "rgb", clip_name+".zip"), "a") as zf_handle:
                zf_handle.writestr(frame_name, frame_bytes)
                current_frame_num = len(zf_handle.namelist())

            if rgb_idx % 2 == 1:

                flow_idx = (rgb_idx // 2 + 1)
                flow_name = "frame_{:010d}.jpg".format(flow_idx)

                uflow_bytes = flow_video_tf.extractfile("./u/"+flow_name).read()
                vflow_bytes = flow_video_tf.extractfile("./v/"+flow_name).read()

                with zipfile.ZipFile(os.path.join(tmp_dir, "flow", clip_name+".zip"), "a") as zf_handle:

                    zf_handle.writestr("u/"+flow_name, uflow_bytes)
                    zf_handle.writestr("v/"+flow_name, vflow_bytes)

            if clip_to_frame_num[clip_name] == current_frame_num:
                # Finished processsing all frames for this clip whose name is $clip_name
                rgb_dest = os.path.join(path, "rgb", "train", person, video, clip_name)
                flow_dest = os.path.join(path, "flow", "train", person, video, clip_name)

                if clip_name in exist_rgb_tar_files:
                    os.system(f"rm {rgb_dest}.tar")
                if clip_name in exist_flow_tar_files:
                    os.system(f"rm {flow_dest}.tar")

                shutil.move(os.path.join(tmp_dir, "rgb", clip_name+".zip"), rgb_dest+".zip")
                shutil.move(os.path.join(tmp_dir, "flow", clip_name+".zip"), flow_dest+".zip")

                to_transfer_clip.remove(clip_name)

    for clip_name in to_transfer_clip:
        rgb_dest = os.path.join(path, "rgb", "train", person, video, clip_name)
        flow_dest = os.path.join(path, "flow", "train", person, video, clip_name)

        if clip_name in exist_rgb_tar_files:
            os.system(f"rm {rgb_dest}.tar")
        if clip_name in exist_flow_tar_files:
            os.system(f"rm {flow_dest}.tar")
        shutil.move(os.path.join(tmp_dir, "rgb", clip_name+".zip"), rgb_dest+".zip")
        shutil.move(os.path.join(tmp_dir, "flow", clip_name+".zip"), flow_dest+".zip")

    os.system(f"rm -rf {tmp_dir}")

    queue.put({
        "video_name": video,
        "thread_name": threading.current_thread().name,
        "state": "done",
    })


def process_worker(data_list, source, max_num_threads, queue):

    def comm_with_main_proces(pack, queue):
        pack.update({
            "process_end": False,
            "process_name": mlp.current_process().name,
        })
        queue.put(pack)
 
    active_thread_num = 0
    thread_pool = {}
    process_name = mlp.current_process().name
    thread_queue = mlp.Queue(2*max_num_threads)
    i = 0
    # print("data_list", len(data_list))

    while i < len(data_list):
        # print("i", i)
        if active_thread_num < max_num_threads:
            thread_name = f"Thread-{i}"
            tmp_dir = "./" + process_name + "/"+thread_name
            os.makedirs(tmp_dir, exist_ok=True)

            thread = threading.Thread(target=thread_worker, args=(source, data_list[i], tmp_dir, thread_queue), name=thread_name)
            thread.start()

            thread_pool[thread_name] = thread
            active_thread_num += 1
            i += 1

        else:
            pack = thread_queue.get()
            comm_with_main_proces(pack, queue)
            thread_name = pack["thread_name"]
            thread_pool[thread_name].join()
            active_thread_num -= 1

    # print("waiting for all processes to end...")
    # for k, thread in thread_pool.items():
    #     thread.join()

    done_thread_num = 0
    print("emptying thread queue...")
    while done_thread_num < active_thread_num or not thread_queue.empty():
        pack = thread_queue.get()
        comm_with_main_proces(pack, queue)

        done_thread_num += 1
        thread_name = pack["thread_name"]
        thread_pool[thread_name].join()

    # inform main process that this process ended
    queue.put({
        "process_end": True,
        "process_name": mlp.current_process().name,

        "video_name": "",
        "state": "",
    })

    return 0


def write2log(pack, logger):
    video_name = pack["video_name"]
    state = pack["state"]

    logger.write(f"{video_name},{state}\n")


def main(args):
    root = "/mnt/shuang/Data/epic-kitchen/3h91syskeag572hl6tvuovwv4d"

    # root = "/data/shared/ssvl/epic-kitchens55/3h91syskeag572hl6tvuovwv4d"
    # dest = "./"

    logfile = args.logfile
    open(logfile, "a+").close()

    anno_path = os.path.join(root, "annotations/EPIC_train_action_labels.pkl")
    print("Processing epic-kitchens55 annotation file..")
    data_dict = read_epic_csv(anno_path)

    source = os.path.join(root, "frames_rgb_flow")

    # do not process videos that have already been transformed into clips
    exclude_videos = []
    with open(logfile, "r") as log_fp:
        for line in log_fp.readlines():
            line = line.strip("\n")
            if line == "":
                continue
            meta = line.split(",")
            exclude_videos.append(meta[0])

    # deduplicate
    exclude_videos = list(set(exclude_videos))

    filtered_data_dict = {}
    filtered_num = 0
    total_clip_num = 0
    print("Skipping processed/excluded videos...")

    # exclude by video
    for video_uid in exclude_videos:
        data_dict.pop(video_uid)
        filtered_num += 1

    filtered_data_dict = data_dict

    for k,v in filtered_data_dict.items():
        total_clip_num += len(v)

    print(f"Clips to be processed:{total_clip_num}")
    print(f"{filtered_num}/[{len(exclude_videos)}] videos have been ignored.")
    # return

    nprocess = args.nprocess
    max_num_threads = args.max_num_threads
    chunked_data_dict = split_data(filtered_data_dict, nprocess) # list[ list[str, dict, dict, ...], list[], ... ]

    process_pool = {}
    queue = mlp.Queue(nprocess*max_num_threads*2)
    for i in range(nprocess):
        process_name = f"Process-{i}"
        process = mlp.Process(target=process_worker, args=(chunked_data_dict[i], source, max_num_threads, queue), name=process_name)
        process.start()
        process_pool[process_name] = process

    iter = 0
    done_process = 0
    progress_bar = tqdm(total=len(filtered_data_dict))

    logger = open(logfile, "a+")
    logger.write("\n")

    tqdm.write(f"recording logs to file: {logfile}")
    while iter < len(filtered_data_dict) and done_process < nprocess :

        pack = queue.get()

        if not pack["process_end"]:
            # if process not end

            write2log(pack, logger)
            iter += 1

            progress_bar.update(1)
            progress_bar.set_postfix_str(f"current process number:{nprocess-done_process}")
            wandb.log({"video": iter})
        else:
            print(f"Process: {pack['process_name']} finished")
            done_process += 1

        # tqdm.set_postfixstr

    print("emptying main process queue...")
    while not queue.empty():
        pack = queue.get()
        if pack["process_end"]:
            continue
        write2log(pack, logger)
    print("done")
    

if __name__ == "__main__":

    args = parse_terminal_args()

    main(args)

