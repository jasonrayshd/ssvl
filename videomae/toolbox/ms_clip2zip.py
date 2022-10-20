"""
split processed video to clip

"""


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

    @property
    def participant(self):
        return self._series['participant_id']

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    # has problem when reading P01_102_47 from csv annotation file
    @property
    def start_frame(self):
        return int(round(timestamp_to_sec(self._series['start_timestamp']) * self.fps))
    # has problem when reading P01_102_47
    @property
    def end_frame(self):
        return int(round(timestamp_to_sec(self._series['stop_timestamp']) * self.fps))

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

        name = f"{video}_{clip_uid}"

        data_dict[name] = {
            "name": name,
            "clip_uid": clip_uid,
            "person": person,
            "video": video,
            "st_f": st_f,
            "end_f": end_f,
        }

    return data_dict


def worker(path, pack, queue,):

    name = pack["name"]
    person = pack["person"]
    video = pack["video"]

    st_f = pack["st_f"]
    end_f = pack["end_f"]

    rgb_frame_path = os.path.join(path, "rgb", "train", person, video)
    flow_frame_path = os.path.join(path, "flow", "train", person, video)

    flow_tf = tarfile.open(os.path.join(flow_frame_path, name+".tar"), "w" )
    rgb_tf = tarfile.open(os.path.join(rgb_frame_path, name+".tar"), "w")

    for idx in range(st_f, end_f+1):
        rgb_idx = (idx + 1)
        flow_idx = (idx // 2 + 1)

        try:
            rgb_tf.add(os.path.join(rgb_frame_path, "frame_{:010d}.jpg".format(rgb_idx)),"frame_{:010d}.jpg".format(rgb_idx))

            file_name_list = flow_tf.getnames()
            if (not "u/frame_{:010d}.jpg".format(flow_idx) in file_name_list ) and ( not "u/frame_{:010d}.jpg".format(flow_idx) in file_name_list):  
                flow_tf.add(os.path.join(flow_frame_path, "u", "frame_{:010d}.jpg".format(flow_idx)), "u/frame_{:010d}.jpg".format(flow_idx))
                flow_tf.add(os.path.join(flow_frame_path, "v", "frame_{:010d}.jpg".format(flow_idx)), "v/frame_{:010d}.jpg".format(flow_idx))

        except Exception as e:
            queue.put({
                "name": name,
                "thread_name": threading.current_thread().name,
                "state": "fail",
                "error": str(e),
            })

    queue.put({
            "name": name,
            "thread_name": threading.current_thread().name,
            "state": "success",
        })


def process_pack(pack, logger):

    uid = pack["name"]

    if pack["state"] == "success":
        print(f"{uid},done\n")
        logger.write(f"{uid},done\n")
        return 1
    else:
        print(f"{uid},{pack['error']}\n")
        logger.write(f"{uid},{pack['error']}\n")
        return 0
    

def main(args):
    root = "/mnt/shuang/Data/epic-kitchen/3h91syskeag572hl6tvuovwv4d"
    dest = "/mnt/shuang/Data/ego4d/preprocessed_data/"

    logfile = os.path.join(dest, "clip2zip.data")
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

    videos = []
    for k, v in data_dict.items():
        if not k in exclude_videos:
           videos.append(v)

    print(f"{len(exclude_videos)} videos will be ignored")

    max_thread_num = args.max_thread_num
    queue = mlp.Queue(maxsize=max_thread_num*2)

    thread_pool = {}
    logger = open(logfile, "a+")

    # single process[debug]
    # worker(f"/data/shared/ssvl/ego4d/v1/egoclip/{processed_videos[0]}", queue, data_dict, processed_video_info)
    # pack = queue.get()
    # clip_books = process_pack(pack, logger)
    # write2csv(clip_books)

    # multi-threading
    pbar = tqdm(videos)
    i = 0
    active_thread_num = 0
    video_done = 0
    try:
        while i < len(videos):

            with DelayedKeyboardInterrupt():
                if active_thread_num < max_thread_num:
                    pack = videos[i]
                    td_name = f"Thread:{i}"
                    thread = threading.Thread(
                        target=worker, 
                        args =(source, pack, queue),
                        name = td_name
                        )

                    thread.start()                    
                    thread_pool[td_name] = thread

                    # logger.write(f"processing {ack["name"].split('/')[-1]p}\n")

                    i += 1
                    active_thread_num += 1
                    
                else:
                    tqdm.write("waiting for pack to arrive")
                    pack = queue.get()
                    ret = process_pack(pack, logger)
                    video_done += ret

                    active_thread_num -= 1
                    td_name = pack["thread_name"]
                    thread_pool[td_name].join()
                    wandb.log({"video": video_done})
                    pbar.update(1)

    except KeyboardInterrupt:
        print("Waiting for all threads to finish...")
        for k, td in thread_pool.items():
            td.join()

        print("Emptying queue")
        while not queue.empty():
            pack = queue.get()
            ret = process_pack(pack, logger)
            video_done += ret
            pbar.update(1)

    else:
        print("waiting for threads to finish")
        for k, td in thread_pool.items():
            td.join()

        print("emptying queue")
        while not queue.empty():
            pack = queue.get()
            ret = process_pack(pack, logger)
            video_done += ret
            pbar.update(1)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--exclude", nargs="*", default=[], help="list of excluded videos")
    parser.add_argument("--max_thread_num", type=int, default=1, help="maximum number of threads")

    args = parser.parse_args()

    main(args)

