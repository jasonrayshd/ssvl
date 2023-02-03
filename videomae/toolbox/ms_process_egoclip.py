# extract frames from ego4d videos given egoclip annotation csv file
# total videos: 7534
# total frames: 106992281 -> ~107M

import os
import io
import av
import csv
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import shutil
import zipfile
import threading
import multiprocessing as mlp
from multiprocessing import Queue

from ego4d_trim import _get_frames
import wandb
wandb.init(project="preprocess_egoclip")

def parse_terminal_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--logfile", type=str, default="processed_video.data", help="Path to log file")
    parser.add_argument("--anno_path", type=str, default="/data/shared/ssvl/ego4d/v1/annotations/egoclip.csv", help="Path to egoclip annotation file")
    parser.add_argument("--source", type=str, default="/data/shared/ssvl/ego4d/v1/full_scale/", help="Path to source videos")
    parser.add_argument("--dest", type=str, default="/data/shared/ssvl/ego4d/v1/egoclip/", help="Path to destination")

    parser.add_argument("--nprocess", type=int, default=2, help="Total number of processes used")
    parser.add_argument("--max_num_threads", type=int, default=32, help="Number of threads used by each process")
    parser.add_argument("--desired_shorter_side", type=int, default=256, help="shorter side size of final frame")

    return parser.parse_args()


def save_frame(dest, frame, idx, frame_idx, related_clips, tmp_dir, desired_shorter_side):
    """
        save given frame

        Parameters:
            dest: str, absolute path of directory to save frames
            frame: numpy.ndarray, frame to be saved, in BGR format
            idx: index of the frame in the frame list
            frame_idx: int, index of the frame in the video
            clip_list: list of list, each sub-list contains frames for each clip 
            tmp_dir: temporal directory to save files locally
            desired_shorter_side: int, shorter side size of saved frame
    
    """

    original_height, original_width, _ = frame.shape

    if original_height < original_width:
        # Height is the shorter side
        new_height = desired_shorter_side
        new_width = np.round(
            original_width*(desired_shorter_side/original_height)
        ).astype(np.int32)
    elif original_height > original_width:
        # Width is the shorter side
        new_width = desired_shorter_side
        new_height = np.round(
            original_height*(desired_shorter_side/original_width)
        ).astype(np.int32)
    else:
        # Both are the same
        new_height = desired_shorter_side
        new_width = desired_shorter_side
    assert np.isclose(
        new_width/new_height,
        original_width/original_height,
        0.01
    )

    frame = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # write to video directory    
    frame_name_in_video_dir = "frame_{:010d}.jpg".format(idx)
    file_path =  os.path.join(dest, frame_name_in_video_dir)

    if not os.path.exists(file_path):
        cv2.imwrite(
            file_path,
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
        )

    frame_name_in_clip_dir =  "frame_{:010d}".format(idx) + "_" + "{:010d}".format(frame_idx)  + '.jpg'
    # write to zip file for the clip, the zip file will be transferred when video has been processed
    for clip in related_clips: # list[int]
        with zipfile.ZipFile(f"{tmp_dir}/{clip}.zip", "a") as zf:
            io_buf = io.BytesIO()
            pil = Image.fromarray(frame)
            pil.save(io_buf, format="jpeg", quality=100, subsampling=0)
            zf.writestr(frame_name_in_clip_dir, io_buf.getvalue())

    return 1


def read_egoclip_csv(anno_path):
    """
        process egoclip csv annotation file

        Parameters:
            anno_path: str, path of egoclip csv annotation file

        Return:
            data_dict: dict, whose key is video uid and value is a list of list which contains frames for each clip

    """
    reader = csv.reader(open(anno_path, "r", encoding='utf_8'))
    next(reader)
    rows = list(reader)
    progress_bar = tqdm(total = len(rows))
    data_dict = {}
    frame_num = 0
    for row in rows:

        meta = row[0].split("\t")

        if meta[0] not in data_dict:
            data_dict[meta[0]] = []

        start_f = max(0, int( float(meta[5]) * 30) )
        end_f = max(0, int( float(meta[6]) * 30) )

        frame_num += end_f - start_f
        data_dict[meta[0]].append(list(range(start_f, end_f+1))) # [[], [], [], ...]

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


def filter_data(data_dict, filter_list, keep_num=-1):
    """
        filter data_dict given a video list: only keep video that is in the given video list

        Parameters:
            data_dict: dict, whose key is video uid and value is a list of all frame indexes of the video 
            filter_list: list[str], contains a list of video uid
            keep_num: int, default=-1, for debugging, only return $keep_num number of data_dict
                if equals to -1, then return all qualified data dict

        Return:
            filtered_data_dict: dict, a subset of data_dict
    """

    filtered_data_dict = {}
    for k,v in data_dict.items():
        if k in filter_list:
            filtered_data_dict[k] = v

        if len(filtered_data_dict) == keep_num:
            break

    return filtered_data_dict


def td_worker(source, dest, data_dict, tmp_dir, td_queue, desired_shorter_side):

    thread = threading.current_thread()
    thread_name = thread.name
    # create temporal directory locally
    os.makedirs(tmp_dir, exist_ok=True)

    uid, clip_list = data_dict[0], data_dict[1:]

    # make video directory on destination
    frame_save_path = os.path.join(dest, uid)
    os.makedirs(frame_save_path, exist_ok=True)

    frame_to_clip = {}  
    frames_list = []
    for clip_idx, clip in enumerate(clip_list):
        # make directory for each clip
        os.makedirs(os.path.join(dest, uid + "_{:05d}".format(clip_idx)), exist_ok=True)

        if os.path.exists(os.path.join(dest, uid + "_{:05d}".format(clip_idx), "frames.zip" )):
            try:
                # verify zip file is not corrupted and is not empty
                zf = zipfile.ZipFile(os.path.join(dest, uid + "_{:05d}".format(clip_idx), "frames.zip" ), "r")
                assert len(zf.namelist()) != 0
                continue
            except:
                # else keep extracting frames for this clip
                pass

        for frame_idx in clip:
            if frame_idx not in frame_to_clip.keys():
                frame_to_clip[frame_idx] = []
            frame_to_clip[frame_idx].append(clip_idx) # mark the clip index of each frame

        frames_list.extend(clip)

    frames_list = sorted(list(set(frames_list)))

    # exist_frames = [frame_name.split(".")[0] for frame_name in os.listdir(frame_save_path)]

    video_path = os.path.join(source, uid+".mp4")
    # print(video_path)

    try:
        container = av.open(video_path)
    except Exception as e:
        # corrupted video
        td_queue.put({"error": str(e), "processed_frame_num": 0, "uid":uid, "thread_name":thread_name})
        return

    iterable_frame_source = _get_frames(
            frames_list,
            container,
            include_audio=False,
            audio_buffer_frames=0
    )

    frame_num = 0 # global index of frame
    for frame in iterable_frame_source:
        frame = frame.to_rgb().to_ndarray()
        frame_idx = frames_list[frame_num] # index of frame in the video
        save_frame(frame_save_path, frame, frame_num, frame_idx, frame_to_clip[frame_idx], tmp_dir, desired_shorter_side)
        frame_num += 1

    # copy all zip files onto remote directory
    for clip_idx in range(len(clip_list)):

        clip_name = str(clip_idx) + ".zip"
        if not os.path.exists(os.path.join(dest, uid + "_{:05d}".format(clip_idx), "frames.zip" )):
            # only transfer zip file that do not exist in dest
            shutil.move(tmp_dir +"/" + clip_name, os.path.join(dest, uid + "_{:05d}".format(clip_idx), "frames.zip" ))

    # Last few frame indexes in frames_list might exceed video duration
    missed_frame = []
    if frame_num < len(frames_list):
        missed_frame = [str(idx) for idx in frames_list[frame_num:]]

    td_queue.put({"processed_frame_num": frame_num, "uid":uid, "missed_frame":missed_frame, "thread_name":thread_name})
    return


def worker(lst_dict, source, dest, queue, max_num_threads, desired_shorter_side):

    # communication function that communicates with main process
    def comm_with_main_process(ret, queue):
        processed_frame_num = ret["processed_frame_num"]
        uid = ret["uid"]

        if processed_frame_num == 0:
            # error occurred
            error = ret["error"]
            queue.put({"video_uid":uid, "pid": os.getpid(), "state":"on", "error": str(error)})
        else:
            missed_frame = ret["missed_frame"]
            queue.put({"video_uid":uid, "pid": os.getpid(), "state":"on", "error":0, "frame_processed":processed_frame_num, "missed_frame": missed_frame})

    i = 0
    thread_pool = {}
    active_thread_num = 0
    td_queue = Queue(maxsize=2*max_num_threads)
    pid = os.getpid()
    next_avail_thread_name = ""
    while i < len(lst_dict):

        data_dict = lst_dict[i]
        if active_thread_num < max_num_threads:
            
            given_thread_name = f"{pid}-Thread:{i}" if next_avail_thread_name == "" else next_avail_thread_name
            tmp_dir = f"./{pid}/{given_thread_name}" # temporal directory to store files locally
            os.makedirs(tmp_dir, exist_ok=True)

            thread = threading.Thread( target=td_worker, args=(source, dest, data_dict, tmp_dir, td_queue, desired_shorter_side), name=given_thread_name)
            thread.start()

            thread_pool[thread.name] = thread
            active_thread_num += 1
            i += 1
        else:
            # lock until thread return some result
            # However, the thread might still be alive
            ret = td_queue.get()
            comm_with_main_process(ret, queue)
            thread_name = ret["thread_name"]
            next_avail_thread_name = thread_name

            # call .join() to wait until some thread finished
            thread_pool[thread_name].join()
            active_thread_num -= 1


    # wait for remaining threads to finish
    print("Waiting for all threads to end")
    for k, td in thread_pool.items():
        td.join()

    # obtain all remaining information
    print("Emptying thread queue")
    while not td_queue.empty():
        # get ret from queue
        ret = td_queue.get()
        comm_with_main_process(ret, queue) 

    queue.put({"video_uid":None, "pid": os.getpid(), "state":"off", "error":0})


def main(args):


    anno_path = args.anno_path
    source = args.source
    dest =  args.dest
    desired_shorter_side  = args.desired_shorter_side

    nprocess = args.nprocess
    max_num_threads = args.max_num_threads


    print("Reading egoclip annotation file..")
    data_dict = read_egoclip_csv(anno_path)

    # obtain processed video list
    processed_video_list = []
    open(args.logfile, "a+").close()
    with open(args.logfile, "r") as fp:
        for line in fp.readlines():
            if line == "\n":
                continue
            meta = line.split(",")
            if meta[1] == "success":
               processed_video_list.append(meta[0])

    processed_video_list = list(set(processed_video_list))

    # obtain exist videos that are not processed yet
    # filter_list = [video.split(".")[0] for video in os.listdir(source) if video.split(".")[0] not in processed_video_list]

    filter_list = [
        "ec344610-74f4-4765-9c3f-0837ef78055d",
        "a4c1f23e-702e-49a1-9600-0538e4e2e7db",
        "4efd8c65-82c6-481b-8cf2-9038b0b7bc3d",
        "170fb6c9-8550-4e82-bb9a-4ad574c0f0fa",
    ]
    # 4efd8c65-82c6-481b-8cf2-9038b0b7bc3d
    # a4c1f23e-702e-49a1-9600-0538e4e2e7db
    # 170fb6c9-8550-4e82-bb9a-4ad574c0f0fa
    # ec344610-74f4-4765-9c3f-0837ef78055d

    # filter data_dict according to filter_list
    data_dict = filter_data(data_dict, filter_list)
    
    # for debugging
    # data_dict = {
    #        "e14466f5-e646-4af2-a53f-1527cdf82cf9": data_dict["e14466f5-e646-4af2-a53f-1527cdf82cf9"],
    #        "e70f4b34-432f-47dc-834f-5bc6bd67dc62": data_dict["e70f4b34-432f-47dc-834f-5bc6bd67dc62"],
    #        }

    # collect total number of frames
    total_frame_num = 0
    for k,v in data_dict.items():
        _temp = []
        for clip in v:
            # clip: list
            _temp.extend(clip)
        total_frame_num += len(list(set(_temp)))

    print("Splitting Data for Multi-Processing..")
    chunked_data_dict = split_data(data_dict, chunk_num=nprocess)

    queue = Queue(maxsize=2*max_num_threads*nprocess)
    process_pool = []
    # print(chunked_data_dict[0][0])
    for i in range(nprocess):
        process = mlp.Process(target=worker, args=(chunked_data_dict[i], source, dest, queue, max_num_threads, desired_shorter_side))
        process.start()
        process_pool.append(process)


    done_process = 0
    failed_video_num = 0
    done_video_frame = 0
    done_video = 0
    pbar = tqdm(total=len(data_dict))

    with open(args.logfile, "a+") as logger:
        logger.write("\n")

        while True:
            pack = queue.get()

            if pack["state"] == "on":

                if pack["error"]:
                    # error occurred while reading/processing the video
                    failed_video_num += 1

                    logger.write(f"{pack['video_uid']},error:{pack['error']}\n")
                    # pbar.set_postfix_str(f"current processes:{nprocess-done_process}, processed frames:{done_video_frame}/[{total_frame_num}], failed video: {failed_video_num}")
                    # pbar.update(1)
                else:
                    frame_processed = pack["frame_processed"]
                    missed_frame = pack["missed_frame"]
                    done_video_frame += frame_processed
                    done_video += 1
                    logger.write(f"{pack['video_uid']},success,frames:{frame_processed},missed:{','.join(missed_frame)}\n")

                pbar.set_postfix_str(f"current processes:{nprocess-done_process}, processed frames:{done_video_frame}/[{total_frame_num}], failed video: {failed_video_num}")
                pbar.update(1)
                wandb.log({"video": done_video, "fail": failed_video_num})
                

            else:
                done_process += 1

            if done_process == nprocess:
                break


    tqdm.write(f"total videos: {len(data_dict)} failed num: {failed_video_num}")
    tqdm.write("Waiting for all process to end")
    for i in range(nprocess):
        process_pool[i].join()

    tqdm.write("Emptying process queue")
    while not queue.empty():

        pack = queue.get()

        if pack["error"]:
            # error occurred while reading/processing the video
            logger.write(f"{pack['video_uid']},error:{pack['error']}\n")
        else:
            frame_processed = pack["frame_processed"]
            missed_frame = pack["missed_frame"]
            logger.write(f"{pack['video_uid']},success,frames:{frame_processed},missed:{','.join(missed_frame)}\n")



if __name__ == "__main__":

    args = parse_terminal_args()
    main(args)
