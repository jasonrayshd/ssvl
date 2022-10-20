"""
split processed video to clip

"""

import os
import shutil
import csv
import signal
import argparse
from tqdm import tqdm
import threading
import multiprocessing as mlp

import zipfile

import wandb
wandb.init(project="preprocess_egoclip")

def parse_terminal_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, default="/data/shared/ssvl/ego4d/v1/egoclip", help="path to procesed videos")

    parser.add_argument("--logfile", type=str, default="video2clip.data", help="path to save log file")
    parser.add_argument("--processed_video_file", default="processed_video.data", type=str, help="path to video processing log file")

    parser.add_argument("--old_anno_path", type=str, default="/data/shared/ssvl/ego4d/v1/annotations/egoclip.csv", help="path to original egoclip annotation file")
    parser.add_argument("--new_anno_path", type=str, default="new_egoclip.csv", help="path to save new egoclip annotation file")

    parser.add_argument("--exclude", nargs="*", default=[], help="list of excluded videos")
    parser.add_argument("--max_thread_num", type=int, default=1, help="maximum number of threads")

    args = parser.parse_args()

    return args



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


def read_egoclip_csv(anno_path):
    """
        process egoclip csv annotation file

        Parameters:
            anno_path: str, path to egoclip csv annotation file

        Return:
            data_dict: a dict[list] object whose key is video uid and value is the list of frame index list of each **clip** (not video) 
            e.g.
            "uid": [[frame0, frame1, ...], [frame100, frame101, ...], ...]

    """
    reader = csv.reader(open(anno_path, "r", encoding='utf_8'))
    next(reader)
    rows = list(reader)
    progress_bar = tqdm(total = len(rows))
    data_dict = {}

    for row in rows:

        meta = row[0].split("\t")
        # print(meta)
        if meta[0] not in data_dict:
            data_dict[meta[0]] = []

        start_f = max(0, int( float(meta[5]) * 30) )
        end_f = max(0, int( float(meta[6]) * 30) )

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
            "clip_name": -1,
            "start_frame": -1,
            "end_frame": -1,
        })

        progress_bar.update(1)

    return data_dict


def process_information(args):
    """
        process frame extraction log file
    """
    processed_video_info = {}
    with open(args.processed_video_file, "r") as f:
        raw_log = f.readlines()
        for line in raw_log:
            if line == "\n":
                continue

            info = line.strip("\n").split(",")
            uid, state = info[:2]
            if state == "success":
                processed_frames = int( info[2].split(":")[1] )

                if uid not in processed_video_info.keys():

                    missed_frames = info[3].split(":")
                    if missed_frames[1] == '':
                        processed_video_info[uid] = {
                            "processed_frames":processed_frames,
                            "missed_frames": []
                        }
                    else:
                        tmp = [int(missed_frames[1])]
                        tmp.extend( [ int(raw_num) for raw_num in info[4:] ] )
                        processed_video_info[uid] = {
                            "processed_frames":processed_frames,
                            "missed_frames": tmp
                        }
                else:
                    print(f"[ATTENTION] Repeatedly processed video: {uid}")

    return processed_video_info


def worker(path, queue, video2clip_dict, processed_video_info):
    """

        path: str, path to the folder that contains video frames

        queue: multiprocessing.Queue

        video2clip_dict: dict[list], dict that contains list of frame index list of each clip
            e.g.
            "uid": [[frame0, frame1, ...], [frame100, frame101, ...], ...]

        processed_video_info: dict, dict that contains information(processed frame number, missed frame) of 
                            successfully processed videos 

    """

    uid = path.split("/")[-1]
    frameidx2times = {} # indicate how many times that each frame has been included in different clips
    # collect how many times that each frame has been included in different clips for next step
    for clip_frame_dict in video2clip_dict[uid]:
        for frameidx in clip_frame_dict["frames"]:
            if frameidx not in frameidx2times.keys():
                frameidx2times[frameidx] = 0
            frameidx2times[frameidx] += 1


    missed_frame = []
    clip_books = []
    opened_frame_dict = {}  # record pil object of opened image
    for i, clip_frame_dict in enumerate(video2clip_dict[uid]):
        clip_frame_lst = sorted(clip_frame_dict["frames"]) # ensure the order of each frame
        clip_name = uid+"_{:05d}".format(i)

        clip_meta = {}
        clip_meta.update(clip_frame_dict)
        clip_meta.pop("frames")
        clip_meta["idx"] = i
        clip_meta["clip_name"]= clip_name
        clip_meta["start_frame"] = clip_frame_lst[0]

        # ATTENTION: the final directory structure is:
        # dest/
        #   video-uid/
        #      clip-id/
        #           rgb/
        #               %05d.jpg
        #               ...
        #          flow/
        #               u/
        #                   %05d.jpg
        #                   ...
        #               v/
        #                   %05d.jpg
        #                   ...

        clip_path = os.path.join( path, clip_name, "rgb" )
        os.makedirs( clip_path, exist_ok=True)

        zip_path = os.path.join( path, clip_name)
        zf = zipfile.open(os.path.join(zip_path, "frames.zip"))
        for j, frame in enumerate( clip_frame_lst ):

            if frame in processed_video_info[uid]["missed_frames"]:
                # frame index exceed video duration, ignore
                break

            frame_path = os.path.join(path, str(frame)+".jpg")

            if os.path.exists(frame_path):
                if frame_path not in opened_frame_dict.keys()
                    opened_frame_dict[frame_path] = Image.open(frame_path)

                pil = opened_frame_dict[frame_path]
                io_buf = io.BytesIO()
                pil.save(io_buf, format="jpg")
                zf.writestr( str(frame)+".jpg", io_buf.getvalue())
             
                if frameidx2times[frame] != 1:
                    # copy frame and rename frame
                    shutil.copy(frame_path, os.path.join(clip_path, str(frame)+".jpg"))
                else:
                    # move frame and rename frame
                    shutil.move(frame_path, os.path.join(clip_path, str(frame)+".jpg"))

                frameidx2times[frame] -= 1

            else:
                # required frames are not processed yet
                missed_frame.append(str(frame))

            clip_meta["end_frame"] = frame

        clip_books.append(clip_meta)

    queue.put({
            "video_uid": uid,
            "missed_frames":missed_frame,
            "clip_books":clip_books,
            "thread_name": threading.current_thread().getName()
        })


def process_pack(pack, logger):
    """
    
    
    """
    uid = pack["video_uid"]
    missed_frames = pack["missed_frames"]
    clip_books = pack["clip_books"]

    # print(clip_books)
    if len(missed_frames) == 0:
        print(f"{uid},done\n")
        logger.write(f"{uid},done\n")
    else:
        print(f"{uid},{','.join(missed_frames)}\n")
        logger.write(f"{uid},{','.join(missed_frames)}\n")

    return clip_books


def write2csv(anno_file, clip_books):
    with open(anno_file,"a+") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")

        for clip in clip_books:
            
            row_info = list(clip.values())
            writer.writerow(row_info)


def main(args):
    old_anno_path = args.old_anno_path
    print("Processing egoclip annotation file..")
    data_dict = read_egoclip_csv(old_anno_path)
    
    source = args.source

    # process information of successfully processed videos
    processed_video_info = process_information(args)

    # do not process videos that have already been transformed into clips
    open(args.logfile, "a+").close()
    exclude_videos = []
    with open(args.logfile, "r") as log_fp:
        for line in log_fp.readlines():
            line = line.strip("\n")
            if line == ""  or line.startswith("processing"):
                continue
            meta = line.split(",")
            exclude_videos.append(meta[0])

    # extra list of videos to exclude
    for file in args.exclude:
        with open(file, "r") as f:
            exclude_videos.extend([line.strip("\n") for line in f.readlines() ])

    # deduplicate 
    exclude_videos = list(set(exclude_videos))
    num = 0
    for video in exclude_videos:
        processed_video_info.pop(video)
        num += 1

    print(f"{num} videos have been processed, ignore")

    # return
    processed_videos = [os.path.join(source, uid) for uid in list( processed_video_info.keys() )] # video uid list

    max_thread_num = args.max_thread_num
    queue = mlp.Queue(maxsize=max_thread_num*2)

    thread_pool = {}
    logger = open(args.logfile, "a+")

    # single process[debug]
    # worker(f"/data/shared/ssvl/ego4d/v1/egoclip/{processed_videos[0]}", queue, data_dict, processed_video_info)
    # pack = queue.get()
    # clip_books = process_pack(pack, logger)
    # write2csv(clip_books)

    # multi-threading
    pbar = tqdm(processed_videos)
    i = 0
    active_thread_num = 0
    done_video = 0
    try:
        while i < len(processed_videos):

            with DelayedKeyboardInterrupt():
                if active_thread_num < max_thread_num:
                    video_path = processed_videos[i]
                    td_name = f"Thread:{i}"
                    thread = threading.Thread(
                        target=worker, 
                        args =(video_path, queue, data_dict, processed_video_info),
                        name = td_name
                        )

                    thread.start()                    
                    thread_pool[td_name] = thread

                    # logger.write(f"processing {video_path.split('/')[-1]}\n")

                    i += 1
                    active_thread_num += 1
                    
                else:
                    tqdm.write("waiting for pack to arrive")
                    pack = queue.get()
                    clip_books = process_pack(pack, logger)
                    write2csv(args.new_anno_path, clip_books)

                    active_thread_num -= 1
                    td_name = pack["thread_name"]
                    thread_pool[td_name].join()
                    done_video += 1
                    pbar.update(1)
                    wandb.log({"video": done_video})

    except KeyboardInterrupt:
        print("Waiting for all threads to finish...")
        for k, td in thread_pool.items():
            td.join()

        print("Emptying queue")
        while not queue.empty():
            pack = queue.get()
            clip_books = process_pack(pack, logger)
            write2csv(args.new_anno_path, clip_books)

            done_video += 1
            pbar.update(1)
            wandb.log({"video": done_video})

    else:
        print("waiting for threads to finish")
        for k, td in thread_pool.items():
            td.join()

        print("emptying queue")
        while not queue.empty():
            pack = queue.get()
            clip_books = process_pack(pack, logger)
            write2csv(args.new_anno_path, clip_books)

            done_video += 1
            pbar.update(1)
            wandb.log({"video": done_video})

if __name__ == "__main__":

    args = parse_terminal_args()

    main(args)
