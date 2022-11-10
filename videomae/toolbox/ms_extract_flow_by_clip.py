import os
import csv
import cv2
import shutil
import zipfile
import argparse
from tqdm import tqdm

import time
import wandb
wandb.init(project="preprocess_egoclip")

import threading
import multiprocessing as mlp

def parse_terminal_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--logfile", type=str, default="extract_flow.data", help="Path to log file")
    parser.add_argument("--anno_path", type=str, default="/data/shared/ssvl/ego4d/v1/annotations/egoclip.csv", help="Path to egoclip annotation file")

    parser.add_argument("--source", type=str, default="/data/shared/ssvl/ego4d/v1/egoclip/", help="Path to source videos")
    parser.add_argument("--gpus", type=str, nargs="+", help="Available gpus")
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


def exec_with_tolerance(func, retry, **kwargs):
    i = 0
    while i < retry:
        try:
            # print(kwargs)
            return func(**kwargs)
        except Exception as e:
            i += 1
            if i == retry:
                raise e
            time.sleep(0.5)

def current_process_name():
    return mlp.current_process().name


def clip_process_worker(c_meta, source, process_tmp_dir, queue):

    uid = c_meta["video_uid"]
    clip_name = c_meta["clip_name"]
    video_frame_name_to_clip_frame_name = {} # dict{ "frame_%010d": "frame_%010d_%010d" }
    tmp_dir = process_tmp_dir + "/" + clip_name
    os.makedirs(tmp_dir, exist_ok=True)

    if not os.path.exists(os.path.join(source, uid, clip_name, "frames.zip")):
        queue.put({
            "process_name": current_process_name(),
            "clip_name": clip_name,
            "state": "cannot find frames.zip",
        })
        print(f"{clip_name}: cannot find frames.zip")
        return 1

    if os.path.exists(os.path.join(source, uid, clip_name, "flows.zip")):
        # flows.zip already exists
        try:
            with zipfile.ZipFile(os.path.join(source, uid, clip_name, "flows.zip")) as zf:
                flow_lst = zf.namelist()

            queue.put({
                "process_name": current_process_name(),
                "clip_name": clip_name,
                "state": "exist",
                })
            return 0

        except:
            # flow zip corrupted
            # do nothing and start re-extracting flow
            pass

    with zipfile.ZipFile(os.path.join(source, uid, clip_name, "frames.zip")) as zf:
        try:
            # extract all frames
            frame_lst = zf.namelist() #  list[str], e.g., frame_0000000758_0000002606.jpg
            zf.extractall(path=tmp_dir)
            for i, clip_frame_name in enumerate(frame_lst): # clip_frame_name: looks like frame_0000000758_0000002606.jpg

                # video_frame_idx = int(clip_frame_name.split("_")[1]) + 1       # let index start from 1
                video_frame_idx = i + 1
                video_frame_name = "frame_{:010d}".format(video_frame_idx) # global index of the frame, of format: frame_%010d

                # load with tolerance
                ret = exec_with_tolerance(
                    func = os.rename,
                    retry=5, 
                    src = f"{tmp_dir}/{clip_frame_name}", 
                    dst = f"{tmp_dir}/{video_frame_name}.jpg"
                )

                if i == len(frame_lst) - 1:
                    # skip last frame, since the corresponding 
                    # flow does not belong to this clip
                    continue

                video_frame_name_to_clip_frame_name[video_frame_name] = clip_frame_name.split(".")[0]

        except Exception as e:
            queue.put({
                "process_name": current_process_name(),
                "clip_name": clip_name,
                "state": str(e),
                # "error": str(e),
            })

            print(f"{clip_name}: {str(e)}, {os.listdir(tmp_dir)}")
            return 1

        # start extract optical flow
        # generated flows will be saved in tmp_dir/u and tmp_dir/v with image's index starts from 1 in format frame_%010d.jpg
        # use CPU
        ret_value = os.system(f"bash ./compute_flow_wrapper.sh  0 {tmp_dir} {tmp_dir} frame_%010d.jpg -b 8")

        message = "success"
        # prepare and transfer flows.zip
        try:
            with zipfile.ZipFile(f"{tmp_dir}/{clip_name}.zip", "a") as zf:
                for video_frame_name, clip_frame_name in video_frame_name_to_clip_frame_name.items():

                    # print(video_frame_name, clip_name, clip_frame_name)
                    u_bytes = exec_with_tolerance(
                            cv2.imencode,
                            retry = 5,
                            ext=".jpg", 
                            img=cv2.imread(f"{tmp_dir}/u/{video_frame_name}.jpg"),
                        )[1].tobytes()

                    v_bytes = exec_with_tolerance(
                            cv2.imencode,
                            retry = 5,
                            ext=".jpg", 
                            img=cv2.imread(f"{tmp_dir}/v/{video_frame_name}.jpg") 
                        )[1].tobytes()

                    zf.writestr(f"u/{clip_frame_name}.jpg", u_bytes)
                    zf.writestr(f"v/{clip_frame_name}.jpg", v_bytes)
        except Exception as e:
            # if error occurred
            message = str(e)
        else:
            # if successfully executed
            ret = exec_with_tolerance(
                shutil.move,
                retry = 5,
                src=f"{tmp_dir}/{clip_name}.zip", 
                dst=os.path.join(source, uid, clip_name, "flows.zip")
            )

        finally:
            # finally, send log information
            queue.put({
                    "process_name": current_process_name(),
                    "clip_name": clip_name,
                    "state": message, # extract flows for the clip no matter whether missing file exists
            })   
            # DELETE frames/flows of current clip
            # os.system(f"rm -rf {tmp_dir}/*")
            os.system(f"rm -rf {tmp_dir}")

    # delete everything
    os.system(f"rm -rf {process_tmp_dir}")

    return 0

def write2log(pack, logfile, num):
    clip_name = pack["clip_name"]
    state = pack["state"]

    wandb.log({"video": num})
    with open("tmp.log", "a+") as fp:
        fp.write(f"{clip_name},{state}\n")
        # print(f"{clip_name},{state}\n")

    shutil.copy("tmp.log", logfile+".tmp")


def main(args):

    tqdm.write("reading egoclip annotation file")
    data_dict = read_egoclip_csv(args.anno_path)

    open(args.logfile, "a+").close()

    checked_clips = []
    with open(args.logfile, "r") as logger:
        for line in logger.readlines():
            if line == "\n" or line == "":
                continue
            line = line.strip("\n")
            meta = line.split(",")
            # format of line: clip_uid,state
            if meta[1] == "success" or meta[1] == "exist":
                checked_clips.append(meta[0])

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
        # total_clip_num += len(new_clip_metas)
        data_dict[video_uid] = new_clip_metas

    tqdm.write(f"skipped {skip_clip_num} clips")

    # avail_gpus = args.gpus
    source = args.source
    nprocess = args.nprocess
    # max_num_threads = args.max_num_threads
    chunked_data_dict = split_data(data_dict, 48)

    clips = []
    for lst in chunked_data_dict[17]:
        clips.extend( lst[1:] )

    # print(len(clips))
    total_clip_num = len(clips)
    # clips = [ ]
    # print(clips[:20])
    # clips = list( data_dict.values() )

    i = 0
    active_process = 0
    process_pool = {}
    progress_bar = tqdm(total=total_clip_num)
    queue = mlp.Queue(nprocess*2)
    num = 0
    tqdm.write(f"recording logs to file: {args.logfile}")
    while i < total_clip_num:

        if active_process < nprocess:
            process_name = f"Process-{i}"
            process_tmp_dir = "./"+process_name
            process = mlp.Process(target=clip_process_worker, args=(clips[i], source, process_tmp_dir, queue), name=process_name)
            process.start()
            process_pool[process_name] = process
            active_process += 1
            i += 1
        else:
            num += 1
            pack = queue.get()  
            write2log(pack, args.logfile, num)

            process_name = pack["process_name"]
            process_pool[process_name].join()

            progress_bar.update(1)
            active_process -= 1

    # wait for process to end
    print("waiting for process to end")
    while active_process > 0:
        num += 1
        pack = queue.get()
        write2log(pack, args.logfile, num)

        process_name = pack["process_name"]
        process_pool[process_name].join()

        progress_bar.update(1)
        active_process -= 1

    print("done")

if __name__ == "__main__":
    args = parse_terminal_args()
    try:
        main(args)
    except Exception as e:
        print(str(e))