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


# def thread_worker(video_meta, source, tmp_dir, gpu_id, thread_queue):
#     # print(video_meta)
#     uid, clip_metas = video_meta[0], video_meta[1:]

#     clip2frame = {} # dict{ str: dict{"frame_%010d": "frame_%010d_%010d"} }
#     # print("clip_metas", len(clip_metas))
#     # extract all frames first
#     for c_meta in clip_metas:

#         uid = c_meta["video_uid"]
#         clip_name = c_meta["clip_name"]
#         # st_f, end_f = c_meta["start_frame"], c_meta["end_frame"]
#         clip2frame[clip_name] = {}

#         if not os.path.exists(os.path.join(source, uid, clip_name, "frames.zip")):
#             thread_queue.put({
#                 "thread_name": threading.current_thread().name,
#                 "clip_name": clip_name,
#                 "thread_end": False,
#                 "state": "cannot find frames.zip",
#             })
#             print(f"{clip_name}: cannot find frames.zip")
#             continue
            
#         if os.path.exists(os.path.join(source, uid, clip_name, "flows.zip")):
#             # flows.zip already exists
#             try:
#                 with zipfile.ZipFile(os.path.join(source, uid, clip_name, "flows.zip")) as zf:
#                     flow_lst = zf.namelist()

#                 thread_queue.put({
#                     "thread_name": threading.current_thread().name,
#                     "clip_name": clip_name,
#                     "thread_end": False,
#                     "state": "exist",
#                     })
#                 continue

#             except:
#                 # flow zip corrupted
#                 # do nothing and start extracting flow
#                 pass

#         with zipfile.ZipFile(os.path.join(source, uid, clip_name, "frames.zip")) as zf:
#             try:
#                 # extract all frames
#                 frame_lst = zf.namelist() #  list[str], e.g., frame_0000000758_0000002606.jpg
#                 # video_frame_num += len(frame_lst)

#                 zf.extractall(path=tmp_dir)
#                 # print(os.listdir(tmp_dir))
#                 for i, clip_frame_name in enumerate(frame_lst): # clip_frame_name: looks like frame_0000000758_0000002606.jpg

#                     video_frame_idx = int(clip_frame_name.split("_")[1]) + 1       # let index start from 1
#                     video_frame_name = "frame_{:010d}".format(video_frame_idx) # global index of the frame, of format: frame_%010d

#                     # load with tolerance
#                     ret = exec_with_tolerance(
#                         func = os.rename,
#                         retry=5, 
#                         src = f"{tmp_dir}/{clip_frame_name}", 
#                         dst = f"{tmp_dir}/{video_frame_name}.jpg"
#                     )

#                     if i == len(frame_lst) - 1:
#                         # skip last frame, since the corresponding 
#                         # flow does not belong to this clip
#                         continue

#                     clip2frame[clip_name][video_frame_name] = clip_frame_name.split(".")[0]

#             except Exception as e:

#                 clip2frame.pop(clip_name) # Remove the clip from frame mapping dict object

#                 thread_queue.put({
#                     "thread_name": threading.current_thread().name,
#                     "thread_end": False,
#                     "clip_name": clip_name,
#                     "state": str(e),
#                     # "error": str(e),
#                 })
#                 print(f"{clip_name}: {str(e)}, {os.listdir(tmp_dir)}")
#                 continue

#     # start extracting frames for the video
#     # command format: compute_flow_wrapper $input $output 
#     # generated flows will be saved in tmp_dir/u and tmp_dir/v with image's index starts from 1 in format frame_%010d.jpg
#     os.system(f"bash ./compute_flow_wrapper.sh {tmp_dir} {tmp_dir} frame_%010d.jpg -g {gpu_id} -b 8")
#     # DELETE rgb frames to free up disk space
#     os.system(f"rm -rf {tmp_dir}/*.jpg")

#     # distribute flows for each clip
#     for clip_name, clip_frame_book in clip2frame.items():
#         with zipfile.ZipFile(f"{tmp_dir}/{clip_name}.zip", "a") as zf:
#             for video_frame_name, clip_frame_name in clip_frame_book.items():

#                 # print(video_frame_name, clip_name, clip_frame_name)
#                 u_bytes = exec_with_tolerance(
#                         cv2.imencode,
#                         retry = 5,
#                         ext=".jpg", 
#                         img=cv2.imread(f"{tmp_dir}/u/{video_frame_name}.jpg"),
#                     )[1].tobytes()

#                 v_bytes = exec_with_tolerance(
#                         cv2.imencode,
#                         retry = 5,
#                         ext=".jpg", 
#                         img=cv2.imread(f"{tmp_dir}/v/{video_frame_name}.jpg") 
#                     )[1].tobytes()

#                 zf.writestr(f"u/{clip_frame_name}.jpg", u_bytes)
#                 zf.writestr(f"v/{clip_frame_name}.jpg", v_bytes)

#         # print("moving zip file")
#         ret = exec_with_tolerance(
#             shutil.move,
#             retry = 5,
#             src=f"{tmp_dir}/{clip_name}.zip", 
#             dst=os.path.join(source, uid, clip_name, "flows.zip")
#         )

#         thread_queue.put({
#                 "thread_name": threading.current_thread().name,
#                 "thread_end": False,
#                 "clip_name": clip_name,
#                 "state": "success", # extract flows for the clip no matter whether missing file exists
#         })

#     thread_queue.put({
#         "thread_name": threading.current_thread().name,
#         "thread_end": True,
#         "clip_name": "",
#         "state": "",
#     })

#     # delete remaining flow images
#     os.system(f"rm -rf {tmp_dir}")

#     return 0


def clip_thread_worker(video_meta, source, thread_tmp_dir, gpu_id, thread_queue):
    # print(video_meta)
    uid, clip_metas = video_meta[0], video_meta[1:]

    for c_meta in clip_metas:

        uid = c_meta["video_uid"]
        clip_name = c_meta["clip_name"]
        video_frame_name_to_clip_frame_name = {} # dict{ "frame_%010d": "frame_%010d_%010d" }
        tmp_dir = thread_tmp_dir + "/" + clip_name
        os.makedirs(tmp_dir, exist_ok=True)

        if not os.path.exists(os.path.join(source, uid, clip_name, "frames.zip")):
            thread_queue.put({
                "thread_name": threading.current_thread().name,
                "clip_name": clip_name,
                "thread_end": False,
                "state": "cannot find frames.zip",
            })
            print(f"{clip_name}: cannot find frames.zip")
            continue

        if os.path.exists(os.path.join(source, uid, clip_name, "flows.zip")):
            # flows.zip already exists
            try:
                with zipfile.ZipFile(os.path.join(source, uid, clip_name, "flows.zip")) as zf:
                    flow_lst = zf.namelist()

                thread_queue.put({
                    "thread_name": threading.current_thread().name,
                    "clip_name": clip_name,
                    "thread_end": False,
                    "state": "exist",
                    })
                continue

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
                thread_queue.put({
                    "thread_name": threading.current_thread().name,
                    "thread_end": False,
                    "clip_name": clip_name,
                    "state": str(e),
                    # "error": str(e),
                })

                print(f"{clip_name}: {str(e)}, {os.listdir(tmp_dir)}")
                continue

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
                thread_queue.put({
                        "thread_name": threading.current_thread().name,
                        "thread_end": False,
                        "clip_name": clip_name,
                        "state": message, # extract flows for the clip no matter whether missing file exists
                })   
                # DELETE frames/flows of current clip
                # os.system(f"rm -rf {tmp_dir}/*")
                os.system(f"rm -rf {tmp_dir}")


    thread_queue.put({
        "thread_name": threading.current_thread().name,
        "thread_end": True,
        "clip_name": "",
        "state": "",
    })

    # delete everything
    os.system(f"rm -rf {thread_tmp_dir}")

    return 0

def process_worker(data_list, source, max_num_threads, gpu_id, queue):

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
            tmp_dir = "./" + process_name + "./"+thread_name
            os.makedirs(tmp_dir, exist_ok=True)

            thread = threading.Thread(target=clip_thread_worker, args=(data_list[i], source, tmp_dir, gpu_id, thread_queue), name=thread_name)
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

    # print("waiting for all processes to end...")
    # for k, thread in thread_pool.items():
    #     thread.join()

    done_thread_num = 0
    print("emptying thread queue...")
    while done_thread_num < active_thread_num or not thread_queue.empty():
        pack = thread_queue.get()
        if pack["thread_end"]:
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


def write2log(pack, log_table):
    clip_name = pack["clip_name"]
    state = pack["state"]

    if "clip_name" not in log_table.keys():
        log_table["clip_name"] = []
    if "state" not in log_table.keys():
        log_table["state"] = []

    # print(clip_name, state)
    log_table["clip_name"].append(clip_name)
    log_table["state"].append(state)

    wandb.log({
        "video": len(log_table["clip_name"]),
        "log_table":wandb.Table(columns=["clip_name", "state"], data=list( zip( *list(log_table.values()) ) ) )
    })
    # print(wdb_tbl)

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
        total_clip_num += len(new_clip_metas)
        data_dict[video_uid] = new_clip_metas

    tqdm.write(f"skipped {skip_clip_num} clips")

    avail_gpus = args.gpus
    source = args.source
    nprocess = args.nprocess
    max_num_threads = args.max_num_threads

    chunked_data_dict = split_data(data_dict, nprocess) # list[ list[str, dict, dict, ...], list[], ... ]

    # for debugging
    # chunked_data_dict = [
    #     [
    #         ["ff8897f5-55e6-430b-8ea0-336d654a09e9", *data_dict["ff8897f5-55e6-430b-8ea0-336d654a09e9"]]

    #     ]
    # ]

    process_pool = {}
    queue = mlp.Queue(nprocess*max_num_threads*2)
    for i in range(nprocess):
        process_name = f"Process-{i}"
        gpu_id = avail_gpus[i % len(avail_gpus)]
        process = mlp.Process(target=process_worker, args=(chunked_data_dict[i], source, max_num_threads, gpu_id, queue), name=process_name)
        process.start()
        process_pool[process_name] = process

    iter = 0
    done_process = 0
    progress_bar = tqdm(total=total_clip_num)
    log_table = {}

    tqdm.write(f"recording logs to file: {args.logfile}")
    while iter < total_clip_num and done_process < nprocess :

        pack = queue.get()

        if not pack["process_end"]:
            # if process not end

            write2log(pack, log_table)
            iter += 1
            progress_bar.update(1)
            progress_bar.set_postfix_str(f"current process number:{nprocess-done_process}")

        else:
            print(f"Process: {pack['process_name']} finished")
            done_process += 1

        # tqdm.set_postfixstr

    print("emptying main process queue...")
    while not queue.empty():
        pack = queue.get()
        if pack["process_end"]:
            continue
        write2log(pack, log_table)


    print("done")

if __name__ == "__main__":

    args = parse_terminal_args()
    main(args)