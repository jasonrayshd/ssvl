import argparse
import os
import av
import cv2
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import zipfile
import wandb

import shutil
from ego4d_trim import _get_frames
import threading
import multiprocessing

def short_side_resize(frame, desired_shorter_side):
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
    return frame


def decode_all_frame(frame_lst, clip_lst, pack_zip_fn, video_path, frame_dest, zip_dest, short_side_size, queue):

    process_name = multiprocessing.current_process().name
    video_uid = video_path.split("/")[-1].split(".")[0]
    try:
        container = av.open(video_path)
    except Exception as e:
        queue.put({"process_name":process_name, "video_uid":video_uid, "state":"fail", "error":str(e)})

    frame_lst = sorted(list(set(frame_lst)))

    if len(frame_lst) == 0:
        print(f"frame list is Empty: {video_uid}")
        queue.put({"process_name":process_name, "video_uid":video_uid, "state":"error"})
        return

    # filtered_frame_lst = []
    # for i in range(len(frame_lst)):
    #     if os.path.exists(os.path.join(frame_dest, "{:010d}.jpg".format(frame_lst[i]))):
    #         continue
    #     filtered_frame_lst.append(frame_lst[i])    
    # frame_lst = filtered_frame_lst

    # if len(frame_lst) == 0:
    #     queue.put({"process_name":process_name, "video_uid":video_uid, "state":"success"})
    #     return

    frame_iter = _get_frames(frame_lst, container, include_audio=False, audio_buffer_frames=0)

    idx = 0
    for frame in frame_iter:

        frame = frame.to_rgb().to_ndarray()
        frame = short_side_resize(frame, short_side_size)
        cv2.imwrite(
            os.path.join(frame_dest, "{:010d}.jpg".format(frame_lst[idx])),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        )
        idx += 1

    container.close()

    # save into zip file
    tmp_dir = "./" + str(multiprocessing.current_process().pid)
    pack_zip_fn(clip_lst, frame_dest, zip_dest, tmp_dir)

    # delete all temporal images
    os.system(f"rm -rf {frame_dest}")

    queue.put({"process_name":process_name, "video_uid":video_uid, "state":"success"})
    

def parse_terminal_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--source", type=str, help="path to full_scale videos")
    parser.add_argument("--dest", type=str, help="path to store video frames")
    parser.add_argument("--anno_path", type=str, help="path to annotations")
    parser.add_argument("--max_num_process", type=int, help="maximum number of processes to use")
    parser.add_argument("--short_side_size", type=int, help="desired short side size of frame")
    parser.add_argument("--logfile", type=str, help="path to log file")
    parser.add_argument("--task", type=str, help="one of [hands, lta, sta]")

    parser.add_argument("--debug", action="store_true", help="whether init wandb loggin or not")

    return parser.parse_args()


def write2log(ret, logfile):

    logfp = open(logfile,"a+")
    if ret["state"] == "success":
        logfp.write(f"{ret['video_uid']},success\n")
        logfp.close()
        return 0
    else:
        logfp.write(f"{ret['video_uid']},{ret['error']}\n")
        logfp.close()
        return 1


def gather_for_fho_lta(anno_path, finished_videos):

    files = [
        os.path.join(anno_path, "fho_lta_train.json"),
        os.path.join(anno_path, "fho_lta_val.json"),
        os.path.join(anno_path, "fho_lta_test_unannotated.json")
    ]
    clip_uid_to_frames = {}
    video_uid_to_clip= {}
    skipped = 0

    for file in files:
        split = file.split(".")[0].split("_")[2] # record the split for each clip
        with open(file, "r") as fp:
            clips = json.load(fp)["clips"]

            for clip in clips:
                video_uid = clip["video_uid"]
                clip_uid = clip["clip_uid"]

                st = int(clip["clip_parent_start_frame"])
                end = int(clip["clip_parent_end_frame"])
                idx_range = list(range(st, end+1))

                if video_uid not in video_uid_to_clip.keys():
                    video_uid_to_clip[video_uid] = []

                if clip_uid not in clip_uid_to_frames.keys():
                    clip_uid_to_frames[clip_uid] = idx_range

                video_uid_to_clip[video_uid].append((clip, split)) # clip contains different information but with same parent start/end frame

    new_video_uid_to_frames = {}
    for video_uid in video_uid_to_clip.keys():

        if video_uid in finished_videos:
            skipped += 1
            print(f"skipped {video_uid}")
            continue
        else:
            if video_uid not in new_video_uid_to_frames.keys():
                new_video_uid_to_frames[video_uid] = []

            for elem in video_uid_to_clip[video_uid]:
                clip = elem[0]
                clip_uid = clip["clip_uid"]
                # might contain repeated frames, will be addressed before extracting frames from videos
                new_video_uid_to_frames[video_uid].extend(clip_uid_to_frames[clip_uid])

    print(f"skipped {skipped} videos")

    return new_video_uid_to_frames, video_uid_to_clip


def gather_for_fho_hands(anno_path, finished_videos):

    files = [
        # os.path.join(anno_path, "fho_hands_train.json"),
        # os.path.join(anno_path, "fho_hands_val.json"),
        os.path.join(anno_path, "fho_hands_test_unannotated.json"),
    ]
    clip_uid_to_frames = {}
    video_uid_to_clip= {}
    skipped = 0

    max_observation_time = 4 # maximum observation time
    max_observation_frame_num = max_observation_time * 30

    # process train and val set first
    for file in files:
        split = file.split(".")[0].split("_")[2]
        with open(file, "r") as fp:
            clips = json.load(fp)["clips"]
            for clip in clips:
                video_uid = clip["video_uid"]
                clip_uid = clip["clip_uid"]

                if video_uid not in video_uid_to_clip.keys():
                    video_uid_to_clip[video_uid] = []

                if clip_uid not in clip_uid_to_frames.keys():
                    clip_uid_to_frames[clip_uid] = []
                    video_uid_to_clip[video_uid].append((clip, split))

                for frame_entry in clip["frames"]:

                    if "test" in file:
                        pre_45_frame = frame_entry["pre_45"]["frame"]
                        pre_frame = frame_entry["pre_frame"]["frame"] if "pre_frame" in frame_entry else -1
                        st = max(0, pre_45_frame - max_observation_frame_num)
                        end = pre_frame + 30 if pre_frame != -1 else pre_45_frame + 45 + 30
                        idx_range = list(range(st, end+1))
                    else:
                        st = int(frame_entry["action_start_frame"])
                        end = int(frame_entry["action_end_frame"])
                        idx_range = list(range(st, end+1))

                    clip_uid_to_frames[clip_uid].extend(idx_range)

                clip_uid_to_frames[clip_uid] = list(set(clip_uid_to_frames[clip_uid]))


    new_video_uid_to_frames = {}
    for video_uid in video_uid_to_clip.keys():

        if video_uid in finished_videos:
            skipped += 1
            print(f"skipped {video_uid}")
            continue
        else:
            if video_uid not in new_video_uid_to_frames.keys():
                new_video_uid_to_frames[video_uid] = []

            for elem in video_uid_to_clip[video_uid]:
                # might contain repeated frames, will be addressed before extracting frames from videos
                clip = elem[0]
                clip_uid = clip["clip_uid"]
                new_video_uid_to_frames[video_uid].extend(clip_uid_to_frames[clip_uid])

            new_video_uid_to_frames[video_uid] = list(set(new_video_uid_to_frames[video_uid]))

    print(f"skipped {skipped} videos")

    return new_video_uid_to_frames, video_uid_to_clip


def gather_for_fho_sta(anno_path, finished_videos):

    files = [
        os.path.join(anno_path, "fho_sta_train.json"),
        os.path.join(anno_path, "fho_sta_val.json"),
        os.path.join(anno_path, "fho_sta_test_unannotated.json"),
    ]
    clip_uid_to_frames = {}
    video_uid_to_clip= {}
    skipped = 0

    max_observation_time = 4
    max_observation_frame = max_observation_time*30
    for file in files:
        split = file.split(".")[0].split("_")[2]
        clips = json.load(open(file, "r"))["annotations"]
        for clip in clips:
            video_uid = clip["video_id"]
            clip_uid = clip["clip_uid"]
            frame = int(clip["frame"])

            if video_uid not in video_uid_to_clip.keys():
                video_uid_to_clip[video_uid] = []

            if clip_uid not in clip_uid_to_frames.keys():
                clip_uid_to_frames[clip_uid] = []
                video_uid_to_clip[video_uid].append((clip, split))

            st = max(0, frame-max_observation_frame)
            end = frame + 30 + 1
            clip_uid_to_frames[clip_uid].extend(list(range(st, end)))

    new_video_uid_to_frames = {}
    for video_uid in video_uid_to_clip.keys():

        if video_uid in finished_videos:
            skipped += 1
            print(f"skipped {video_uid}")
            continue
        else:
            if video_uid not in new_video_uid_to_frames.keys():
                new_video_uid_to_frames[video_uid] = []

            for elem in video_uid_to_clip[video_uid]:
                clip = elem[0]
                clip_uid = clip["clip_uid"]
                new_video_uid_to_frames[video_uid].extend(list(set(clip_uid_to_frames[clip_uid])))

            new_video_uid_to_frames[video_uid] = list(set(new_video_uid_to_frames[video_uid]))

    return new_video_uid_to_frames, video_uid_to_clip


def distribute_fho_sta(clip_lst, source, dest, tmp_dir):

    max_observation_time = 4 # maximum observation time
    max_observation_frame = max_observation_time * 30

    # start distributing
    for i, elem in enumerate(clip_lst):

        clip, split = elem
        if i == 0:
            os.makedirs(os.path.join(dest, split), exist_ok=True)

        uid = clip["uid"]
        video_uid = clip["video_id"]
        clip_uid = clip["clip_uid"]
        frame = clip["frame"]

        st = max(0, frame-max_observation_frame)
        end = frame + 30

        os.makedirs(os.path.join(dest, split, clip_uid), exist_ok=True)
        final_dest = os.path.join(dest, split, clip_uid, uid+".zip")

        os.makedirs(os.path.join(tmp_dir, split, clip_uid), exist_ok=True)
        tmp_dest = os.path.join(tmp_dir, split, clip_uid, uid+".zip")

        with zipfile.ZipFile(tmp_dest, "w") as zipfp:
            for idx in range(int(st), int(end)+1):
                frame_path = os.path.join(source, "{:010d}.jpg".format(idx))
                if os.path.exists(frame_path):
                    zipfp.write(frame_path, arcname="{:010d}.jpg".format(idx))

        shutil.move(tmp_dest, final_dest)


def distribute_fho_hands(clip_lst, source, dest, tmp_dir):

    # anno_files = [
    #     os.path.join( anno_path, "fho_hands_train.json"),
    #     os.path.join( anno_path, "fho_hands_val.json"),
    #     os.path.join( anno_path, "fho_hands_test_unannotated.json"),
    # ]

    # # process annotation file
    # split2clips = {}
    # for anno_file in anno_files:
    #     split = anno_file.split(".")[0].split("_")[2]
    #     with open(anno_file, "r") as fp:
    #         content = json.load(fp)
    #     clips = content["clips"]
    #     split2clips[split] = clips

    max_observation_time = 4 # maximum observation time
    max_observation_frame_num = max_observation_time * 30
    # start distributing
    for elem in clip_lst:

        clip, split = elem
        os.makedirs(os.path.join(dest, split), exist_ok=True)

        video_uid = clip["video_uid"]
        clip_uid = clip["clip_uid"]

        # if clip_uid != "33b46d1c-b0d2-40c2-b10c-bfd651298e56":
        #     continue

        for i, frame_entry in enumerate( clip["frames"] ):

            if split in ["train", "val"]:
                st = frame_entry["action_start_frame"]
                end = frame_entry["action_end_frame"]

            else:
                pre_45_frame = frame_entry["pre_45"]["frame"]
                pre_frame = frame_entry["pre_frame"]["frame"] if "pre_frame" in frame_entry else -1
                st = max(0, pre_45_frame - max_observation_frame_num)
                end = pre_frame + 30 if pre_frame != -1 else pre_45_frame + 45 + 30

            # Considering the network transmit speed, 
            # First, select corresponding image files and pack it into zip file locally
            # After that, move the entire zip file to remote destination
            os.makedirs(os.path.join(dest, split, clip_uid), exist_ok=True)
            final_dest = os.path.join(dest, split, clip_uid, clip_uid+"_{:05d}.zip".format(i))

            os.makedirs(os.path.join(tmp_dir, split, clip_uid), exist_ok=True)
            tmp_dest = os.path.join(tmp_dir, split, clip_uid, clip_uid+"_{:05d}.zip".format(i))

            with zipfile.ZipFile(tmp_dest, "w") as zipfp:
                for idx in range(int(st), int(end)+1):
                    frame_path = os.path.join(source, "{:010d}.jpg".format(idx) )
                    if os.path.exists(frame_path):
                        zipfp.write(frame_path, arcname="{:010d}.jpg".format(idx))

            shutil.move(tmp_dest, final_dest)


def distribute_fho_lta(clip_lst, source, dest, tmp_dir):

     for i, elem in enumerate(clip_lst):

        clip, split = elem
        if i == 0:
            os.makedirs(os.path.join(dest, split), exist_ok=True)

        start = clip["clip_parent_start_frame"] + clip["action_clip_start_frame"]
        end = clip["clip_parent_start_frame"] + clip["action_clip_end_frame"]
        action_idx = clip["action_idx"]

        # Considering the network transmit speed, 
        # First, select corresponding image files and pack it into zip file locally
        # After that, move the entire zip file to remote destination
        os.makedirs(os.path.join(dest, split, clip["clip_uid"]), exist_ok=True)
        final_dest = os.path.join(dest, split, clip["clip_uid"], clip["clip_uid"]+"_{:05d}.zip".format(action_idx))
        # temporal zip file
        os.makedirs(os.path.join(tmp_dir, split, clip["clip_uid"]), exist_ok=True)
        tmp_dest = os.path.join(tmp_dir, split, clip["clip_uid"], clip["clip_uid"]+"_{:05d}.zip".format(action_idx))

        with zipfile.ZipFile(tmp_dest, "w") as zipfp:
            for idx in range(int(start), int(end)+1):
                zipfp.write(os.path.join(source, "{:010d}.jpg".format(idx) ), arcname="{:010d}.jpg".format(idx))

        # move local zip file to remote destination
        shutil.move(tmp_dest, final_dest)

def main(args):

    source = args.source
    dest = args.dest
    max_num_process = args.max_num_process
    short_side_size = args.short_side_size
    logfile = args.logfile

    finished_videos = []
    open(logfile, "a+")
    with open(logfile, "r") as logfp:
        logs = logfp.readlines()
        for log in logs:
            video_uid, state = log.split(",")
            if state.strip("\n") == "success":
                finished_videos.append(video_uid)

    print(f"found {len(finished_videos)} successfully processed videos")

    if args.task == "lta":
        video_uid_to_frames, video_uid_to_clip = gather_for_fho_lta(args.anno_path, finished_videos)
        pack_zip_fn = distribute_fho_lta
    elif args.task == "hands":
        video_uid_to_frames, video_uid_to_clip = gather_for_fho_hands(args.anno_path, finished_videos)
        pack_zip_fn = distribute_fho_hands
    elif args.task == "sta":
        video_uid_to_frames, video_uid_to_clip = gather_for_fho_sta(args.anno_path, finished_videos)
        pack_zip_fn = distribute_fho_sta


    video_uids = list(video_uid_to_frames.keys())
    # print(video_uids)
    # print(len(video_uid_to_frames["aa2cbed5-ae1f-44e3-9043-6e90826f387b"]), len(video_uid_to_frames["d362ba00-301d-4e88-8786-c4da2684e62e"]))
    # return 
    i = 0
    active_process_num = 0
    fail_num = 0
    process_pool = {}
    queue = multiprocessing.Queue(maxsize=2*max_num_process)
    progress = tqdm(total=len(video_uids))
    
    zip_dest = dest
    tmp_dir = "./"

    while i < len(video_uids):
        if active_process_num < max_num_process:
            video_uid = video_uids[i]
            video_path = os.path.join(source, video_uid+".mp4")

            if not os.path.exists(video_path):
                fail_num += write2log({"video_uid":video_uid, "state":"fail", "error":"video not exist"}, logfile)
                i += 1
                continue

            frame_dest = os.path.join(tmp_dir, video_uid)

            os.makedirs(frame_dest, exist_ok=True)

            process_name = f"process-{i}"
            process = multiprocessing.Process(
                target=decode_all_frame, 
                args=(video_uid_to_frames[video_uid], video_uid_to_clip[video_uid], pack_zip_fn ,video_path, frame_dest, zip_dest, short_side_size, queue), 
                name=process_name
            )
            process.start()
            process_pool[process_name] = process

            active_process_num += 1
            i+=1
        else:
            ret = queue.get()
            fail_num += write2log(ret, logfile)
            process_name = ret["process_name"]
            process_pool[process_name].join()

            active_process_num -=1
            progress.update(1)
            progress.set_postfix_str(f"fail num:{fail_num}")

    print("waiting for all threads to end")
    for k, process in process_pool.items():
        process.join()

    print("emptying queue")
    while not queue.empty():
        ret = queue.get()
        fail_num += write2log(ret, logfile)
        process_name = ret["process_name"]
        process_pool[process_name].join()

        progress.update(1)
        progress.set_postfix_str(f"fail num:{fail_num}")


if __name__ == "__main__":
    args = parse_terminal_args()
    if not args.debug:
        wandb.init(project="data")
    main(args)