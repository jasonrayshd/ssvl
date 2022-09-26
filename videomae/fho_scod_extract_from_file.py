# (1) Extract pre/pnr/post frames for state change object detection
# (2) Save dict object that map pnr frame to its pre and post frames into pickle object

from builtins import sorted
import av
import os
import cv2
import json
from tqdm import tqdm
from ego4d_trim import _get_frames
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str)

args = parser.parse_args()

def get_frame_for(uid, video_path, frames_list):
    cv2.setNumThreads(5)
    # official code where av == 6.0.0
    frame_num = 0
    container = av.open(video_path)
    for frame in _get_frames(
            frames_list,
            container,
            include_audio=False,
            audio_buffer_frames=0
        ):  
        frame = frame.to_rgb().to_ndarray()

        save_frame(uid, frame, frames_list[frame_num])

        frame_num += 1

    return frame_num

def save_frame(uid, frame, frame_idx):

    cv2.imwrite(
        os.path.join(
            frame_save_path,
            uid,
            f'{frame_idx}.jpeg'
        ),

        frame
    )


video_path = "/data/shared/ssvl/ego4d/v1/full_scale"
frame_save_path = "/data/shared/ssvl/ego4d/v1/fho_scod/pre_pnr_post_frames"
anno_file_path = [
    "/data/shared/ssvl/ego4d/v1/annotations/fho_scod_train.json",
    "/data/shared/ssvl/ego4d/v1/annotations/fho_scod_val.json",
]

lines = open(args.file, "r").readlines()

anno = []
for anno_file in anno_file_path:
    anno.extend( json.load(open(anno_file,"r"))["clips"] )


frames_list = []
pre_uid = None
for i, line in enumerate(tqdm(lines)):

    flag, item = line.strip("\n").split(":")
    if flag == "video":
        uid = item
        for clip in anno:
            if clip["video_uid"] == uid.split(".")[0]:
                frames_list.append(clip["pre_frame"]["frame_number"])
                frames_list.append(clip["pnr_frame"]["frame_number"])
                frames_list.append(clip["post_frame"]["frame_number"])
        print(f"saving frames for {uid}")
        os.makedirs(os.path.join(frame_save_path, uid.split(".")[0]), exist_ok=True)
        frames = get_frame_for(uid, os.path.join(video_path, uid), frames_list)
        print(frame_num)
        print("Done")
        frames_list = []

    elif flag == "frame":
        uid, frame = item.split("/")
        uid = uid.split(".")[0]
        frame_idx = int(frame.split(".")[0])

        if pre_uid is None or (uid == pre_uid and i != len(lines)-1):
            frames_list.append(frame_idx)

        elif i == len(lines) - 1:
            frames_list.append(frame_idx)
            frame_num = get_frame_for(pre_uid, os.path.join(video_path, pre_uid+".mp4"), frames_list)
            print(frame_num)

        elif uid != pre_uid:
            frame_num = get_frame_for(pre_uid, os.path.join(video_path, pre_uid+".mp4"), frames_list)
            print(frame_num)

            frames_list = [frame_idx]

        pre_uid = uid