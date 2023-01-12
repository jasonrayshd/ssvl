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
parser.add_argument("--json_files", nargs="+")

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

path = "/data/shared/ssvl/ego4d/v1/full_scale"
frame_save_path = "/data/shared/ssvl/ego4d/v1/fho_scod/pre_pnr_post_frames"
# json_files = ["/data/shared/ssvl/ego4d/v1/annotations/fho_scod_train.json", "/data/shared/ssvl/ego4d/v1/annotations/fho_scod_val.json", "/data/shared/ssvl/ego4d/v1/annotations/fho_scod_test_unannotated.json"]
# json_files = [ "/data/shared/ssvl/ego4d/v1/annotations/fho_scod_test_unannotated.json"]

for file in args.json_files:
    print(f"processing {file}")
    meta = json.load(open(file, "r"))

    pre_uid = None
    video_cap = None
    frame_idxs = []
    pbar = tqdm(total=len(meta["clips"]))
    print("Sorting clips")
    # clips =  meta["clips"]
    clips = sorted(meta["clips"], key=lambda x: x["video_uid"])
    print("Done")

    for i, clip in enumerate(clips):

        uid = clip["video_uid"]

        if i == 0 or uid == pre_uid:
            frame_idxs.extend([clip["pre_frame"]["frame_number"], clip["pnr_frame"]["frame_number"], clip["post_frame"]["frame_number"]])

        elif uid != pre_uid or i == len(clips) -1:
            if i == len(clips) -1:
                frame_idxs.extend([clip["pre_frame"]["frame_number"], clip["pnr_frame"]["frame_number"], clip["post_frame"]["frame_number"]])

            frame_idxs = list(set(frame_idxs))
            print(f"[{i}/{len(meta['clips'])}]saving {len(frame_idxs)} frames for {pre_uid}")

            if os.path.exists(os.path.join(frame_save_path, pre_uid)):
                # might have been processed before
                frames_exist = len( os.listdir(os.path.join(frame_save_path,pre_uid)) )
                if frames_exist >= len(frame_idxs):
                    print(f"Has been processed, {frames_exist} frames exist, Skip")
                    frame_idxs = [clip["pre_frame"]["frame_number"], clip["pnr_frame"]["frame_number"], clip["post_frame"]["frame_number"]]
                    pre_uid = uid
                    pbar.update(1)
                    continue

            os.makedirs(os.path.join(frame_save_path,pre_uid), exist_ok=True)
            frame_num = get_frame_for(pre_uid, os.path.join(path, pre_uid+".mp4"), frame_idxs)
            print(f"Sampled frame numbers:{frame_num}")

            frame_idxs = [clip["pre_frame"]["frame_number"], clip["pnr_frame"]["frame_number"], clip["post_frame"]["frame_number"]]
            # break

        pre_uid = uid
        pbar.update(1)