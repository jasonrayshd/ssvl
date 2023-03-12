import os
import json
import zipfile
import argparse
from tqdm import tqdm
import wandb


def parse_terminal_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="path to video frames")
    parser.add_argument("--dest", type=str, help="path to save zip files") 
    parser.add_argument("--anno_path", type=str, help="path to directory that contains annotaions")
    parser.add_argument("--task", type=str, help="task name, one of [hands, scod]")
    parser.add_argument("--debug", action="store_true", help="whether init wandb logging or not")
    return parser.parse_args()


def main(args):

    if args.task == "lta":
        distribute_fho_lta(args)
    elif args.task == "hands":
        distribute_fho_hands(args)
    elif args.task == "sta":
        distribute_fho_sta(args)


def distribute_fho_sta(args):
    
    source = args.source
    anno_path = args.anno_path
    dest = args.dest

    anno_files = [
        os.path.join( anno_path, "fho_sta_train.json"),
        os.path.join( anno_path, "fho_sta_val.json"),
        os.path.join( anno_path, "fho_sta_test_unannotated.json"),
    ]

    # process annotation file
    split2clips = {}
    for anno_file in anno_files:
        split = anno_file.split(".")[0].split("_")[2]
        with open(anno_file, "r") as fp:
            content = json.load(fp)
        clips = content["annotations"]
        split2clips[split] = clips

    max_observation_time = 4 # maximum observation time
    max_observation_frame = max_observation_time * 30

    # start distributing
    for split, clips in split2clips.items():
        os.makedirs(os.path.join(dest, split), exist_ok=True)

        for clip in tqdm(clips):
            uid = clip["uid"]
            video_uid = clip["video_id"]
            clip_uid = clip["clip_uid"]

            frame = clip["frame"]

            st = max(0, frame-max_observation_frame)
            end = frame + 30

            os.makedirs(os.path.join(dest, split, clip_uid), exist_ok=True)

            with zipfile.ZipFile(os.path.join(dest, split, clip_uid, uid+".zip"), "w") as zipfp:
                for idx in range(int(st), int(end)+1):
                    frame_path = os.path.join(dest, video_uid, "{:010d}.jpg".format(idx))
                    if os.path.exists(frame_path):
                        zipfp.write(frame_path, arcname="{:010d}.jpg".format(idx))
        # print(f"{clip['clip_uid']} Done")



def distribute_fho_hands(args):

    source = args.source
    anno_path = args.anno_path
    dest = args.dest

    anno_files = [
        os.path.join( anno_path, "fho_hands_train.json"),
        os.path.join( anno_path, "fho_hands_val.json"),
        os.path.join( anno_path, "fho_hands_test_unannotated.json"),
    ]

    # process annotation file
    split2clips = {}
    for anno_file in anno_files:
        split = anno_file.split(".")[0].split("_")[2]
        with open(anno_file, "r") as fp:
            content = json.load(fp)
        clips = content["clips"]
        split2clips[split] = clips

    max_observation_time = 4 # maximum observation time
    max_observation_frame_num = max_observation_time * 30
    # start distributing
    for split, clips in split2clips.items():
        os.makedirs(os.path.join(dest, split), exist_ok=True)

        for clip in tqdm(clips):
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
                    end = pre_frame + 30 if pre_frame == -1 else pre_45_frame + 45 + 30

                os.makedirs(os.path.join(dest, split, clip_uid), exist_ok=True)

                with zipfile.ZipFile(os.path.join(dest, split, clip_uid, clip_uid+"_{:05d}.zip".format(i)), "w") as zipfp:
                    for idx in range(int(st), int(end)+1):
                        frame_path = os.path.join(source, video_uid, "{:010d}.jpg".format(idx) )
                        if os.path.exists(frame_path):
                            zipfp.write(frame_path, arcname="{:010d}.jpg".format(idx))


def distribute_fho_lta(args):
    
    source = args.source
    anno_path = args.anno_path
    dest = args.dest

    anno_files = [
        os.path.join( anno_path, "fho_lta_test_unannotated.json"),
        os.path.join( anno_path, "fho_lta_train.json"),
        os.path.join( anno_path, "fho_lta_val.json"),
    ]

    # process annotation file
    split2clips = {}
    for anno_file in anno_files:
        split = anno_file.split(".")[0].split("_")[2]
        with open(anno_file, "r") as fp:
            content = json.load(fp)
        clips = content["clips"]
        split2clips[split] = clips

    # start distributing
    for split, clips in split2clips.items():
        os.makedirs(os.path.join(dest, split), exist_ok=True)

        for clip in tqdm(clips):
            start = clip["clip_parent_start_frame"] + clip["action_clip_start_frame"]
            end = clip["clip_parent_start_frame"] + clip["action_clip_end_frame"]
            action_idx = clip["action_idx"]
            os.makedirs(os.path.join(dest, split, clip["clip_uid"]), exist_ok=True)

            with zipfile.ZipFile(os.path.join(dest, split, clip["clip_uid"], clip["clip_uid"]+"_{:05d}.zip".format(action_idx)), "w") as zipfp:
                for idx in range(int(start), int(end)+1):
                    zipfp.write(os.path.join(source, clip["video_uid"], "{:010d}.jpg".format(idx) ), arcname="{:010d}.jpg".format(idx))

            # with zipfile.ZipFile(os.path.join(source, split, clip["clip_uid"], clip["clip_uid"]+"_{:05d}.zip".format(action_idx)), "a") as zipfp:

            #     zipfp.write(os.path.join(source, "frames", clip["video_uid"], "{:010d}.jpg".format(end) ), arcname="{:010d}.jpg".format(end))



if __name__ == "__main__":
    

    args = parse_terminal_args()
    if not args.debug:
        wandb.init(project="data")

    main(args)