import os
import shutil
import zipfile
from tqdm import tqdm
import wandb
wandb.init(project="preprocess_egoclip")

path = "/mnt/shuang/Data/ego4d/preprocessed_data/egoclip"

uids = os.listdir(path)

for uid in tqdm(uids):
    
    if "_" not in uid: continue

    video_uid, clip_idx = uid.split("_")    
    tmp_path = os.path.join(path, uid)
    
    if len(os.listdir(tmp_path)) == 0:
        print(f"{tmp_path} is empty, deleting...")
        os.system(f"rm -r {tmp_path}")
