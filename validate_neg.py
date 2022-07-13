import json
import av
from tqdm import tqdm

path = "/data/shared/ssvl/ego4d/v1"

anno_path = path + "/annotations/fho_oscc-pnr_val.json"
video_path = path + "/full_scale"

clips = json.load(open(anno_path, "r"))["clips"]

print("start")

for clip in tqdm(clips[-100:]):

    video = video_path + "/"+ str(clip["video_uid"]) + ".mp4"

    container = av.open(video)





