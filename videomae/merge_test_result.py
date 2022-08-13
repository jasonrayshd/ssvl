import os
import re
import numpy as np
import json
import argparse

# "/data/shared/output/preepic55ftego4d_A2/0.txt"

# parse required arguments
parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, help="path of prediction file")
parser.add_argument("--num_crop", default=3, type=int, help="number of spatial crops for each test video")
parser.add_argument("--annotation_file", type=str,
                    default="/data/shared/ego4d/v1/annotations/fho_oscc-pnr_test_unannotated.json" ,
                    help="path of test annotation file"
                    )

args = parser.parse_args()


path = args.path
output_path = "/".join(args.path.split("/")[:-1])
num_crop = args.num_crop
annotation_file = args.annotation_file

# find all results
raw = open(path, "r").read()
pattern_twohead = "(.*?) \[(.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?)\] \[(.*?), (.*?)\] (\d)"
pred_dict = {}
results = re.findall(pattern_twohead, raw)

# wash results
for result in results:
    id = result[0]

    loc_preds = result[1:18]
    cls_pred = result[18:20]
    crop_num = result[20]
    if id not in pred_dict.keys():
        pred_dict[id] = {
            0:{},
            1:{},
            2:{},
        }
    pred_dict[id][int(crop_num)] = {
        "loc": [float(pred) for pred in loc_preds],
        "cls": [float(pred) for pred in cls_pred],
    }

# combine results
final_preds = {}
for k,v in pred_dict.items():
    loc = []
    cls = 0
    for i in range(num_crop):
        try:
            loc.append(np.argmax(v[i]["loc"]))
            cls += np.argmax(v[i]["cls"])
        except:
            # in case some predictions are missing
            print(f"{i}th crop of {k} do not exist ")
            continue

    final_preds[k] = [loc, cls/num_crop]


# save results to json file

cls_final = []
for k, v in final_preds.items():
    cls_final.append({
        "unique_id": k,
        "state_change": True if v[1] > 0.5 else False,
    })

clip_rawinfo = json.load(open(annotation_file))["clips"]
clip_dict = {}

for clip in clip_rawinfo:
    id = clip["unique_id"]
    sf = clip["parent_start_frame"]
    clip_dict[id] = sf

loc_final = []
for k, v in final_preds.items():
    pnr_frame = np.mean(v[0])
    loc_np = np.array(v[0])
    if v[1] > 0.5:
        # model predict that state change occurs in given clip
        pnr_frame = np.where(loc_np==16, np.zeros_like(loc_np), loc_np)
        pnr_frame = np.mean(pnr_frame)
        # print(pnr_frame)
        if pnr_frame == 0:
            pnr_frame = 16

    loc_final.append({
        "unique_id": k,
        "pnr_frame":  pnr_frame + clip_dict[k]
    })


cls_bar = open(os.path.join(output_path, "cls_final.json"), "w")
cls_bar.write(json.dumps(cls_final))

loc_bar = open(os.path.join(output_path, "cls_pred.json"), "w")
loc_bar.write(json.dumps(loc_final))