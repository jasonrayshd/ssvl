import os
import re
import numpy as np
import json


path = "/data/shared/output/preepic55ftego4d_A2/0.txt"
num_crop = 3
raw = open(path, "r").read()
pattern = "(.*?) \[(.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?), (.*?)\] \[(.*?), (.*?)\] (\d)"
# raw = "2583-1191-1199-2583 [-1.408203125, -1.3564453125, -1.4384765625, -1.25390625, -0.81640625, -0.04681396484375, 0.48486328125, 0.544921875, 0.2132568359375, -0.560546875, -1.0751953125, -1.583984375, -1.9150390625, -2.166015625, -2.376953125, -2.056640625, 1.80078125] [-0.09002685546875, 0.08953857421875] 2"

pred_dict = {}

# # for line in pred_lines:
results = re.findall(pattern, raw)

# print(len(results))
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


final_preds = {}
for k,v in pred_dict.items():
    loc = 0
    cls = 0
    for i in range(num_crop):
        # print(np.argmax(v[i]["loc"]))
        # print(np.argmax(v[i]["cls"]))
        try:
            loc += np.argmax(v[i]["loc"])
            cls += np.argmax(v[i]["cls"])
        except:
            print(f"{i}th crop of {k} do not exist ")
            continue

    final_preds[k] = [loc/num_crop, cls/num_crop]


# clip_info = json.load(open("/data/shared/ego4d/v1/annotations/fho_oscc-pnr_test_unannotated.json"))["clips"]

# loc_pred = open("cls_pred.txt", "a+")

cls_final = []
for k, v in final_preds.items():
    cls_final.append({
        "unique_id": k,
        "state_change": True if v[1] > 0.5 else False,
    })


cls_bar = open("cls_final.json", "w")
cls_bar.write(json.dumps(cls_final))
