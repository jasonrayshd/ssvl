import os
import json
import decord
# import wandb

# wandb.init(project="ego4d-state-change-videomae")
# print(f"current working directory: {os.getcwd()}")
import cv2

anno = "/mnt/shared/random/ego4d/v1/annotations/fho_oscc-pnr_train.json"
video_path = "/mnt/shared/random/ego4d/v1/full_scale/6c03be74-4692-4e3e-8eab-01f9f8e0d3ba.mp4"
info = json.load(open(anno, "r"))

frame_list = [8907, 8908, 8909, 8910, 8911, 8912, 8913, 8914, 8915, 8916, 8917, 8918, 8919, 8920, 8921, 8922, 8923, 8924, 8925, 8926, 8927, 8928, 8929, 8930, 8931, 8932, 8933, 8934, 8935, 8936, 8937, 8938, 8939, 8940, 8941]
frame_list2 = [8942, 8943, 8944, 8945, 8946, 8947, 8948, 8949, 8950, 8951, 8952, 8953, 8954, 8955, 8956, 8957, 8958, 8959, 8960, 8961, 8962, 8963, 8964, 8965, 8966, 8967, 8968, 8969, 8970, 8971, 8972, 8973, 8974, 8975, 8976]
vr = decord.VideoReader(video_path)
try:
    frames = vr.get_batch([1000]).asnumpy()
    print(frames.shape)
except:

    cap = cv2. VideoCapture(video_path)
    length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
    print(length)

    cap.set(2,1000);
    ret, frame = cap.read()
    print(ret)
    print(frame.shape)  


# for clip in info["clips"]:
#     if clip["video_uid"] != "6c03be74-4692-4e3e-8eab-01f9f8e0d3ba":
#         continue
    
#     frame_list = [i for i in range( int(clip["parent_start_frame"]), int(clip["parent_end_frame"]+1) ) ]
#     vr = decord.VideoReader(video_path)
#     try:
#         frames = vr.get_batch(frame_list).asnumpy()
#     except decord._ffi.base.DECORDError as e:
#         print(frame_list)
#         raise e
