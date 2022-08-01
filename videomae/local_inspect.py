"""
Edited by jiachen

this file is used to inspect different part of pretraining code

"""
import cv2
import torch
import numpy as np
from einops import rearrange
import argparse
from config_utils import parse_yml, combine
from datasets import build_pretraining_dataset

def get_args():
    parser = argparse.ArgumentParser('VideoMAE pre-training script', add_help=False)

    parser.add_argument('--config', default='none', type=str,
                        help='path to configuration file')
    
    parser.add_argument('--overwrite', default='command-line', type=str,
                        help='')
 
    return parser.parse_args()

# inspect data preparation


opts = get_args()
config = parse_yml(opts.config)
if config is not None:
    args = combine(opts, config)


args.window_size = (8, 14, 14)
args.patch_size = 16

dataset_train = build_pretraining_dataset(args, cache_manager=None)

frame, mask = dataset_train[10100][:2]

mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1).repeat(1, 1, 224, 224)
std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1).repeat(1, 1, 224, 224)

frame = frame*std + mean
videos_squeeze = rearrange(frame, 'c (t p0) (h p1) (w p2) -> (t h w) (p0 p1 p2) c', p0=2, p1=16, p2=16)
videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
videos_patch = rearrange(videos_norm, '(t h w) (p0 p1 p2) c -> c (t p0) (h p1) (w p2)', t=8, h=14, w=14, p0=2, p1=16, p2=16)
frame = videos_patch


frame_sample = frame[:, 0, ...].numpy().transpose(1, 2, 0)
frame_sample = cv2.cvtColor(frame_sample, cv2.COLOR_RGB2BGR)
print(frame_sample.shape)
# # print(type(mask))
# # print(mask.shape)
# mask = torch.from_numpy(mask).to(torch.int8)
# mask_reshape = rearrange(mask, "(t h w) -> t h w", t=8, h=14, w=14).squeeze()

# mask_combine = mask_reshape[0]
# mask_combine = np.repeat( np.repeat(mask_combine, 16, axis=1), 16, axis=0)
# mask_combine =  np.expand_dims(mask_combine, axis=2)
# black = np.ones_like(frame_sample)
# # print(type(mask_combine))
# # print(mask_combine.shape)
# # print(mask_combine[:32,:32, 0])
# # print(frame_sample[mask_combine].shape)
# frame_sample = frame_sample*(1-mask_combine) + black*(mask_combine)


cv2.imwrite("inspect.png", frame_sample)


