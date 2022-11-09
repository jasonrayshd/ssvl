import torch
import torchvision
import torch.nn.functional as F

# from torchvision.utils import flow_to_image
# from torchvision.models.optical_flow import Raft_Large_Weights
# from torchvision.models.optical_flow import raft_large

import cv2
import time
from PIL import Image
# import matplotlib.pyplot as plt
import numpy as np
from flow_vis import flow_to_color

# plt.rcParams["savefig.bbox"] = "tight"
# # sphinx_gallery_thumbnail_number = 2

# def plot(imgs, **imshow_kwargs):
#     if not isinstance(imgs[0], list):
#         # Make a 2d grid even if there's just 1 row
#         imgs = [imgs]

#     num_rows = len(imgs)
#     num_cols = len(imgs[0])
#     _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
#     for row_idx, row in enumerate(imgs):
#         for col_idx, img in enumerate(row):
#             ax = axs[row_idx, col_idx]
#             img = F.to_pil_image(img.to("cpu"))
#             ax.imshow(np.asarray(img), **imshow_kwargs)
#             ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

#     plt.tight_layout()


#     plt.savefig("result_flow.png")


# weights = Raft_Large_Weights.DEFAULT
# transforms = weights.transforms()

# def preprocess(img1_batch, img2_batch):
#     img1_batch = F.resize(img1_batch, size=[520, 960])
#     img2_batch = F.resize(img2_batch, size=[520, 960])
#     return transforms(img1_batch, img2_batch)

# If you can, run this example on a GPU, it will be a lot faster.
# device = "cuda:7"
pil = [
        Image.open("/data/shared/ssvl/epic-kitchens50/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/train/P01/P01_01/frame_0000001000.jpg"),
        Image.open("/data/shared/ssvl/epic-kitchens50/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/rgb/train/P01/P01_01/frame_0000001005.jpg")    
    ]
img = [
        np.array(pil[0]),
        np.array(pil[1])
    ]

gray = [
    0.2989 * img[0][..., 0] + 0.5870 * img[0][..., 1] + 0.1140 * img[0][..., 2],
    0.2989 * img[1][..., 0] + 0.5870 * img[1][..., 1] + 0.1140 * img[1][..., 2],
]

# print(img[0].shape, img[1].shape)
# model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
# model = model.eval()

st_time = time.time()
flow = cv2.calcOpticalFlowFarneback(gray[0], 
                                   gray[1], 
                                   None, 0.5, 3, 10, 5, 3, 5, 1)

print(f"Duration: {time.time() - st_time}")

flow_rgb = flow_to_color(np.array(flow), convert_to_bgr=True)

cv2.imwrite("cv2_flow.png",np.concatenate([*img, flow_rgb], axis=1))