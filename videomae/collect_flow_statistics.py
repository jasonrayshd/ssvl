"""
Edited by Jiachen
This script is used to collect the pixel value distribution of flow images

"""

import os
import torch
import utils
import argparse
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm

from config_utils import parse_yml, combine
from datasets import build_pretraining_dataset


class CustomFlowDataset(Dataset):

    def __init__(self, path):
        # super.__init__()

        self.path = path
        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.uflow, self.vflow = self.make_path_lst(path)

    def make_path_lst(self, path):
        ulst = []
        vlst = []
        participants = os.listdir(path)

        print("start making path list")
        pbar = tqdm(participants)
        for p in pbar:

            videos = [file for file in os.listdir(os.path.join(path, p)) if not "tar" in file]

            for video in videos:
                vid_path = os.path.join(path, p, video)

                uframes = os.listdir(os.path.join(vid_path, "u"))
                vframes = os.listdir(os.path.join(vid_path, "v"))

                ulst.extend([os.path.join(vid_path, "u", frame) for frame in uframes])
                vlst.extend([os.path.join(vid_path, "v", frame) for frame in vframes])

                pbar.set_postfix_str(video)

        return ulst, vlst


    def __getitem__(self, idx):

        upil = self.trans( Image.open(self.uflow[idx]) )
        vpil = self.trans( Image.open(self.vflow[idx]) )
        return torch.stack([upil, vpil], dim=0)


    def __len__(self):

        return len(self.uflow)


@torch.no_grad()
def main():

    path = "/data/shared/ssvl/epic-kitchens50/3h91syskeag572hl6tvuovwv4d/frames_rgb_flow/flow/train/"

    dataset = CustomFlowDataset(path)
    data_loader_train = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
    )

    flow_hist = None
    pbar = tqdm(data_loader_train)
    print("start collecting flow pixel value statistics")
    for batch in pbar:

        flows = batch[0]*255
        b, c, h, w = flows.shape
        num = b*c*h*w

        if flow_hist is None:
            flow_hist = torch.histc(flows, bins=256, min=0, max=255) / num
        else:
            flow_hist += torch.histc(flows, bins=256, min=0, max=255) / num





if __name__ == "__main__":
    main()