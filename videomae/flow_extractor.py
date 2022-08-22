"""
Edited by jiachen

flow image extracter based on multiprocessing.manager

"""
import os
import torch
from torchvision import transforms
from PIL import Image
from mmflow.apis import init_model
from mmflow.datasets import visualize_flow, write_flow


class flowExtractor(object):
    def __init__(self, 
        # default file saved in docker image
            device="",
            config_file = '/root/.cache/mim/pwcnet_ft_4x1_300k_sintel_final_384x768.py',
            checkpoint_file = '/root/.cache/mim/pwcnet_ft_4x1_300k_sintel_final_384x768.pth',
        ):

        self.device = device if device != "" else "cuda:" + str(torch.cuda.current_device())
        # build the model from a config file and a checkpoint file
        print(f"Current device: {self.device} manager process:{os.getpid()} parent process:{os.getppid()}")
        self.model = init_model(config_file, checkpoint_file, device=self.device)

    @torch.no_grad()
    def ext(self, x):
        """
            x: torch.Tensor, T, C, H, W
        """
        # extract flow images given frames

        flows = self.model(x.to(self.device))
        del x

        return flows


if __name__ == "__main__":
    import time
    trans = transforms.Compose([
        transforms.CenterCrop((256, 256)) ,
        transforms.ToTensor()
    ])
    frames = [
        trans(Image.open(f"/data/epic-kitchens55/frames_rgb_flow/rgb/train/P01/P01_01/frame_000000{i}.jpg"))
           for i in range(1000, 1010)
    ]

    frames = torch.stack(frames, dim=0)
    flowExt = flowExtractor()

    st_time = time.time()
    shifted_frames = torch.roll(frames, -1, 0)

    frames = torch.cat((frames, shifted_frames), dim=1)

    print(frames.shape)
    
    flow_lst_dict = flowExt.ext(frames)

    print(f"duration: {time.time()-st_time}")

    # print(len(result))
    # print(result[0]["flow"].shape)

    flow_map = visualize_flow(flow_lst_dict[0]["flow"], save_file='flow_map.png')