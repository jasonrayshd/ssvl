# import matplotlib.pyplot as plt

from datasets import build_pretraining_dataset
import torch
import torch.nn.functional as F

from timm.models import create_model
import modeling_pretrain
from pathlib import Path
import argparse
from config_utils import parse_yml, combine

import numpy as np
from einops import rearrange
from flow_vis import flow_to_color
from torchvision.utils import save_image
import cv2

def get_args():
    parser = argparse.ArgumentParser('VideoMAE pre-training script', add_help=False)

    # Dataset parameters

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # configuration file
    parser.add_argument('--config', default='none', type=str,
                        help='path to configuration file')

    parser.add_argument('--ckpt', default='none', type=str,
                        help='path to checkpoint')

    parser.add_argument('--overwrite', default='command-line', type=str,
                        help='overwrite args in command-line or configuration file')
    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")

    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth,

    )
    return model


@torch.no_grad()
def main(args):
    device = args.device
    model = get_model(args)

    checkpoints = []
    for ckpt in args.ckpt.split(","):
        checkpoints.append(torch.load(ckpt, map_location='cpu'))

    patch_size = model.encoder.rgb_patch_embed.patch_size

    # import sys
    # sys.exit(0)
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    dataset_train = build_pretraining_dataset(args, cache_manager=None)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
        # worker_init_fn=utils.seed_worker
    )

    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1, 1).repeat(1, 1, 224, 224).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1, 1).repeat(1, 1, 224, 224).to(device)

    model.to(device)
    model.eval()
    for i, batch in enumerate(data_loader_train):

        for j, weights in enumerate(checkpoints):
            model.load_state_dict(weights['model'], strict=True)

            frame, mask, flows = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            mask = mask.flatten(1).to(torch.bool).cpu()
            output = model(frame, flows, mask, all_token=True)
            

            # de-normalize input frames
            unnormed_frame = frame.squeeze() * std + mean
            # print(unnormed_frame[:, 0, :10, :10]*255)
            unnormed_frame_pre = unnormed_frame[:, 0, ...].cpu().numpy().transpose(1, 2, 0)*255
            unnormed_frame_pre = cv2.cvtColor(unnormed_frame_pre, cv2.COLOR_BGR2RGB)
            unnormed_frame_post = unnormed_frame[:, 1, ...].cpu().numpy().transpose(1, 2, 0)*255
            unnormed_frame_post = cv2.cvtColor(unnormed_frame_post, cv2.COLOR_BGR2RGB)

            rgb_hat, flow_hat = output
            B, N ,C = rgb_hat.shape
            rgb_rgb_hat = rgb_hat[:B//2, :, :]
            flow_rgb_hat = rgb_hat[B//2:, :, :]

            flow_flow_hat = flow_hat[:B//2, :, :]
            rgb_flow_hat = flow_hat[B//2:, :, :]

            masked_tokens = int(14*14*args.mask_ratio*8)

            unnormed_frame = frame.squeeze() * std + mean
            # print(torch.cat([mask,mask],dim=0).shape)
            rgb_rgb_hat_reshape = unpatchify_rgb(rgb_rgb_hat, mask, masked_tokens)
            print((rgb_rgb_hat_reshape-unnormed_frame).pow(2).mean())

            flow_rgb_hat_reshape = unpatchify_rgb(flow_rgb_hat, mask, masked_tokens)
            flow_flow_hat_reshape = unpatchify_flow(flow_flow_hat, mask, masked_tokens)
            rgb_flow_hat_reshape = unpatchify_flow(rgb_flow_hat, mask, masked_tokens)

            unnormed_frame = unnormed_frame.transpose(0, 1)
            flows_rgb = []
            flow_flow_hat_rgb = []
            rgb_flow_hat_rgb = []
            for t in range(8):
                flows_rgb.append(flow_to_color(flows.squeeze()[:, t, ...].cpu().numpy().transpose(1, 2, 0), convert_to_bgr=False))
                flow_flow_hat_rgb.append(flow_to_color(flow_flow_hat_reshape[:, t, ...].cpu().numpy().transpose(1, 2, 0), convert_to_bgr=False))
                rgb_flow_hat_rgb.append(flow_to_color(rgb_flow_hat_reshape[:, t, ...].cpu().numpy().transpose(1, 2, 0), convert_to_bgr=False))

            flows_rgb = torch.from_numpy(np.stack(flows_rgb, axis=0).transpose(0, 3, 1, 2))
            flow_flow_hat_rgb = torch.from_numpy(np.stack(flow_flow_hat_rgb, axis=0).transpose(0, 3, 1, 2))
            rgb_flow_hat_rgb = torch.from_numpy(np.stack(rgb_flow_hat_rgb, axis=0).transpose(0, 3, 1, 2))

            all_cat = torch.cat((flows_rgb/255, flow_flow_hat_rgb/255, rgb_flow_hat_rgb/255, unnormed_frame.cpu(), rgb_rgb_hat_reshape.cpu().transpose(0, 1), flow_rgb_hat_reshape.cpu().transpose(0, 1)), dim=0)
            save_image(all_cat, f"./log/flow_vis_{i}.png")

           


def warp_flow(curImg, flows):
    print(curImg.shape, flows.shape)
    h, w = flows.shape[:2]
    flows[:,:,0] += np.arange(w).astype(np.uint8)
    flows[:,:,1] += np.arange(h)[:,np.newaxis].astype(np.uint8)
    prevImg = cv2.remap(curImg.astype(np.float32), flows.astype(np.float32), None, cv2.INTER_LINEAR)

    return prevImg


def unpatchify_rgb(rgb_raw, mask, masked_tokens):
    """
        rgb_raw: torch.Tensor, (1, N, C)
        mask: torch.Tensor, (1, N)
        masked_tokens: int, number of masked tokens

        Return
        ---
        rgb_hat_reshape: reconstructed RGB image, (3, T, H, W)
    """
    # mask = mask.squeeze() # N
    img = torch.zeros_like(rgb_raw) # 1, N ,C
    img[mask] = rgb_raw[:, -masked_tokens:]
    img[~mask] = rgb_raw[:, :-masked_tokens]
    img = img.squeeze()

    # rgb_hat = rearrange(img, 'n (p c) -> n p c', c=3)
    rgb_hat_reshape = rearrange(img, '(t h w) (p0 p1 p2 c) -> c (t p0) (h p1) (w p2)', p0=2, p1=16, p2=16, h=14, w=14)

    return rgb_hat_reshape

def unpatchify_flow(flow_raw, mask, masked_tokens):
    """
        flow_raw: torch.Tensor, (1, N, C)
        mask: torch.Tensor, (1, N)
        masked_tokens: int, number of masked tokens

        Return
        ---
        flow_hat_reshape: reconstructed flow image, (2, T, H, W)
    """
    flow_hat = torch.zeros_like(flow_raw)
    flow_hat[mask] = flow_raw[:, -masked_tokens:]
    flow_hat[~mask] = flow_raw[:, :-masked_tokens]
    flow_hat = flow_hat.squeeze()
    flow_hat_reshape = rearrange(flow_hat, "(t h w) (p1 p2 c) -> c t (h p1) (w p2)", c=2, t=8, h=14, w=14, p1=16, p2=16)

    return flow_hat_reshape


if __name__ == "__main__":
    opts = get_args()
    config = parse_yml(opts.config)
    if config is not None:
        opts = combine(opts, config)

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)

    main(opts)
