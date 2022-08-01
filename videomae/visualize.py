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

    if not args.ts_pretrain:
        model = create_model(
            args.model,
            pretrained=False,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            decoder_depth=args.decoder_depth
        )
    else:
        model = create_model(
            args.model,
            pretrained=False,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            decoder_depth=args.decoder_depth,

            fuse_scheme = args.fuse_scheme,
            tokenizer_backbone = args.tokenizer_backbone,
        )
    return model


@torch.no_grad()
def main(args):
    device = args.device
    model = get_model(args)

    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)

    if args.ts_pretrain:
        patch_size = model.rgb_encoder.patch_embed.patch_size
    else:
        patch_size = model.encoder.patch_embed.patch_size

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
        if args.ts_pretrain:
            frame, mask, flows = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            mask = mask.flatten(1).to(torch.bool).cpu()
            output = model(frame, flows, mask, all_token=True)
        elif args.flow_mode != "":
            frame, mask, flows = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            mask = mask.flatten(1).to(torch.bool).cpu()
            output = model(frame, mask, all_token=True)
        else:
            frame, mask = batch[0].to(device), batch[1].to(device)
            mask = mask.flatten(1).to(torch.bool).cpu()
            output = model(frame, mask, all_token=True)

        # de-normalize input frames
        unnormed_frame = frame.squeeze() * std + mean
        # print(unnormed_frame[:, 0, :10, :10]*255)
        unnormed_frame_pre = unnormed_frame[:, 0, ...].cpu().numpy().transpose(1, 2, 0)*255
        unnormed_frame_pre = cv2.cvtColor(unnormed_frame_pre, cv2.COLOR_BGR2RGB)
        unnormed_frame_post = unnormed_frame[:, 1, ...].cpu().numpy().transpose(1, 2, 0)*255
        unnormed_frame_post = cv2.cvtColor(unnormed_frame_post, cv2.COLOR_BGR2RGB)

        border = np.zeros((224, 4, 3))
        if not args.ts_pretrain and args.flow_mode == "local":
            flow_hat = output.squeeze()
            flows_reshape = rearrange(flows, "b c t (h p0) (w p1) -> b (h w t) (c p0 p1)", p0=16, p1=16).squeeze()
            flow_hat_reshape = rearrange(flow_hat, "(h w t) (c p0 p1) -> c t (h p0) (w p1)", c=2, t=8, h=14, w=14, p0=16, p1=16)
    
            error = F.mse_loss(flow_hat, flows_reshape).cpu().item()

            mask_reshape = rearrange(mask, "b (t h w) -> b t h w", t=8, h=14, w=14).squeeze()
            mask_combine = mask_reshape[0]
            mask_combine = np.repeat( np.repeat(mask_combine, 16, axis=1), 16, axis=0)
            mask_combine = np.expand_dims(mask_combine, axis=2)

            # convert ground truth and reconstructed flows to rgb
            flows_rgb = flow_to_color(flows.squeeze()[:, 0, ...].cpu().numpy().transpose(1, 2, 0)*255, convert_to_bgr=True)
            flow_hat_rgb = flow_to_color(flow_hat_reshape[:, 0, ...].cpu().numpy().transpose(1, 2, 0)*255, convert_to_bgr=True)

            # convert flow for warping frame
            # from torch.Tensor [0, 1] to numpy.ndarray [0, 255]
            # flow_warped = flows.squeeze()[:, 0, ...].cpu().numpy().transpose(1, 2, 0)*255

            # warped_post = warp_flow(unnormed_frame_pre, flow_warped)
            # warped_post = np.array(warped_post)
            # print(warped_post.shape)
            # print( np.linalg.norm(np.array(warped_post)- unnormed_frame_post) )

            combined = (1-mask_combine)* flows_rgb+ mask_combine*flow_hat_rgb

            
            all_concat = np.concatenate((unnormed_frame_pre, border, unnormed_frame_post, flows_rgb, border, combined, border, flow_hat_rgb), axis=1)
            all_concat = np.concatenate((all_concat, np.zeros((20, all_concat.shape[1], 3))), axis=0)

            BLACK = (255,255,255)
            font = cv2.FONT_HERSHEY_PLAIN 
            font_size = 1
            font_color = BLACK
            font_thickness = 1
            text = 'error: {:.4f}'.format(error)
            x,y = 20,240
            all_concat_wtext = cv2.putText(all_concat, text, (x,y), font, font_size, font_color, font_thickness, cv2.LINE_AA)

            cv2.imwrite(f"./log/flow_vis_{i}.png", all_concat_wtext)

        elif args.ts_pretrain:
            flows_rgb = flow_to_color(flows.squeeze()[:, 0, ...].cpu().numpy().transpose(1, 2, 0)*255, convert_to_bgr=True)

            rgb_rgb_hat, rgb_flow_hat, flow_rgb_hat, flow_flow_hat, rgb_vis, flow_vis, rgb_token, flow_token = output

            rgb_rgb_reshape = rearrange(rgb_rgb_hat.squeeze(), "(h w t) (c p0 p1 p2) -> c (t p0) (h p1) (w p2)", c=3, t=8, h=14, w=14, p0=2, p1=16, p2=16)   
            rgb_rgb_unnormed =  rgb_rgb_reshape * std + mean
            rgb_rgb_unnormed_pre = rgb_rgb_unnormed[:, 0, ...].cpu().numpy().transpose(1, 2, 0)*255
            rgb_rgb_unnormed_post = rgb_rgb_unnormed[:, 1, ...].cpu().numpy().transpose(1, 2, 0)*255

            flow_flow_reshape = rearrange(flow_flow_hat.squeeze(), "(h w t) (c p0 p1) -> c t (h p0) (w p1)", c=2, t=8, h=14, w=14, p0=16, p1=16)
            flow_flow_hat = flow_to_color(flow_flow_reshape[:, 0, ...].cpu().numpy().transpose(1, 2, 0)*255, convert_to_bgr=True)

            rgb_flow_reshape = rearrange(rgb_flow_hat.squeeze(), "(h w t) (c p0 p1) -> c t (h p0) (w p1)", c=2, t=8, h=14, w=14, p0=16, p1=16)
            rgb_flow_hat = flow_to_color(rgb_flow_reshape[:, 0, ...].cpu().numpy().transpose(1, 2, 0)*255, convert_to_bgr=True)

            flow_rgb_reshape = rearrange(flow_rgb_hat.squeeze(), "(h w t) (c p0 p1 p2) -> c (t p0) (h p1) (w p2)", c=3, t=8, h=14, w=14, p0=2, p1=16, p2=16)
            flow_rgb_unnormed = flow_rgb_reshape * std + mean
            flow_rgb_unnormed_pre = flow_rgb_unnormed[:, 0, ...].cpu().numpy().transpose(1, 2, 0)*255
            flow_rgb_unnormed_post = flow_rgb_unnormed[:, 1, ...].cpu().numpy().transpose(1, 2, 0)*255

            all_concat_GT = np.concatenate((unnormed_frame_pre, border, unnormed_frame_post, border, flows_rgb), axis=1)
            all_concat_recons = np.concatenate((rgb_rgb_unnormed_pre, border, rgb_rgb_unnormed_post, border, flow_flow_hat), axis=1)
            all_concat_crossmodality = np.concatenate((flow_rgb_unnormed_pre, border, flow_rgb_unnormed_post, border, rgb_flow_hat), axis=1)

            all_concat = np.concatenate((all_concat_GT, all_concat_recons, all_concat_crossmodality), axis=0)

            cv2.imwrite(f"./log/flow_vis_{i}.png", all_concat)
        elif args.flow_mode == "":
            unnormed_frame = frame.squeeze() * std + mean
            unnormed_frame = unnormed_frame.transpose(0, 1)
            rgb_hat = output.squeeze()
            rgb_hat_reshape = rearrange(rgb_hat, "(h w t) (c p0 p1 p2) -> (t p0) c (h p1) (w p2)", c=3, t=8, h=14, w=14, p0=2, p1=16, p2=16)
            all_concat = torch.cat((unnormed_frame, rgb_hat_reshape), axis=0)
            save_image(all_concat, f"./log/flow_vis_{i}.png")
            # unnormed_rgb_hat = rgb_hat_reshape * std + mean
            # rgb_hat_reshape = rgb_hat_reshape[:, 0, ...].cpu().numpy().transpose(1, 2, 0)*255
            # cv2.imwrite(f"./log/flow_vis_{i}.png", all_concat)
            # print(f"saved ./log/flow_vis_{i}.png")

        elif args.flow_mode == "online":
            pass


def warp_flow(curImg, flows):
    print(curImg.shape, flows.shape)
    h, w = flows.shape[:2]
    flows[:,:,0] += np.arange(w).astype(np.uint8)
    flows[:,:,1] += np.arange(h)[:,np.newaxis].astype(np.uint8)
    prevImg = cv2.remap(curImg.astype(np.float32), flows.astype(np.float32), None, cv2.INTER_LINEAR)

    return prevImg


if __name__ == "__main__":
    opts = get_args()
    config = parse_yml(opts.config)
    if config is not None:
        opts = combine(opts, config)

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)

    main(opts)
