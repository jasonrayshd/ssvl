import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import numpy as np
import cv2

class FlowMSELoss(nn.Module):

    def __init__(self, patch_size, tublet_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.tublet_size = tublet_size

        self.recons_loss = nn.MSELoss()

    def forward(self, output, raw_target, mask):
        """

            Parameters
            ---

            output: flow prediction of VideoMAE
            
            e.g. when tublet size is 2, and embeded with 16x16 window
                shape (B, num_frames/2 x 14 x 14, 3x16x16)
                8 flow images in total

            target: input frames of VideoMAE, which might be normalized or not

            e.g. shape (B, channels, num_frames, height, width)

        """
        B, C, T, H, W = raw_target.shape
        flows = rearrange(output, 'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)', t=T/self.tublet_size, h=H/self.patch_size, w=W/self.patch_size,p0=self.tublet_size, p1=self.patch_size, p2=self.patch_size)
        recons_from_flow = self.from_flow(flows, raw_target) # b (t h w) (p0 p1 p2 c)

        target = rearrange(raw_target, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=self.tublet_size, p1=self.patch_size, p2=self.patch_size)
        loss = self.recons_loss(recons_from_flow, target[mask])
        return loss

    def from_flow(self, flows, images):
        # reconstruct images given flows
        # TODO accelearte with cython
        B, C, T, H, W = flows.shape
        recons_from_flow = []
        for b in range(B):
            recons = []
            for t in range(T):
                # number of flows
                recons.append(self.warp(flows[b, :, t:t+2, :, :], images[b, :, (t+1)*self.tublet_size-1, :, :],))
            recons_from_flow.append(recons)
        return torch.stack(recons_from_flow, dim=0)

    def warp(flow, prev_img):
        c, _, h, w = flow.shape
        flow[:, 0, :,:] += np.arange(w)
        flow[:, 1, :,:] += np.arange(h)[:,np.newaxis]

        next_img = cv2.remap(prev_img, flow, None, cv2.INTER_LINEAR)

        return next_img


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None,

                    use_preprocessed_flow = False, # use preprocessed flow images or not
                    flow_pretrain = False, # reconstruct input based on predicted flow images or not
                    ):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    flow_based_recons = (flow_pretrain and not use_preprocessed_flow)

    if flow_based_recons:
        # if predicting flow images but do not use preprocessed flow images
        loss_func = FlowMSELoss()
    else:
        loss_func = nn.MSELoss()

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bool_masked_pos = batch[0], batch[1]
        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        # use given target or simply reconstruct input video
        if not use_preprocessed_flow:
            with torch.no_grad():
                # calculate the predict label
                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
                std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]

                unnorm_videos = videos * std + mean  # in [0, 1]

                if normlize_target:
                    videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                    videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                        ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                    # we find that the mean is about 0.48 and standard deviation is about 0.08.
                    if flow_based_recons:
                        B, C, T, H, W = unnorm_videos.shape
                        labels = rearrange(videos_norm, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', t = T/2, h=H/patch_size, w=W/patch_size,p0=2, p1=patch_size, p2=patch_size)
                    else:
                        videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
                        B, _, C = videos_patch.shape
                        labels = videos_patch[bool_masked_pos].reshape(B, -1, C)
                else:
                    if flow_based_recons:
                        labels = unnorm_videos
                    else:
                        videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)
                        B, _, C = videos_patch.shape
                        labels = videos_patch[bool_masked_pos].reshape(B, -1, C)
        else:
            # target will be processed in dataset code
            # reconstruct entire given flow images
            labels = batch[2] # bs, 2, flow nubmers, h, w
            B, _, N, H, W = labels.shape
            _, _, T, H, W = videos.shape
            assert T%N == 0, f"Number of flows:{T} to be predicted should be divisible by number of frames:{N}"
            # print(labels.shape)

            labels = rearrange(labels, 'b c t (h p1) (w p2) -> b (t h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

            tublet_size = 2
            bool_masked_pos_label = rearrange(bool_masked_pos, "b (t h w) -> b t h w", t=T//tublet_size, h=H//patch_size,w=W//patch_size)
            bool_masked_pos_label = bool_masked_pos_label.repeat(1, N//(T//tublet_size), 1, 1)
            bool_masked_pos_label = bool_masked_pos_label.reshape(B, -1)
            # print(bool_masked_pos_label.shape)
            # print(labels.shape)

            labels = labels[bool_masked_pos_label]
            labels = rearrange(labels, '(b t n) d -> b (t n) d', b=B, t=N//tublet_size)

            labels = labels.to(device, non_blocking=True)
            # print(f"final label: {labels.shape}")
 
        with torch.cuda.amp.autocast():
            outputs = model(videos, bool_masked_pos)
            if flow_based_recons:
                loss =  loss_func(input=outputs, target=labels, mask = bool_masked_pos)
            else:
                loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
