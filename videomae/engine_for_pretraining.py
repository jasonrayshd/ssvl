import os
import sys
import cv2
import math
import random
import numpy as np

from typing import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torchvision.utils import save_image


# class FlowMSELoss(nn.Module):

#     def __init__(self, patch_size, tublet_size=2):
#         super().__init__()
#         self.patch_size = patch_size
#         self.tublet_size = tublet_size

#         self.recons_loss = nn.MSELoss()

#     def forward(self, output, raw_target, mask):
#         """

#             Parameters
#             ---

#             output: flow prediction of VideoMAE
            
#             e.g. when tublet size is 2, and embeded with 16x16 window
#                 shape (B, num_frames/2 x 14 x 14, 3x16x16)
#                 8 flow images in total

#             target: input frames of VideoMAE, which might be normalized or not

#             e.g. shape (B, channels, num_frames, height, width)

#         """
#         B, C, T, H, W = raw_target.shape
#         flows = rearrange(output, 'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)', t=T/self.tublet_size, h=H/self.patch_size, w=W/self.patch_size,p0=self.tublet_size, p1=self.patch_size, p2=self.patch_size)
#         recons_from_flow = self.from_flow(flows, raw_target) # b (t h w) (p0 p1 p2 c)

#         target = rearrange(raw_target, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=self.tublet_size, p1=self.patch_size, p2=self.patch_size)
#         loss = self.recons_loss(recons_from_flow, target[mask])
#         return loss

#     def from_flow(self, flows, images):
#         # reconstruct images given flows
#         # TODO accelearte with cython
#         B, C, T, H, W = flows.shape
#         recons_from_flow = []
#         for b in range(B):
#             recons = []
#             for t in range(T):
#                 # number of flows
#                 recons.append(self.warp(flows[b, :, t:t+2, :, :], images[b, :, (t+1)*self.tublet_size-1, :, :],))
#             recons_from_flow.append(recons)
#         return torch.stack(recons_from_flow, dim=0)

#     def warp(flow, prev_img):
#         c, _, h, w = flow.shape
#         flow[:, 0, :,:] += np.arange(w)
#         flow[:, 1, :,:] += np.arange(h)[:,np.newaxis]

#         next_img = cv2.remap(prev_img, flow, None, cv2.INTER_LINEAR)

#         return next_img


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None,

                    train_wo_amp = False,
                    predict_preprocessed_flow = False,

                    ):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # flow_based_recons = (predict_flow_pretrain and not predict_preprocessed_flow)

    # if flow_based_recons:
    #     # if predicting flow images but do not use preprocessed flow images
    #     loss_func = FlowMSELoss()
    # else:
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

        if len(videos.shape) == 6: # repeated sampling is used
            B, Repeat, C, T, H, W = videos.shape
            videos = videos.reshape(-1, *videos.shape[2:])
            bool_masked_pos = bool_masked_pos.reshape(-1, *bool_masked_pos.shape[2: ])

        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        # use given target or simply reconstruct input video
        if not predict_preprocessed_flow:
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

                    videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
                    B, _, C = videos_patch.shape
                    labels = videos_patch[bool_masked_pos].reshape(B, -1, C)
                else:

                    videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)
                    B, _, C = videos_patch.shape
                    labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

        else:
            # target will be processed in dataset code
            # reconstruct entire given flow images
            labels = batch[2] # bs, 2, flow nubmers, h, w

            if len(labels.shape) == 6: # repeated sampling is used
                labels = labels.reshape(-1, *labels.shape[2:])

            B, _, N, H, W = labels.shape
            _, _, T, H, W = videos.shape
            assert T%N == 0, f"Number of flows:{T} to be predicted should be divisible by number of frames:{N}"
            # print(labels.shape)
            # print(f"label shape: {labels.shape}")
            labels = rearrange(labels, 'b c t (h p1) (w p2) -> b (t h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

            tublet_size = 2
            bool_masked_pos_label = rearrange(bool_masked_pos, "b (t h w) -> b t h w", t=T//tublet_size, h=H//patch_size,w=W//patch_size)
            bool_masked_pos_label = bool_masked_pos_label.repeat(1, N//(T//tublet_size), 1, 1)
            bool_masked_pos_label = bool_masked_pos_label.reshape(B, -1)
            # print(bool_masked_pos_label.shape)
            # print(labels.shape)

            labels = labels[bool_masked_pos_label]
            labels = rearrange(labels, '(b n) d -> b n d', b=B)

            labels = labels.to(device, non_blocking=True)
            # print(f"final label: {labels.shape}")
 
        with torch.cuda.amp.autocast(enabled=not train_wo_amp):
            outputs = model(videos, bool_masked_pos)

            loss = loss_func(input=outputs, target=labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        if not train_wo_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order)
            loss_scale_value = loss_scaler.state_dict()["scale"]
        else:
            loss.backward()
            optimizer.step()

            grad_norm = 0
            loss_scale_value = 0

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


class TwoStreamVitLoss(nn.Module):

    def __init__(self, ctr="easy", lamb = [1, 1, 1, 1, 1, 1], tau=0.8):
        super().__init__()
        self.ctr_type = ctr
        self.lamb = lamb # weights for each loss component
        self.tau = tau# temperature

    # referred to MoCov3-pytorch
    # https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py
    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output

    def ctr(self, q, k):
        if self.ctr_type == "easy":
            return self.easy_ctr(q, k)
        elif self.ctr_type == "hard":
            return self.hard_ctr(q, k)
        elif self.ctr_type == "hard2":
            return self.hard_ctr_v2(q, k)

    # referred to VAAT
    # https://github.com/google-research/google-research/blob/master/vatt/utils/train/objectives.py
    def easy_ctr(self, q, k):
        # normalize embeddings
        q = F.normalize(q, p=2, dim=2) # B, N1, C
        k = F.normalize(k, p=2, dim=2) # B, N2, C

        # gather cross rank negative
        k_all = self.concat_all_gather(k)

        q_vs_k = torch.einsum("bmd,cnd->bcmn", q, k) # (B, B, N1, N2)
        q_vs_k = q_vs_k.flatten(2)

        q_vs_kall = torch.einsum("bmd,cnd->bcmn", q, k_all) # (B, B_all, N1, N2)
        q_vs_kall = q_vs_kall.flatten(1)

        pos_sim = torch.einsum("bbn->bn", q_vs_k)

        # NOTE: [07.24 by jiachen]
        # when training in half-precision (16-bit), intermediate result of logsumexp
        # will always exceed the range of 16-bit floating point number
        # Simple solution: convert input tensor to 32-bit floating point before compute logsumexp
        # and back to 16-bit after
        logsumexp_pos = torch.logsumexp(pos_sim.float()/self.tau, dim=1).half()
        logsumexp_all = torch.logsumexp(q_vs_kall.float()/self.tau, dim=1).half()

        nce = logsumexp_all - logsumexp_pos

        # according to implementation of VAAT
        # if none of the samples are valid, the logsumexp could be NaN, hence
        # we manually set them to zero
        nce = torch.where(torch.isnan(nce), torch.zeros(nce.shape).half().to(nce.device), nce).mean()

        return nce

    def hard_ctr(self, q, k):
        # normalize embeddings
        q = F.normalize(q, p=2, dim=2) # B, N1, C
        k = F.normalize(k, p=2, dim=2) # B, N2, C

        # gather cross rank negative
        k_all = self.concat_all_gather(k)
        q_all = self.concat_all_gather(q)
        B_all, N_all, C_all = q_all.shape

        # positive pairs accross modality
        B,N,C = q.shape
        q_vs_k = torch.einsum("bmd,cnd->bcmn", q, k) # (B, B, N1, N2)
        q_vs_k = q_vs_k.flatten(2)

        q_vs_k = torch.einsum("bbn -> bn", q_vs_k)
        q_vs_k = q_vs_k.reshape(B, N, N)
        pos_sim = torch.einsum("bnn -> bn", q_vs_k)

        # q_vs_q =  torch.einsum("bmd,cnd -> bcmn", q, q)
        # q_vs_q = q_vs_q.flatten(2)
        # q_vs_q =  torch.einsum("bbn -> bn", q_vs_q)
        # intra_neg_sim = q_vs_q[:, :-1].reshape(B, N-1, N+1)[..., 1:].flatten(1) # (B, N3)

        # negative pairs within the same modality but across batch
        rank = int( os.environ["RANK"] )
        q_vs_q_all =  torch.einsum("bmd,cnd -> bcmn", q, q_all)
        q_vs_q_all = [ torch.cat((q_vs_q_all[i, :rank*B + i], q_vs_q_all[i, rank*B + i + 1:]), dim=0)  for i in range(B)]
        q_vs_q_all = torch.stack(q_vs_q_all, dim=0) # (B, B_all-1, N1, N2)
        q_vs_q_all = q_vs_q_all.flatten(1)

        # negative pairs across modality
        q_vs_kall = torch.einsum("bmd,cnd->bcmn", q, k_all) # (B, B_all, N1, N2)
        q_vs_kall = q_vs_kall.flatten(1)

        all_sim = torch.cat((q_vs_q_all, q_vs_kall), dim=1)

        # NOTE: [07.24 by jiachen]
        # when training in half-precision (16-bit), intermediate result of logsumexp
        # will always exceed the range of 16-bit floating point number
        # Simple solution: convert input tensor to 32-bit floating point before compute logsumexp
        # and back to 16-bit after
        logsumexp_pos = torch.logsumexp(pos_sim.float()/self.tau, dim=1).half()
        logsumexp_all = torch.logsumexp(all_sim.float()/self.tau, dim=1).half()

        # logsumexp_intra_neg = torch.logsumexp(intra_neg_sim.float()/self.tau, dim=1).half()

        nce = logsumexp_all  - logsumexp_pos

        # according to implementation of VAAT
        # if none of the samples are valid, the logsumexp could be NaN, hence
        # we manually set them to zero
        nce = torch.where(torch.isnan(nce), torch.zeros(nce.shape).half().to(nce.device), nce).mean()

        return nce

    def hard_ctr_v2(self, q, k):
        # normalize embeddings
        q = F.normalize(q, p=2, dim=2) # B, N1, C
        k = F.normalize(k, p=2, dim=2) # B, N2, C

        # gather cross rank negative
        k_all = self.concat_all_gather(k)
        q_all = self.concat_all_gather(q)
        B_all, N_all, C_all = q_all.shape

        # positive pairs accross modality
        B,N,C = q.shape
        q_vs_k = torch.einsum("bmd,cnd->bcmn", q, k) # (B, B, N1, N2)
        q_vs_k = q_vs_k.flatten(2)

        q_vs_k = torch.einsum("bbn -> bn", q_vs_k)
        q_vs_k = q_vs_k.reshape(B, N, N)
        pos_sim = torch.einsum("bnn -> bn", q_vs_k)

        # negative pairs within same modality
        q_vs_q_all =  torch.einsum("bmd,cnd -> bcmn", q, q_all)
        q_vs_q_all = q_vs_q_all.flatten(2)
        q_vs_q_all = q_vs_q_all[:, :, :-1].reshape(B, B_all, N-1, N+1)[..., 1:].flatten(1) # (B, N3)

        # negative pairs within the same modality but across batch
        # rank = int( os.environ["RANK"] )
        # q_vs_q_all =  torch.einsum("bmd,cnd -> bcmn", q, q_all)
        # q_vs_q_all = [ torch.cat((q_vs_q_all[i, :rank*B + i], q_vs_q_all[i, rank*B + i + 1:]), dim=0)  for i in range(B)]
        # q_vs_q_all = torch.stack(q_vs_q_all, dim=0) # (B, B_all-1, N1, N2)
        # q_vs_q_all = q_vs_q_all.flatten(1)

        # negative pairs across modality
        q_vs_kall = torch.einsum("bmd,cnd->bcmn", q, k_all) # (B, B_all, N1, N2)
        q_vs_kall = q_vs_kall.flatten(1)

        all_sim = torch.cat((q_vs_q_all, q_vs_kall), dim=1)

        # NOTE: [07.24 by jiachen]
        # when training in half-precision (16-bit), intermediate result of logsumexp
        # will always exceed the range of 16-bit floating point number
        # Simple solution: convert input tensor to 32-bit floating point before compute logsumexp
        # and back to 16-bit after
        logsumexp_pos = torch.logsumexp(pos_sim.float()/self.tau, dim=1).half()
        logsumexp_all = torch.logsumexp(all_sim.float()/self.tau, dim=1).half()

        # logsumexp_intra_neg = torch.logsumexp(intra_neg_sim.float()/self.tau, dim=1).half()

        nce = logsumexp_all  - logsumexp_pos

        # according to implementation of VAAT
        # if none of the samples are valid, the logsumexp could be NaN, hence
        # we manually set them to zero
        nce = torch.where(torch.isnan(nce), torch.zeros(nce.shape).half().to(nce.device), nce).mean()

        return nce

    def _ctr_raw(self, feat, token):
        # feat, token: (B, N, C)
        logits = torch.mm(feat.flatten(1), token.flatten(1).t())
        B, _ = logits.shape
        labels = torch.tensor(list(range(B))).long().to(feat.device)
        loss = F.cross_entropy(logits/self.tau, labels)

        return 2*self.tau*loss

    def recons(self, predict, target, weight=None):
        if weight is None:
            loss = F.mse_loss(predict, target)
        else:
            loss = weight *(predict - target)**2
            loss = loss.mean()

        return loss

    def forward(self, output_lst, target_lst, weight=None):
        """
            weight: torch.Tensor, weight for reconstruction loss, the shape should be equal to reconstruction target

        """

        if len(output_lst) == 8:
            rgb_rgb_hat, rgb_flow_hat, flow_rgb_hat, flow_flow_hat = output_lst[:4]
            rgb_vis, flow_vis, rgb_token, flow_token = output_lst[4:]
        else:
            rgb_rgb_hat, rgb_flow_hat, flow_flow_hat = output_lst[:3]
            flow_rgb_hat = None
            rgb_vis, flow_vis, rgb_token, flow_token = output_lst[3:]

        rgb_target, flow_target = target_lst


        l_rgb_recons = self.recons(rgb_rgb_hat, rgb_target)
        l_flow_recons = self.recons(flow_flow_hat, flow_target)
        l_rgb_flow_recons  = self.recons(rgb_flow_hat, flow_target)
        if flow_rgb_hat is not None:
            l_flow_rgb_recons = self.recons(flow_rgb_hat, rgb_target, weight=weight)
        else:
            l_flow_rgb_recons = 0

        l_rgb_contrast = self.ctr(rgb_vis, flow_token)
        l_flow_contrast = self.ctr(flow_vis, rgb_token)

        if flow_rgb_hat is not None:
            loss =  self.lamb[0]*l_rgb_recons + self.lamb[1]*l_flow_recons + \
                    self.lamb[2]*l_rgb_flow_recons + self.lamb[3]*l_flow_rgb_recons + \
                    self.lamb[4]*l_rgb_contrast + self.lamb[5]*l_flow_contrast

            return {
                "sum": loss,
                "rgb_recons": l_rgb_recons.cpu().item(),
                "flow_recons": l_flow_recons.cpu().item(),
                "rgb_flow_recons": l_rgb_flow_recons.cpu().item(),
                "flow_rgb_recons": l_flow_rgb_recons.cpu().item(),
                "rgb_contrast": l_rgb_contrast.cpu().item(),
                "flow_contrast": l_flow_contrast.cpu().item(),
            }
        else:
            loss =  self.lamb[0]*l_rgb_recons + self.lamb[1]*l_flow_recons + \
                    self.lamb[2]*l_rgb_flow_recons + \
                    self.lamb[4]*l_rgb_contrast + self.lamb[5]*l_flow_contrast
            return {
                "sum": loss,
                "rgb_recons": l_rgb_recons.cpu().item(),
                "flow_recons": l_flow_recons.cpu().item(),
                "rgb_flow_recons": l_rgb_flow_recons.cpu().item(),
                # "flow_rgb_recons": l_flow_rgb_recons.cpu().item(),
                "rgb_contrast": l_rgb_contrast.cpu().item(),
                "flow_contrast": l_flow_contrast.cpu().item(),
            }

def train_tsvit_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None,  flow_encoder_lr_schedule_values = None,
                    wd_schedule_values=None,
                    flow_optimizer = None,


                    weighted_flow2rgb_recons = False,
                    ctr="easy",
                    tau = 0.8,
                    lamb = [0.25, 0.25, 0.25, 0.25],                    
                    ):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = TwoStreamVitLoss(ctr=ctr, lamb=lamb, tau=tau)

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration

        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        if flow_optimizer is not None:
            for i, param_group in enumerate(flow_optimizer.param_groups):
                param_group["lr"] = flow_encoder_lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bool_masked_pos, flows = batch[0], batch[1], batch[2]

        if len(videos.shape) == 6: # repeated sampling is used
            videos = videos.reshape(-1, *videos.shape[2:])
            bool_masked_pos = bool_masked_pos.reshape(-1, *bool_masked_pos.shape[2:])
            flows = flows.reshape(-1, *flows.shape[2:])

        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        # use given target or simply reconstruct input video

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]

            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)

                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
                B, _, C = videos_patch.shape
                rgb_target = videos_patch[bool_masked_pos].reshape(B, -1, C)
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)
                B, _, C = videos_patch.shape
                rgb_target = videos_patch[bool_masked_pos].reshape(B, -1, C)

        # target will be processed in dataset code
        # reconstruct entire given flow images
        # flows = batch[2] # bs, 2, flow nubmers, h, w

        B, _, N, H, W = flows.shape
        _, _, T, H, W = videos.shape
        assert T%N == 0, f"Number of flows:{T} to be predicted should be divisible by number of frames:{N}"
        # print(flows.shape)

        flow_target = rearrange(flows, 'b c t (h p1) (w p2) -> b (t h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

        tublet_size = 2
        bool_masked_pos_label = rearrange(bool_masked_pos, "b (t h w) -> b t h w", t=T//tublet_size, h=H//patch_size,w=W//patch_size)
        bool_masked_pos_label = bool_masked_pos_label.repeat(1, N//(T//tublet_size), 1, 1)
        bool_masked_pos_label = bool_masked_pos_label.reshape(B, -1)

        flow_target = flow_target[bool_masked_pos_label]
        flow_target = rearrange(flow_target, '(b n) d -> b n d', b=B)

        flow_target = flow_target.to(device, non_blocking=True)

        weight = None
        if weighted_flow2rgb_recons:
            # print(flows.shape)
            # weight = flows.permute(1,2)
            # weight[:,:,0,...] = (weight[:,:, 0,...] - weight[:,:, 0, ...].mean())
            # weight[:,:,1,...] = (weight[:,:, 1,...] - weight[:,:, 1, ...].mean())

            weight = flows - 0.5
            weight = torch.sqrt(weight[:, 0, ...]**2 + weight[:, 1, ...]**2) # B, T=8, H, W
            B, T, H, W = weight.shape
            # weight = weight / weight.sum((2,3)).view(B, T, 1, 1).repeat(1, 1, H, W)
            weight = weight.unsqueeze(1).repeat(1, 3, 2, 1, 1)
            # save_image(videos[0].transpose(0, 1), "frame.png")
            # save_image(weight[0].unsqueeze(1)*255, "weight.png")
            # print(weight[0, 0, 0].sum())
            weight = rearrange(weight, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)
            B, _ ,C = weight.shape
            weight = weight[bool_masked_pos_label].reshape(B, -1, C)
            weight = weight.to(device, non_blocking=True)
            # print(weight.shape)

        # print(f"final label: {flows.shape}")

        with torch.cuda.amp.autocast():
            outputs = model(videos, flows, bool_masked_pos)
 
            loss_dct = loss_func(outputs, [rgb_target, flow_target], weight=weight)
            loss = loss_dct["sum"]

        loss_value = new_func(loss)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        if flow_optimizer is not None:
            flow_optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer if flow_optimizer is None else [optimizer, flow_optimizer], 
                                clip_grad=max_norm, parameters=model.parameters(),
                                create_graph=is_second_order)

        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(rgb_recons_loss=loss_dct["rgb_recons"])
        metric_logger.update(flow_recons_loss=loss_dct["flow_recons"])
        metric_logger.update(recons_loss=loss_dct["rgb_recons"]+loss_dct["flow_recons"])

        metric_logger.update(rgb_flow_recons=loss_dct["rgb_flow_recons"])

        if "flow_rgb_recons" in loss_dct:
            metric_logger.update(flow_rgb_recons=loss_dct["flow_rgb_recons"])
            metric_logger.update(cross_recons=loss_dct["rgb_flow_recons"]+loss_dct["flow_rgb_recons"])


        metric_logger.update(rgb_contrast=loss_dct["rgb_contrast"])
        metric_logger.update(flow_contrast=loss_dct["flow_contrast"])
        metric_logger.update(contrast_loss=loss_dct["rgb_contrast"] + loss_dct["flow_contrast"])

        metric_logger.update(loss_scale=loss_scale_value)

        min_lr = 10.
        max_lr = 0.
        flow_encoder_min_lr = 10.
        flow_encoder_max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        if flow_optimizer is not None:
            for group in flow_optimizer.param_groups:
                flow_encoder_min_lr = min(flow_encoder_min_lr, group["lr"])
                flow_encoder_max_lr = max(flow_encoder_max_lr, group["lr"])

            metric_logger.update(flow_encoder_lr=flow_encoder_max_lr)
            metric_logger.update(flow_encoder_min_lr=flow_encoder_min_lr)

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
            log_writer.update(rgb_recons=loss_dct["rgb_recons"], head="rgb_recons")
            log_writer.update(flow_recons=loss_dct["flow_recons"], head="flow_recons")
            log_writer.update(recons_loss=loss_dct["rgb_recons"] + loss_dct["flow_recons"], head="recons_loss")

            log_writer.update(rgb_flow_recons=loss_dct["rgb_flow_recons"], head="rgb_flow_recons")

            if "flow_rgb_recons" in loss_dct:
                log_writer.update(flow_rgb_recons=loss_dct["flow_rgb_recons"], head="flow_rgb_recons")
                log_writer.update(cross_recons=loss_dct["rgb_flow_recons"] + loss_dct["flow_rgb_recons"], head="cross_recons")

            log_writer.update(rgb_contrast=loss_dct["rgb_contrast"], head="rgb_contrast")
            log_writer.update(flow_contrast=loss_dct["flow_contrast"], head="flow_contrast")
            log_writer.update(contrast_loss= loss_dct["rgb_contrast"] + loss_dct["flow_contrast"], head="contrast_loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")

            if flow_optimizer is not None:
                log_writer.update(flow_encoder_lr=flow_encoder_max_lr, head="opt")
                log_writer.update(flow_encoder_min_lr=flow_encoder_min_lr, head="opt")

            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class MultiModalLoss(nn.Module):

    def __init__(self, lamb):
        super().__init__()
        self.lamb = lamb
        self.recons_loss = nn.MSELoss(reduction="none")

    def forward(self, output, rgb_target, flow_target):
        """
            output: tuple(rgb_recons, flow_recons), where rgb_recons shape is (2*B, N, C1)
            rgb_target: torch.Tensor, (B, N, C1)
            flow_target: torch.Tensor, (B, N, C2)
        """
        B, N, C = rgb_target.shape

        # print(output[0].shape)
        rgb_recons, flow_recons = output
        # print(rgb_recons.shape)

        unreduced_rgb_recons_loss = self.recons_loss(rgb_recons, torch.cat([rgb_target, rgb_target], dim=0))
        unreduced_flow_recons_loss = self.recons_loss(flow_recons, torch.cat([flow_target, flow_target], dim=0))
        rgb_recons_loss, rgb_flow_recons_loss = unreduced_rgb_recons_loss[:B].mean(), unreduced_rgb_recons_loss[B:].mean()
        flow_recons_loss, flow_rgb_recons_loss = unreduced_flow_recons_loss[:B].mean(), unreduced_flow_recons_loss[B:].mean()

        # rgb_recons_loss = self.recons_loss(rgb_recons, rgb_target).mean()
        # flow_recons_loss = self.recons_loss(flow_recons, flow_target).mean()

        loss = {
            "sum": self.lamb[0]*rgb_recons_loss +  self.lamb[1]*flow_rgb_recons_loss +  self.lamb[2]*flow_recons_loss +  self.lamb[3]*rgb_flow_recons_loss,
            # "sum": self.lamb[0]*rgb_recons_loss +  self.lamb[1]*flow_recons_loss,

            "rgb_recons": rgb_recons_loss,
            "flow_recons":flow_recons_loss,

            "rgb_flow_recons": rgb_flow_recons_loss,
            "flow_rgb_recons": flow_rgb_recons_loss,
        }


        return loss


def train_multimodal_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None,

                    mask_generators = None,
                    lamb = [1,1,1,1],
                    ):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = MultiModalLoss(lamb=lamb)

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration

        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, flows = batch[0], batch[1]

        B, *_ = videos.shape
        rgb_mask, flow_mask = mask_generators[0](batch_size=B), mask_generators[1](batch_size=B) # numpy.ndarray

        if len(videos.shape) == 6: # repeated sampling is used, (B, Repeat, C, T, H, W)
            _, repeat, *_ = videos.shape
            videos = videos.reshape(-1, *videos.shape[2:])
            for i in range(repeat):
                rgb_mask.extend(mask_generators[0](batch_size=B))
                flow_mask.extend(mask_generators[1](batch_size=B))

            flows = flows.reshape(-1, *flows.shape[2:])

        videos = videos.to(device, non_blocking=True)
        flows = flows.to(device, non_blocking=True)

        rgb_mask = torch.from_numpy(np.stack(rgb_mask, axis=0)).to(device, non_blocking=True).flatten(1).to(torch.bool)
        flow_mask = torch.from_numpy(np.stack(flow_mask, axis=0)).to(device, non_blocking=True).flatten(1).to(torch.bool)

        # use given target or simply reconstruct input video

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]

            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)

                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
                B, _, C = videos_patch.shape
                rgb_target = videos_patch[rgb_mask].reshape(B, -1, C)
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)
                B, _, C = videos_patch.shape
                rgb_target = videos_patch[rgb_mask].reshape(B, -1, C)

        # target will be processed in dataset code
        # reconstruct entire given flow images
        # flows = batch[2] # bs, 2, flow nubmers, h, w

        B, _, N, H, W = flows.shape
        _, _, T, H, W = videos.shape
        assert T%N == 0, f"Number of flows:{N} to be predicted should be divisible by number of frames:{T}"
        # print(flows.shape)

        flow_target = rearrange(flows, 'b c t (h p1) (w p2) -> b (t h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

        # tublet_size = 2
        # bool_masked_pos_label = rearrange(bool_masked_pos, "b (t h w) -> b t h w", t=T//tublet_size, h=H//patch_size,w=W//patch_size)
        # bool_masked_pos_label = bool_masked_pos_label.repeat(1, N//(T//tublet_size), 1, 1)
        # bool_masked_pos_label = bool_masked_pos_label.reshape(B, -1)

        flow_target = flow_target[flow_mask]
        flow_target = rearrange(flow_target, '(b n) d -> b n d', b=B)

        flow_target = flow_target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(videos, flows, rgb_mask, flow_mask)
 
            loss_dct = loss_func(outputs, rgb_target, flow_target)
            loss = loss_dct["sum"]

        loss_value = new_func(loss)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(),
                                create_graph=is_second_order)

        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(rgb_recons_loss=loss_dct["rgb_recons"])
        metric_logger.update(flow_recons_loss=loss_dct["flow_recons"])
        metric_logger.update(recons_loss=loss_dct["rgb_recons"]+loss_dct["flow_recons"])

        metric_logger.update(rgb_flow_recons=loss_dct["rgb_flow_recons"])
        metric_logger.update(flow_rgb_recons=loss_dct["flow_rgb_recons"])
        metric_logger.update(cross_recons=loss_dct["rgb_flow_recons"]+loss_dct["flow_rgb_recons"])

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
            log_writer.update(rgb_recons=loss_dct["rgb_recons"], head="rgb_recons")
            log_writer.update(flow_recons=loss_dct["flow_recons"], head="flow_recons")
            log_writer.update(recons_loss=loss_dct["rgb_recons"] + loss_dct["flow_recons"], head="recons_loss")

            log_writer.update(rgb_flow_recons=loss_dct["rgb_flow_recons"], head="rgb_flow_recons")
            log_writer.update(flow_rgb_recons=loss_dct["flow_rgb_recons"], head="flow_rgb_recons")
            log_writer.update(cross_recons=loss_dct["rgb_flow_recons"] + loss_dct["flow_rgb_recons"], head="cross_recons")

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


class MultiCAELoss(nn.Module):

    def __init__(self, lamb):
        super().__init__()
        self.lamb = lamb
        self.recons_loss = nn.MSELoss(reduction="mean")

    def forward(self, output, rgb_target, flow_target):
        """
            output: tuple(rgb_recons, flow_recons), where rgb_recons shape is (2*B, N, C1)
            rgb_target: torch.Tensor, (B, N, C1)
            flow_target: torch.Tensor, (B, N, C2)
        """
        B, N, C = rgb_target.shape

        # print(output[0].shape)
        rgb_recons, flow_recons = output

        mean_rgb_recons_loss = self.recons_loss(rgb_recons, rgb_target)
        mean_flow_recons_loss = self.recons_loss(flow_recons, flow_target)

        loss = {
            "sum": self.lamb[0]*mean_rgb_recons_loss +  self.lamb[1]*mean_flow_recons_loss,
            "rgb_recons": mean_rgb_recons_loss,
            "flow_recons":mean_flow_recons_loss,
        }

        return loss
    

def train_multicae_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, 
                    normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None,
                    lamb = [1,1,1,1],
                    ):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_func = MultiCAELoss(lamb=lamb)

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration

        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        videos, bool_masked_pos, flows = batch[0], batch[1], batch[2]

        if len(videos.shape) == 6: # repeated sampling is used
            videos = videos.reshape(-1, *videos.shape[2:])
            bool_masked_pos = bool_masked_pos.reshape(-1, *bool_masked_pos.shape[2:])
            flows = flows.reshape(-1, *flows.shape[2:])

        videos = videos.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        # use given target or simply reconstruct input video

        with torch.no_grad():
            # calculate the predict label
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]

            unnorm_videos = videos * std + mean  # in [0, 1]

            if normlize_target:
                videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)

                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
                B, _, C = videos_patch.shape
                rgb_target = videos_patch[bool_masked_pos].reshape(B, -1, C)
            else:
                videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', p0=2, p1=patch_size, p2=patch_size)
                B, _, C = videos_patch.shape
                rgb_target = videos_patch[bool_masked_pos].reshape(B, -1, C)

        # target will be processed in dataset code
        # reconstruct entire given flow images
        # flows = batch[2] # bs, 2, flow nubmers, h, w

        B, _, N, H, W = flows.shape
        _, _, T, H, W = videos.shape
        assert T%N == 0, f"Number of flows:{T} to be predicted should be divisible by number of frames:{N}"
        # print(flows.shape)

        flow_target = rearrange(flows, 'b c t (h p1) (w p2) -> b (t h w) (p1 p2 c)', p1=patch_size, p2=patch_size)

        tublet_size = 2
        bool_masked_pos_label = rearrange(bool_masked_pos, "b (t h w) -> b t h w", t=T//tublet_size, h=H//patch_size,w=W//patch_size)
        bool_masked_pos_label = bool_masked_pos_label.repeat(1, N//(T//tublet_size), 1, 1)
        bool_masked_pos_label = bool_masked_pos_label.reshape(B, -1)

        flow_target = flow_target[bool_masked_pos_label]
        flow_target = rearrange(flow_target, '(b n) d -> b n d', b=B)

        flow_target = flow_target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            p = random.random()
            if p > 0.5:
                flows = None
            outputs = model(videos, flows, bool_masked_pos)
 
            loss_dct = loss_func(outputs, rgb_target, flow_target)
            loss = loss_dct["sum"]

        loss_value = new_func(loss)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(),
                                create_graph=is_second_order)

        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(rgb_recons_loss=loss_dct["rgb_recons"])
        if flows is not None:
            metric_logger.update(flow_recons_loss=loss_dct["flow_recons"])
        else:
            metric_logger.update(rgb2flow_loss=loss_dct["flow_recons"])

        metric_logger.update(recons_loss=loss_dct["rgb_recons"]+loss_dct["flow_recons"])
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
            log_writer.update(rgb_recons=loss_dct["rgb_recons"], head="rgb_recons")

            if flows is not None:
                log_writer.update(flow_recons=loss_dct["flow_recons"], head="flow_recons")
            else:
                log_writer.update(flow_recons=loss_dct["rgb2flow_recons"], head="flow_recons")

            log_writer.update(recons_loss=loss_dct["rgb_recons"] + loss_dct["flow_recons"], head="recons_loss")

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



def new_func(loss):
    loss_value = loss.item()
    return loss_value


if __name__ == "__main__":
    # B = 2
    # N = 3
    # C = 4
    # q = F.normalize( torch.rand(B, N, C), p=2, dim=2 )
    # k = F.normalize( torch.rand(B, N, C), p=2, dim=2 )
    # print(q)
    # print(k)
    # q_vs_k = torch.einsum("bmd,cnd->bcmn", q, k) # (B, B, N1, N2)
    # print(q_vs_k)
    # q_vs_k = q_vs_k.flatten(2)

    # q_vs_k = torch.einsum("bbn -> bn", q_vs_k)
    # print(q_vs_k)

    # q_vs_k = q_vs_k.reshape(B, N, N)
    # print(q_vs_k)
    
    # pos_sim = torch.einsum("bnn -> bn", q_vs_k)
    # print(pos_sim)

    # pos = logsumexp_pos = torch.logsumexp(pos_sim.float(), dim=1)

    # print(pos)

    q = torch.randn(2, 3, 4)
    print(q)
    q_vs_q =  torch.einsum("bmd,cnd -> bcmn", q, q)
    print(q_vs_q)
    q_vs_q = q_vs_q.flatten(2)
    
    q_vs_q =  torch.einsum("bbn -> bn", q_vs_q)
    print(q_vs_q)
    q_vs_q = q_vs_q[:, :-1].reshape(2, 2, 4)[..., 1:].flatten()

    print(q_vs_q)

