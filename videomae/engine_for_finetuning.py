import os
import sys
import math
import numpy as np
from functools import partial

import torch
from torch import nn

from timm.data import Mixup
from timm.utils import accuracy

import utils
from typing import Iterable, Optional

# from loss import Ego4dTwoHead_Criterion

from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance

def train_class_batch(model, samples, target, criterion):
    # print("train_class_batch")
    frames, flows = samples
    if flows is None:
        outputs = model(frames)
    else:
        outputs = model(frames, flows)

    loss = criterion(outputs, target)

    return loss, outputs


def lta_metric(out_actions, target_actions):

    """
        Args:
        out_actions: list[Tensor: B,num_class] of length Z
        target_actions: list[Tensor: B] of length Z

    """

    out_actions = [torch.argmax(action, dim=1) for action in out_actions] # list[Tensor: B] of length Z
    out_actions = torch.stack(out_actions, dim=0).transpose(0,1) # (B, Z)
    out_actions_lst = out_actions.tolist()
    target_actions = torch.stack(target_actions, dim=0).transpose(0,1) # (B, Z)
    target_actions_lst = target_actions.tolist()

    B, Z = out_actions.shape
    unreduced_score = 0.
    action_pred_accuracy = []
    # compute action classfication accuracy
    for i in range(Z):
        class_acc = (out_actions[:, i] == target_actions[:, i]).float().mean()
        action_pred_accuracy.append(class_acc)
    
    # compute edit distance
    for i in range(B):
        unreduced_score += normalized_damerau_levenshtein_distance(out_actions_lst[i], target_actions_lst[i])
    score = unreduced_score / B

    return score, action_pred_accuracy


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale    

def lta_train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,
                    head_type = "varant",
                    ):

    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, flows, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=False)
        if flows is not None:
            flows = flows.to(device, non_blocking=False)

        # verbs, nouns = targets
        # verbs = verbs.to(device, non_blocking=False)
        # nouns = nouns.to(device, non_blocking=False)
        # targets = [verbs, nouns]

        targets = [target.to(device, non_blocking=False) for target in targets] # verb or noun, depends on dataset
 
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            # print(samples.shape, samples.device, targets[0].device, targets[1].device)

        
        if loss_scaler is None:
            # print("loss scaler is None")
            samples = samples.half()
            if flows is not None:
                flows = flows.half()
            loss, output = train_class_batch(
                model, [samples, flows], targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, [samples, flows], targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        if head_type == "varant":
            out_cur, out_future, kld_obs_goal, kld_next_goal, kld_goal_diff, kld_future_goal, kld_future_goal_dis = output
            metric_logger.update(kld_obs_goal=kld_obs_goal)
            metric_logger.update(kld_next_goal=kld_next_goal)
            metric_logger.update(kld_goal_diff=kld_goal_diff)
            metric_logger.update(kld_future_goal=kld_future_goal)
            metric_logger.update(kld_future_goal_dis=kld_future_goal_dis)

            out_action =[out_cur, *out_future]
        elif head_type == "baseline":
            out_action = output


        if mixup_fn is None:
            # compute metrics for action anticipation
            
            DL_edit_distance, action_pred_acc = lta_metric(out_action, targets)

            metric_logger.update(DL_edit_distance=DL_edit_distance)
            metric_logger.update(action_pred_acc_0=action_pred_acc[0])
            metric_logger.update(action_pred_acc_1=action_pred_acc[1])
            metric_logger.update(action_pred_acc_2=action_pred_acc[2])
            metric_logger.update(action_pred_acc_3=action_pred_acc[3])
            metric_logger.update(action_pred_acc_4=action_pred_acc[4])
            metric_logger.update(action_pred_acc_5=action_pred_acc[5])
            metric_logger.update(action_pred_acc_6=action_pred_acc[6])
            metric_logger.update(action_pred_acc_7=action_pred_acc[7])
            metric_logger.update(action_pred_acc_8=action_pred_acc[8])
            metric_logger.update(action_pred_acc_9=action_pred_acc[9])
            metric_logger.update(action_pred_acc_10=action_pred_acc[10])
            metric_logger.update(action_pred_acc_11=action_pred_acc[11])
            metric_logger.update(action_pred_acc_12=action_pred_acc[12])
            metric_logger.update(action_pred_acc_13=action_pred_acc[13])
            metric_logger.update(action_pred_acc_14=action_pred_acc[14])
            metric_logger.update(action_pred_acc_15=action_pred_acc[15])
            metric_logger.update(action_pred_acc_16=action_pred_acc[16])
            metric_logger.update(action_pred_acc_17=action_pred_acc[17])
            metric_logger.update(action_pred_acc_18=action_pred_acc[18])
            metric_logger.update(action_pred_acc_19=action_pred_acc[19])

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

            if head_type == "varant":
                log_writer.update(kld_obs_goal=kld_obs_goal, head="loss")
                log_writer.update(kld_next_goal=kld_next_goal, head="loss")
                log_writer.update(kld_goal_diff=kld_goal_diff, head="loss")
                log_writer.update(kld_future_goal=kld_future_goal, head="loss")
                log_writer.update(kld_future_goal_dis=kld_future_goal_dis, head="loss")

            if mixup_fn is None:
                log_writer.update(DL_edit_distance=DL_edit_distance, head="loss")

            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def osccpnr_train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, flows, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=False)
        if flows is not None:
            flows = flows.to(device, non_blocking=False)
        targets = targets.to(device, non_blocking=False)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            # print(samples.shape, samples.device, targets[0].device, targets[1].device)

        if loss_scaler is None:
            # print("loss scaler is None")
            samples = samples.half()
            if flows is not None:
                flows = flows.half()
            loss, output = train_class_batch(
                model, [samples, flows], targets, criterion)

        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, [samples, flows], targets, criterion)
        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        class_acc = None
        if mixup_fn is None:
            if isinstance(targets, torch.Tensor):
                class_acc = (output.max(-1)[-1] == targets).float().mean()

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
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

        # if log_writer is not None:
        #     log_writer.update(loss=loss_value, head="loss")
        #     log_writer.update(class_acc=class_acc, head="loss")
        #     log_writer.update(loss_scale=loss_scale_value, head="opt")
        #     log_writer.update(lr=max_lr, head="opt")
        #     log_writer.update(min_lr=min_lr, head="opt")
        #     log_writer.update(weight_decay=weight_decay_value, head="opt")
        #     log_writer.update(grad_norm=grad_norm, head="opt")

        #     log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, criterion, task):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        flows = batch[1]
        targets = batch[2]

        videos = videos.to(device, non_blocking=True)
        if flows is not None:
            flows = flows.to(device, non_blocking=True)

        if "lta" in task: # [lta_verb, lta_noun]
            targets = [target.to(device, non_blocking=False) for target in targets] # verb or noun, depends on dataset
        else:
            targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            if flows is not None:
                output = model(videos, flows)
                loss = criterion(output, targets)
            else:
                output = model(videos)
                loss = criterion(output, targets)

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())

        if "lta" in task: # [lta_verb, lta_noun]

            if criterion.head_type == "varant":
                out_cur, out_future, kld_obs_goal, kld_next_goal, kld_goal_diff, kld_future_goal, kld_future_goal_dis = output
                out_action =[out_cur, *out_future]

                metric_logger.update(kld_obs_goal=kld_obs_goal)
                metric_logger.update(kld_next_goal=kld_next_goal)
                metric_logger.update(kld_goal_diff=kld_goal_diff)
                metric_logger.update(kld_future_goal=kld_future_goal)
                metric_logger.update(kld_future_goal_dis=kld_future_goal_dis)

            else:
                out_action = output

            # compute metrics for action anticipation
            DL_edit_distance, action_pred_acc = lta_metric(out_action, targets)

            metric_logger.update(DL_edit_distance=DL_edit_distance)

            metric_logger.update(action_pred_acc_0=action_pred_acc[0])
            metric_logger.update(action_pred_acc_1=action_pred_acc[1])
            metric_logger.update(action_pred_acc_2=action_pred_acc[2])
            metric_logger.update(action_pred_acc_3=action_pred_acc[3])
            metric_logger.update(action_pred_acc_4=action_pred_acc[4])
            metric_logger.update(action_pred_acc_5=action_pred_acc[5])
            metric_logger.update(action_pred_acc_6=action_pred_acc[6])
            metric_logger.update(action_pred_acc_7=action_pred_acc[7])
            metric_logger.update(action_pred_acc_8=action_pred_acc[8])
            metric_logger.update(action_pred_acc_9=action_pred_acc[9])

            metric_logger.update(action_pred_acc_10=action_pred_acc[10])
            metric_logger.update(action_pred_acc_11=action_pred_acc[11])
            metric_logger.update(action_pred_acc_12=action_pred_acc[12])
            metric_logger.update(action_pred_acc_13=action_pred_acc[13])
            metric_logger.update(action_pred_acc_14=action_pred_acc[14])
            metric_logger.update(action_pred_acc_15=action_pred_acc[15])
            metric_logger.update(action_pred_acc_16=action_pred_acc[16])
            metric_logger.update(action_pred_acc_17=action_pred_acc[17])
            metric_logger.update(action_pred_acc_18=action_pred_acc[18])
            metric_logger.update(action_pred_acc_19=action_pred_acc[19])

            score = DL_edit_distance

        elif task == "oscc" or task == "pnr":
            if output.shape[1] > 5:
                acc1, acc5 = accuracy(output, targets, topk=(1, 5))
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            else:
                acc1 = accuracy(output, targets, topk=(1,))[0]
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            
            score = acc1.item()

        metric_logger.meters["score"].update(score, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #     .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    info = ""
    for k, meter in metric_logger.meters.items():
        info += f"{k}: {meter.global_avg} "
    info += "\n"
    print(info)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file, criterion):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)

        if not isinstance(target, list):
            target = target.to(device, non_blocking=True)
        else:
            labels, states = target[0].to(device, non_blocking=True), target[1].to(device, non_blocking=True)
            target = [labels, states]

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            # print(output.shape)
            loss = criterion(output, target)

        for i in range(output.size(0)):
            if isinstance(output, list):
                string = "{} {} {} {} {}\n".format(ids[i], \
                                                    str(output[0].data[i].cpu().numpy().tolist()), \
                                                    str(int(target[0][i].cpu().numpy())), \
                                                    str(int(chunk_nb[i].cpu().numpy())), \
                                                    str(int(split_nb[i].cpu().numpy())))
            else:
                string = "{} {} {} {} {}\n".format(ids[i], \
                                                    str(output.data[i].cpu().numpy().tolist()), \
                                                    str(int(target[i].cpu().numpy())), \
                                                    str(int(chunk_nb[i].cpu().numpy())), \
                                                    str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
