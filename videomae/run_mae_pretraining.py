import argparse
from cgi import parse_multipart
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

import copy

from pathlib import Path
from flow_extractor import flowExtractor
from timm.models import create_model
from optim_factory import create_optimizer
from datasets import build_pretraining_dataset
from engine_for_pretraining import train_one_epoch, train_tsvit_one_epoch
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import modeling_pretrain
import wandb

import logging
import socket
from epickitchens_utils import CacheManager
from multiprocessing.managers import SyncManager
import multiprocessing as mp

from config_utils import parse_yml, combine

def get_args():
    parser = argparse.ArgumentParser('VideoMAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)

    # Model parameters
    parser.add_argument('--model', default='pretrain_videomae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--decoder_depth', default=4, type=int,
                        help='depth of decoder')

    parser.add_argument('--mask_type', default='tube', choices=['random', 'tube'],
                        type=str, help='masked strategy of video tokens/patches')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
                        
    parser.add_argument('--normlize_target', default=True, type=bool,
                        help='normalized the target patch pixels')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.0, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Dataset parameters
    parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str,
                        help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_frames', type=int, default= 16)
    parser.add_argument('--sampling_rate', type=int, default= 4)
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # configuration file
    parser.add_argument('--config', default='none', type=str,
                        help='path to configuration file')
    parser.add_argument('--overwrite', type=str, default="command-line", help="overwrite command-line argument or arguments from configuration file")
    parser.add_argument('--debug', action='store_true', help="whether in debugging or not; this will prevent wandb logging and some other features")
    parser.add_argument('--name', default='temp', type=str,help='name of the experiment')
    parser.add_argument('--project', default='temp', type=str,help='name of wandb project')
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--wandb_id', default=None, type=str,
                        help='run id of wandb')
    
    parser.add_argument('--train_wo_amp', action='store_true')
    parser.add_argument('--tau', type=float, default=0.01,
                        help='temperature hyper-parameter tau')
    return parser.parse_args()


def get_model(args):
    """
        Create model for Two-Stream Pretraining or VideoMAE
    """
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

            version = args.version,
            use_rgb_stat = args.use_rgb_stat, 
            share_within_modality_proj_layer = args.share_within_modality_proj_layer,
            mask_tokenizer = args.mask_tokenizer,
            share_proj_layer = args.share_proj_layer,
            fuse_scheme = args.fuse_scheme,
            tokenizer_backbone = args.tokenizer_backbone,
        )
    return model


def load_weight_for_rgb_encoder(raw_checkpoints):
    """
        when pretraining, load weight for rgb cross-modality encoder
    """
    rgb_encoder_checkpoints = {}

    for k, v in raw_checkpoints["model"].items():
        if k.startswith("encoder"):
            rgb_encoder_checkpoints["rgb_encoder"+k[7:]] = v

    return rgb_encoder_checkpoints


def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()

    if utils.get_rank() == 0 and not args.debug:
        wandb.init(project=args.project, id=args.wandb_id, resume="must" if args.wandb_id else None, config=vars(opts))

    print(args)

    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    model = get_model(args)

    # Load weight for RGB cross-modality Encoder
    ckpt = getattr(args, "ckpt", "")
    if ckpt != "":
        raw_checkpoints = torch.load(ckpt, map_location="cpu")
        rgb_encoder_checkpoints = load_weight_for_rgb_encoder(raw_checkpoints)
        missing_keys_lst, unexpected_keys_lst = model.load_state_dict(rgb_encoder_checkpoints, strict=False)
        # Check if rgb cross-modality encoder weights are loaded successfully
        flag = True
        for k in missing_keys_lst:
            if "rgb_encoder." in k:
                flag = False
                print(f"Found an unloaded paramter of RGB cross-modality encoder:{k}")
        if flag:
            print("Successfully load pretrained weight for RGB cross-modality Encoder")

    if args.ts_pretrain:
        assert model.rgb_encoder.patch_embed.patch_size == model.flow_encoder.patch_embed.patch_size
        patch_size = model.rgb_encoder.patch_embed.patch_size
    else:
        patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))

    # window size for masking
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    if args.flow_mode == "online":
        mp.set_start_method('spawn')
        SyncManager.register("flowExtractor", flowExtractor)
        m = SyncManager()
        m.start()
        flow_extractor = m.flowExtractor(device=f"cuda:{args.gpu}")
        print(f"Flow extractor manager started by {os.getpid()}.")
    else:
        flow_extractor = None

    dataset_train = build_pretraining_dataset(args, flow_extractor=flow_extractor)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank
    num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
    )

    print("Sampler_train = %s" % str(sampler_train))

    assert args.log_dir is not None and args.output_dir is not None, "log_dir and output_dir should not be empty"
    args.log_dir = os.path.join(args.log_dir, args.name)
    args.output_dir = os.path.join(args.output_dir, args.name)
    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)  
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        multiprocessing_context="spawn" if args.flow_mode == "online" else None,
        worker_init_fn=utils.seed_worker
    )

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    total_batch_size = args.batch_size * utils.get_world_size()

    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.ts_pretrain and args.flow_encoder_lr is not None:
        ignore_param = {
            # *["flow_encoder_to_decoder."+name for name, param in model.module.flow_encoder_to_decoder.named_parameters()],
            *["flow_encoder."+name for name, param in model.module.flow_encoder.named_parameters() ]
        }
        print(ignore_param)

        optimizer = create_optimizer(
            args, model_without_ddp, 
            ignore_param = ignore_param
        )

        flow_optimizer = create_optimizer(
            args, 
            # torch.nn.Sequential(model.module.flow_encoder_to_decoder, model.module.flow_encoder),
            torch.nn.Sequential( model.module.flow_encoder),
            lr = args.flow_encoder_lr,
        )
        flow_encoder_lr_schedule_values = utils.cosine_scheduler(
            args.flow_encoder_lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            start_warmup_value=args.warmup_lr,  warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )

    else:
        optimizer = create_optimizer(
            args, model_without_ddp)

        flow_optimizer = None
        flow_encoder_lr_schedule_values = None

    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        start_warmup_value=args.warmup_lr,  warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        # strategy="fixed_in_epoch",
    )

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    torch.cuda.empty_cache()
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        if not args.ts_pretrain:
            train_stats = train_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, log_writer=log_writer,
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
                patch_size=patch_size[0],
                normlize_target=args.normlize_target,
                # use mixed precision or not
                train_wo_amp = args.train_wo_amp, 
                # whether predict given flow images or recons input based on flow images
                predict_preprocessed_flow = (args.flow_mode != ""),
            )
        else:
            train_stats = train_tsvit_one_epoch(
                model, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, log_writer=log_writer,
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values,
                flow_encoder_lr_schedule_values = flow_encoder_lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
                patch_size=patch_size[0],
                normlize_target=args.normlize_target,
                flow_optimizer = flow_optimizer,

                weighted_flow2rgb_recons = args.weighted_flow2rgb_recons,
                ctr = args.ctr,
                tau = float(args.tau),
                lamb = args.lamb,
            ) 

        if args.output_dir:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if global_rank == 0 and not args.debug:
            wandb.log(log_stats)

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    # combine arguments from configuration file and command line
    opts = get_args()
    config = parse_yml(opts.config)
    if config is not None:
        opts = combine(opts, config)

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)

    os.makedirs(os.path.join(opts.output_dir, opts.name), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(opts.output_dir, opts.name, f"console_{utils.get_rank()}_{os.environ['LOCAL_RANK']}.log"),
        filemode="w",
        level=logging.DEBUG,
    )

    main(opts)
