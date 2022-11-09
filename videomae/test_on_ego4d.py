import argparse
from cmath import e
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict

from tqdm import tqdm
from timm.models import create_model
from datasets import build_dataset
import utils
from utils import samples_collate_ego4d_test
import modeling_finetune
from config_utils import parse_yml, combine

from multiprocessing.managers import SyncManager
import multiprocessing as mp
from flow_extractor import flowExtractor


def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default= 2)
    parser.add_argument('--input_size', default=224, type=int,
                        help='videos input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, metavar='PCT',
                        help='Attention dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--short_side_size', type=int, default=224)
    parser.add_argument('--test_num_segment', type=int, default=5)
    parser.add_argument('--test_num_crop', type=int, default=3)

    # Finetuning params
    parser.add_argument('--ckpt', default='', help='checkpoint for testing')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    # parser.add_argument('--init_scale', default=0.001, type=float)
    parser.add_argument('--use_mean_pooling', action='store_true')
    parser.set_defaults(use_mean_pooling=True)
    parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')

    # Finetuning on ego4d
    # parser.add_argument('--clip_len', type=int, default=8, help="time duration of clip, default is 8s")

    # Dataset parameters
    # parser.add_argument('--data_path', default='/path/to/list_kinetics-400', type=str,
    #                     help='dataset path')
    # parser.add_argument('--eval_data_path', default=None, type=str,
    #                     help='dataset path for evaluation')
    # parser.add_argument('--nb_classes', default=400, type=int,
    #                     help='number of the classification types')
    # parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    # parser.add_argument('--num_segments', type=int, default= 1)
    # parser.add_argument('--num_frames', type=int, default= 16)
    # parser.add_argument('--sampling_rate', type=int, default= 4)

    # parser.add_argument('--output_dir', default='',
    #                     help='path where to save, empty for no saving')
    # parser.add_argument('--log_dir', default=None,
    #                     help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')


    parser.add_argument('--name', type=str, default="temp", help="name of current run")
    parser.set_defaults(debug=False)
    parser.add_argument('--anno_path', type=str, default="", help="save path of annotation files of ego4d state change, which includes train.json, val.json, test.json")
    parser.add_argument('--config', type=str, default="", help="path to configuration file")

    parser.add_argument('--overwrite', type=str, default="command-line", help="overwrite command-line argument or arguments from configuration file")
    return parser.parse_args()



def main(args):

    utils.init_distributed_mode(args)
    # codes below should be called after distributed initialization
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    if args.flow_mode == "online":
        mp.set_start_method('spawn')
        SyncManager.register("flowExtractor", flowExtractor)
        m = SyncManager()
        m.start()
        flow_extractor = m.flowExtractor(device=f"cuda:{args.gpu}")
        print(f"Flow extractor manager started by {os.getpid()}.")
    else:
        flow_extractor = None

    dataset_test, _ = build_dataset(is_train=False, test_mode=True, args=args, flow_extractor=flow_extractor)
    if args.dist_eval:
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    assert args.log_dir is not None and args.output_dir is not None, "log_dir and output_dir should not be empty"
    args.log_dir = os.path.join(args.log_dir, args.name)
    args.output_dir = os.path.join(args.output_dir, args.name)
    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)  


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn = samples_collate_ego4d_test,
    )

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes, # when equals to 1, perform ego4d state classification and localization tasks at the same time
        all_frames = args.cfg.DATA.CLIP_LEN_SEC * args.cfg.DATA.SAMPLING_FPS,
        tubelet_size=args.tubelet_size,
        # drop_rate=args.drop,
        # drop_path_rate=args.drop_path,
        # attn_drop_rate=args.attn_drop_rate,
        # drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        # init_scale=args.init_scale,

        # if is ego4d and state change localization task, then the output dimension of feature 
        # keep_dim = True if (args.nb_classes == args.num_frames+1) and ("ego4d" in args.data_set.lower()) else False
    )

    # patch_size = model.patch_embed.patch_size
    # print("Patch size = %s" % str(patch_size))
    # args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    # args.patch_size = patch_size

    checkpoint = torch.load(args.ckpt, map_location='cpu')
    print("Load ckpt from %s" % args.ckpt)
    checkpoint_model = None
    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    model.load_state_dict(checkpoint_model, strict=True)
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)

    print(f"Start Testing on Ego4d")
    start_time = time.time()

    preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
    test_on_ego4d(data_loader_test, model, device, preds_file)

    torch.distributed.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


@torch.no_grad()
def test_on_ego4d(data_loader, model, device, file):

    # switch to evaluation mode
    model.eval()
    final_result = []

    if not os.path.exists(file):
        os.mknod(file)
    open(file,"w").close() # clear previous content
    f =  open(file, 'a+')

    for batch in tqdm(data_loader):
        # print(len(batch))

        videos = batch[0]
        flows = batch[1]
        info = batch[2]
        frame_idx = batch[3]

        # print(videos.shape)
        batch_size = videos.shape[0]
        videos = videos.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast():
            if flows is not None:
                flows = flows.to(device, non_blocking=True)
                output = model(videos, flows)
            else:
                output = model(videos)

        for i in range(batch_size):

            if isinstance(output, tuple):
                string = "{} {} {} {} {}\n".format(info[i]["unique_id"],
                                                    str(output[0].data[i].cpu().numpy().tolist()),
                                                    str(output[1].data[i].cpu().numpy().tolist()),
                                                    str(info[i]["crop"]),
                                                    str(frame_idx[i].tolist()),
                                                    )
            else:
                string = "{} {} {} {}\n".format(info[i]["unique_id"],
                                                    str(output.data[i].cpu().numpy().tolist()),
                                                    str(info[i]["crop"]),
                                                    str(frame_idx[i].tolist()),
                                                    )

            f.write(string)

    f.close()



if __name__ == '__main__':
    opts = get_args()
    config = parse_yml(opts.config)
    if config is not None:
        opts = combine(opts, config)

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
