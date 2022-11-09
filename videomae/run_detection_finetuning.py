# Reference
# [1] https://github.com/facebookresearch/detectron2
# [2] https://github.com/EGO4D/hands-and-objects/blob/main/state-change-object-detection-baselines/FasterRCNN/train_net.py

import os
import time
import json
import random
import logging

import torch

from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.utils import comm

from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog


def load_ego4d_scod_json(json_file, image_root, is_test=False):
    meta = json.load(open(json_file, "r"))
    clips = meta["clips"]
    
    lst_dict = []
    image_id = 1

    for clip in clips:
        data_dict = {}
        data_dict["file_name"] = os.path.join(image_root, clip["video_uid"], str(clip["pnr_frame"]["frame_number"])+".jpeg")
        data_dict["pre_file_name"] = os.path.join(image_root, clip["video_uid"], str(clip["pre_frame"]["frame_number"])+".jpeg")
        data_dict["post_file_name"] = os.path.join(image_root, clip["video_uid"], str(clip["post_frame"]["frame_number"])+".jpeg")
        data_dict["height"] = clip["pnr_frame"]["height"]
        data_dict["width"] = clip["pnr_frame"]["width"]
        data_dict["image_id"] = image_id
        data_dict["annotations"] = []

        if is_test:
            image_id += 1
            lst_dict.append(data_dict)
            continue

        for bbox in clip["pnr_frame"]["bbox"]:

            if bbox["object_type"] == "object_of_change":
                data_dict["annotations"].append({
                    "segmentation": [],
                    "category_id": 1,
                    "bbox": [bbox["bbox"]["x"], bbox["bbox"]["y"], bbox["bbox"]["width"], bbox["bbox"]["height"]],
                    "bbox_mode": 1,
                    "iscrowd": 0,
                })

        image_id += 1
        lst_dict.append(data_dict)

    return lst_dict


DATA_FOLDER = "/data/shared/ssvl/ego4d/v1/"
for split in ["train", "val"]: # test_unannotated
    
    # register_coco_instances(
    #     f"ego4dv1_pnr_objects_{split}", {},
    #     os.path.join(DATA_FOLDER, "annotations", 'fho_scod_coco_annotations', f"{split}.json"),
    #     os.path.join(DATA_FOLDER, "fho_scod", "pre_pnr_post_frames"))
    json_file = os.path.join(DATA_FOLDER, "annotations", f"fho_scod_{split}.json")
    image_root = os.path.join(DATA_FOLDER, "fho_scod", "pre_pnr_post_frames")
 
    DatasetCatalog.register(f"ego4dv1_pnr_objects_{split}", lambda: load_ego4d_scod_json(json_file, image_root, is_test=("test" in split)))


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if comm.is_main_process():
        if len(missing_keys) > 0:
            print("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(ignore_missing_keys) > 0:
            print("Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys))
        if len(error_msgs) > 0:
            print('\n'.join(error_msgs))


class Ego4dDetectionCheckpointer(DetectionCheckpointer):

    def load(self, path, checkpointables=[]):
        if path == "":
            return

        raw_state_dict = torch.load(path)
        state_dict = {}

        for k, v in raw_state_dict["model"].items():

            if k.startswith("encoder.blocks"):
                state_dict["backbone.net."+k[8:]] = v
        # print(state_dict)
        load_state_dict(self.model, state_dict, prefix="")

        return raw_state_dict.keys() - state_dict


class Ego4dTrainer(AMPTrainer):


    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with autocast():

            loss_dict = self.model(data)

            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()



def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    print(optim.param_groups[0]["lr"])

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = Ego4dTrainer(model, train_loader, optim)
    checkpointer = Ego4dDetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        Ego4dDetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )