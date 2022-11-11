
from cmath import e
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


import utils.logging as logging
from utils.parser import parse_args, load_config
from tasks.keyframe_detection import StateChangeAndKeyframeLocalisation
import wandb



logger = logging.get_logger(__name__)

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def main(cfg):
    if cfg.DATA.TASK == "state_change_detection_and_keyframe_localization":
        TaskType = StateChangeAndKeyframeLocalisation
    else:
        raise NotImplementedError('Task {} not implemented.'.format(
            cfg.DATA.TASK
        ))

    task = TaskType(cfg)

    if cfg.MISC.ENABLE_LOGGING:
        args = {
            'callbacks': [LearningRateMonitor()]
        }
    else:
        args = {
            'logger': False
        }
    
    print(cfg.MISC.NUM_SHARDS)
    trainer = Trainer(
        gpus=[2],
        num_nodes=cfg.MISC.NUM_SHARDS,
        # accelerator=cfg.SOLVER.ACCELERATOR, # deprecated
        strategy=cfg.SOLVER.ACCELERATOR,
        # strategy="dp",
        max_epochs=cfg.SOLVER.MAX_EPOCH,
        num_sanity_val_steps= 0, # run validation before training

        benchmark=True,
        replace_sampler_ddp=False,
        checkpoint_callback=ModelCheckpoint(
            monitor=task.checkpoint_metric,
            # monitor='val_loss',
            mode="max",
            save_last=True,
            save_top_k=3,
        ),
        # limit_train_batches = 1, # limit the training number of batches in each epoch to 1
        # limit_val_batches = 1, 
        fast_dev_run=cfg.MISC.FAST_DEV_RUN,
        default_root_dir=cfg.MISC.OUTPUT_DIR,
        # default_root_dir="/mnt/traffic/home/zhongjieming",
        resume_from_checkpoint=cfg.MISC.CHECKPOINT_FILE_PATH,
        # enable_progress_bar=False, # disable progress bar
        **args
    )

    if cfg.TRAIN.TRAIN_ENABLE and cfg.TEST.ENABLE:
        trainer.fit(task)
        return trainer.test()

    elif cfg.TRAIN.TRAIN_ENABLE:
        return trainer.fit(task)

    elif cfg.TEST.ENABLE:
        return trainer.test(task)


if __name__ == "__main__":
    args = parse_args()
    wandb.init(project="keyframe_loc_baseline",
            name="i3d_resnet",
            config={"learning_rate":0.0001, "batch_size":8, "epoch":100},
            resume='None')
    main(load_config(args))
    
