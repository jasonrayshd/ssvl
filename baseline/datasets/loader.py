"""
Data Loader
"""

import torch
from torch.utils.data.dataloader import default_collate
from .build_dataset import build_dataset

def construct_loader(cfg, split):
    """
    Construct the data loader for the given dataset
    """
    assert split in [
        'train',
        'val',
        'test'
    ], "Split `{}` not supported".format(split)

    if split == 'train':
        dataset_name = cfg.TRAIN.DATASET
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = cfg.DATA_LOADER.SHUFFLE
        drop_last = cfg.DATA_LOADER.DROP_LAST
    elif split == 'val':
        dataset_name = cfg.TRAIN.DATASET
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = False
        # drop_last = False # commented by Jiachen according to https://github.com/Lightning-AI/lightning/issues/11910
        drop_last = True 
    elif split == 'test':
        dataset_name = cfg.TEST.DATASET
        batch_size = cfg.TEST.BATCH_SIZE
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)
    if cfg.SOLVER.ACCELERATOR == 'dp':
        sampler =  None
    else:
        raise NotImplementedError("{} not implemented".format(
            cfg.SOLVER.ACCELERATOR
        ))
    # sampler = torch.utils.data.distributed.DistributedSampler(
    #     dataset,
    #     num_replicas = ,
    #     rank = ,
    #     shuffle = shuffle,
    #     drop_last=drop_last,
    # )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn = c_fn,
    )
    return loader


def c_fn(batch_lst):
    
    frames = []
    label = []
    state = []
    fps = []
    info = []

    for batch in batch_lst:
        _frames, _label, _state, _fps, _info = batch
        # _frames is a lst that contains torch.Tensor of shape (b,c,t,h,w)
        fps.append(_fps)
        info.append(_info)
        frames.extend(_frames)
        label.append(torch.from_numpy(_label).long())
        state.append(torch.tensor(_state).long())
    
    frames = torch.stack(frames,dim=0)
    label = torch.stack(label, dim=0)
    state = torch.stack(state, dim=0)
 
    return [frames], label, state, fps, info
