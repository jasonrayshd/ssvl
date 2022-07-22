import os
import csv
import pandas as pd
import torch
import torch.utils.data
import random
import numpy as np
import logging

from PIL import Image

import video_transforms as transform
import epickitchens_utils as utils
from epickitchens_record import EpicKitchensVideoRecord

logger = logging.getLogger(__name__)


def get_start_end_idx(video_size, clip_size, clip_idx, num_clips, flow_pretrain=False):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.

        flow_pretrain (bool): whether pretraining and predicting flow images
        random_strategy (bool): whether sample clip of random size this will lead to no adjascent frames will be sampled

    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """


    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        # if flow_pretrain:
        #     # edited by jiachen
        #     # when predicting flow imagse in pretraining, only even number of frames will be 
        #     # sampled to predict flow images.
        #     # must start from even-indexed frame (when index starts from 0)
        #     # since flow images in Epic-kitchens are sampled between images of which previous image is always even-index
        #     start_idx = random.uniform(0, delta - 1)
        # else:
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips

    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


def temporal_sampling(num_frames, start_idx, end_idx, num_samples, start_frame=0, flow_pretrain=False):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        num_frames (int): number of frames of the trimmed action clip
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
        start_frame (int): starting frame of the action clip in the untrimmed video

        flow_pretrain (bool): whether predicting flow images at pretraining
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """

    if flow_pretrain:
        # NOTE
        # 1. num_samples is an even number
        # 2. start_idx is a odd number
        # 3. start_idx and end_idx returned by get_start_end_idx() are increased by one

        assert num_samples % 2 == 0, f"number of frames:{num_samples} to be sampled should be even"
        # assert start_idx % 2 == 1, f"start index:{start_idx} is not an odd number"
        raw_index = start_frame + torch.linspace(start_idx, end_idx, num_samples//2).long()

        index = [ ]
        rep_flag = False
        for i in range(len(raw_index)):
            if raw_index[i] % 2 != 0:
                if raw_index[i] + 1 < start_frame + num_frames:
                    index.append(raw_index[i])
                    index.append(raw_index[i] + 1)
                else:
                    # replicate last frames
                    rep_flag = True
                    print("Replicating last two frames")
                    index.extend(index[-2:])
            else:
                if raw_index[i] < start_frame + num_frames:
                    index.append(raw_index[i] - 1)
                    index.append(raw_index[i])
                else:
                    rep_flag = True
                    print("Replicating last two frames")
                    index.extend(index[-2:])

        index = torch.clamp( torch.as_tensor(index), start_frame, start_frame + num_frames - 1)

        if rep_flag:
            print(raw_index, index, start_frame, start_frame + num_frames, num_frames)

        return index
    else:
        index = torch.linspace(start_idx, end_idx, num_samples)
        index = torch.clamp(index, 0, num_frames - 1).long()
        return start_frame + index


def pack_frames_to_video_clip(cfg, video_record, temporal_sample_index, target_fps=60,
                            as_pil=False, use_preprocessed_flow=False, flow_mode="A", flow_pretrain=False,
                            mode = "train"
                            ):

    """
        ...

        as_pil (bool): whether return frames as pil image
        use_preprocessed_flow (bool): whether use flow images or not (for pretraining)
        flow_mode (str): work with use_preprocessed_flow, indicates different flow image sampling strategy
        flow_pretrain (bool): whether predicting flow images at pretraining
    """

    if cfg.VERSION == 100:
        # if is epic-kitchen 100
        path_to_video = '{}/{}/rgb_frames/{}'.format(cfg.EPICKITCHENS.VISUAL_DATA_DIR,
                                                    video_record.participant,
                                                    video_record.untrimmed_video_name)

        path_to_flow = '{}/{}/flow_frames/{}'.format(cfg.EPICKITCHENS.VISUAL_DATA_DIR,
                                                    video_record.participant,
                                                    video_record.untrimmed_video_name)

    elif cfg.VERSION == 55:
        message = f"Unkown split:{mode} for Epic-Kitchen55, expect one of [train/test]"
        assert mode in ["train", "test"], message
        # else if epic-kitchen 55
        # Load video by loading its extracted frames
        path_to_video = '{}/rgb/{}/{}/{}'.format(cfg.EPICKITCHENS.VISUAL_DATA_DIR,
                                                    mode,
                                                    video_record.participant,
                                                    video_record.untrimmed_video_name)
        path_to_flow = '{}/flow/{}/{}/{}'.format(cfg.EPICKITCHENS.VISUAL_DATA_DIR,
                                                    mode,
                                                    video_record.participant,
                                                    video_record.untrimmed_video_name)

    else:
        raise ValueError(f"Unknwon Epic-kitchen version: {cfg.VERSION}")

    # code below will extract frames from compressed file [zip, tar] if the directory not exist
    if not os.path.isdir(path_to_video):
        utils.extract_zip(path_to_video)

    img_tmpl = "frame_{:010d}.jpg"
    fps, sampling_rate, num_samples = video_record.fps, cfg.DATA.SAMPLING_RATE, cfg.DATA.NUM_FRAMES

    if flow_pretrain is True:
        # indicates that we are pretrainning on Epic-kitchen by predicting flow images
        assert num_samples % 2 == 0, \
            "When pretraining on Epic-kitchen and predicting flow images, number of sampled frames should be even number"

    start_idx, end_idx = get_start_end_idx(
        video_record.num_frames,
        num_samples * sampling_rate * fps / target_fps,
        temporal_sample_index,
        cfg.TEST.NUM_ENSEMBLE_VIEWS,
        flow_pretrain = flow_pretrain,
    )

    start_idx, end_idx = start_idx + 1, end_idx + 1
    frame_idx = temporal_sampling(video_record.num_frames,
                                  start_idx, end_idx, num_samples,
                                  start_frame=video_record.start_frame,
                                  flow_pretrain = flow_pretrain,
                                  )
    img_paths = [os.path.join(path_to_video, img_tmpl.format(idx.item())) for idx in frame_idx]
    # if use flow, this indicates pretrain is used then return frames of pil format
    frames = utils.retry_load_images(img_paths, as_pil=as_pil, path_to_compressed = path_to_video)

    if use_preprocessed_flow:
        # NOTE
        # idx range in [1, video frames]
        if not os.path.isdir(path_to_flow):
            utils.extract_zip(path_to_flow)

        if flow_mode == "A":
            # sample strategy:
            # frame of odd index:  sample flow between it and its next frame
            # frame of even index: sample flow between its next frames

            # NOTE
            # when using this strategy last frame of a video should not be sampled
            # since corresponding flow image might not exist

            u_flow_paths = []
            v_flow_paths = []

            for i in range(0, len(frame_idx), 2):
                idx  = frame_idx[i]
                assert idx % 2 == 1, f"idx:{idx} should be an odd number. video_record.start_frame:{video_record.start_frame} path_to_video:{path_to_video}, frame_idx:{frame_idx}. {start_idx}, {end_idx}, untrimmed_video_name:{video_record.untrimmed_video_name}, {video_record.num_frames}"
    
                upath = os.path.join(path_to_flow, "u", img_tmpl.format( idx.item()//2 + 1))
                vpath = os.path.join(path_to_flow, "v", img_tmpl.format( idx.item()//2 + 1))

                # if not os.path.exists(upath) or not os.path.exists(vpath):
                #     # if we sampled last frame of a video that corresponding flow or flow image of its subsequent frames does not exist
                #     # then this should be the last iteration and this frame should be the last frame in the sampled frame list
                #     # we can simply drop this flow and do not compute the loss of corresponding predicted flow image

                #     assert i == len(frame_idx) - 1, f"Corresponding flow image of this frame or flow image of its subsequent frames does not exist\n and this frame is not the last sampled frame:\n {path_to_video}: {idx}"

                #     print(f"Warning: found a none existent flow image: {path_to_flow} {upath.split('/')[-1]}")
                #     print(" last predicted flow image will not be computed in loss")

                u_flow_paths.append(upath)
                v_flow_paths.append(vpath)

            uflows = utils.retry_load_images(u_flow_paths, as_pil=True, path_to_compressed= path_to_flow)
            vflows = utils.retry_load_images(v_flow_paths, as_pil=True, path_to_compressed= path_to_flow)
            
            # print(np.array(uflows[0])[:10,:10])
            return frames, uflows, vflows
        else:
            raise ValueError(f"Unknown flow mode {flow_mode}, available modes are [A]")

    return frames


"""
Used Configuration:

# general
EPICKITCHENS.ANNOTATIONS_DIR    path to directory that contains annotation file
EPICKITCHENS.VISUAL_DATA_DIR    path to directory that contains data of different participants
EPICKITCHENS.TRAIN_LIST         path to pickle file that contains information of training data
EPICKITCHENS.VAL_LIST           ...
EPICKITCHENS.TEST_LIST          ...
DATA.MEAN                       float that represents mean used to normalize dataset
DATA.STD                        float that represents std used to normalize dataset
DATA.SAMPLING_RATE              int 
DATA.NUM_FRAMES                 int 

# train, val, trian+val
DATA.TRAIN_JITTER_SCALES
DATA.TRAIN_CROP_SIZE

# test
TEST.NUM_SPATIAL_CROPS
TEST.NUM_ENSEMBLE_VIEWS

DATA.TEST_CROP_SIZE

# slowfast
MODEL.ARCH
MODEL.SINGLE_PATHWAY_ARCH
MODEL.MULTI_PATHWAY_ARCH

"""

class Epickitchens(torch.utils.data.Dataset):

    def __init__(self, cfg, mode, pretrain=False, predict_preprocessed_flow=False, pretrain_transform=None,  flow_mode = "A"):

        assert mode in [
            "train",
            "val",
            "test",
            "train+val"
        ], "Split '{}' not supported for EPIC-KITCHENS".format(mode)
        self.cfg = cfg
        self.mode = mode
        self.pretrain = pretrain                      # pretrain or not
        self.pretrain_transform = pretrain_transform  # data transformation for pretraining
        self.predict_preprocessed_flow = predict_preprocessed_flow # whether predicting flow images at pretraining
        # self.use_preprocessed_flow = use_preprocessed_flow
        # self.flow_pretrain = flow_pretrain            
        self.flow_mode = flow_mode                    # mode of loading flow images, different modes will produce different number of flow images

        self.target_fps = 60
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val", "train+val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing EPIC-KITCHENS {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        if self.mode == "train":
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TRAIN_LIST)]
        elif self.mode == "val":
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.VAL_LIST)]
        elif self.mode == "test":
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, self.cfg.EPICKITCHENS.TEST_LIST)]
        else:
            # train and val
            path_annotations_pickle = [os.path.join(self.cfg.EPICKITCHENS.ANNOTATIONS_DIR, file)
                                       for file in [self.cfg.EPICKITCHENS.TRAIN_LIST, self.cfg.EPICKITCHENS.VAL_LIST]]

        for file in path_annotations_pickle:
            assert os.path.exists(file), "{} dir not found".format(
                file
            )
        self._video_records = []
        self._spatial_temporal_idx = []
        for file in path_annotations_pickle:
            if "csv" in file:
                reader = csv.reader(open(file, "r"))
                for row in reader:
                    tup = [row[0], 
                    {
                        "participant_id": row[1],
                        "video_id": row[2],
                        "start_timestamp": "00:" + row[4],
                        "stop_timestamp": "00:" + row[5],
                        "verb_class": row[10],
                        "noun_class": row[12],
                    }]
                    for idx in range(self._num_clips):
                        self._video_records.append(EpicKitchensVideoRecord(tup))
                        self._spatial_temporal_idx.append(idx)

            elif "pkl" in file:
                for tup in pd.read_pickle(file).iterrows():
                    for idx in range(self._num_clips):
                        self._video_records.append(EpicKitchensVideoRecord(tup))
                        self._spatial_temporal_idx.append(idx)

        assert (
                len(self._video_records) > 0
        ), "Failed to load EPIC-KITCHENS split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        logger.info(
            "Constructing epickitchens dataloader (size: {}) from {}".format(
                len(self._video_records), path_annotations_pickle
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.mode in ["train", "val", "train+val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            if self.cfg.TEST.NUM_SPATIAL_CROPS == 3:
                spatial_sample_index = (
                    self._spatial_temporal_idx[index]
                    % self.cfg.TEST.NUM_SPATIAL_CROPS
                )
            elif self.cfg.TEST.NUM_SPATIAL_CROPS == 1:
                spatial_sample_index = 1
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        # load frames (and flows)
        if not self.predict_preprocessed_flow:
            # if not pretrainning or is pretraining but do not need to use flow images,
            # then not load flow images
            frames = pack_frames_to_video_clip(self.cfg, self._video_records[index], temporal_sample_index, as_pil=self.pretrain, flow_pretrain=False, mode=self.mode)
        else:
            # else load flow images according to given mode
            frames, vflows, uflows = pack_frames_to_video_clip(self.cfg, self._video_records[index], temporal_sample_index, as_pil=self.pretrain, use_preprocessed_flow=True, flow_mode=self.flow_mode, flow_pretrain=True, mode=self.mode)

        # data augmentation
        if not self.pretrain:
            # Perform color normalization.
            frames = frames.float()
            frames = frames / 255.0
            frames = frames - torch.tensor(self.cfg.DATA.MEAN)
            frames = frames / torch.tensor(self.cfg.DATA.STD)
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            frames = self.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
            )
        else:
            # frames, flows share the same mask
            if self.predict_preprocessed_flow:
                flows = [uflows, vflows]

                frames, flows, mask = self.pretrain_transform((frames, flows), use_preprocessed_flow=True) # frames shape: C*T, H, W
                frames = frames.view((self.cfg.DATA.NUM_FRAMES, 3) + frames.size()[-2:]).transpose(0,1) # 3, num_frames, H, W
                # flows are processed in pretrain_transform
                # flows = flows.view((self.cfg.DATA.NUM_FRAMES, 2) + frames.size()[1:3]).transpose(0,1) # 2, num_flows, H, W
            else:
                # vanilla MAE pretraininng or recontruct input with predicted flow images
                frames, mask = self.pretrain_transform((frames, None), use_preprocessed_flow=False)
                frames = frames.view((self.cfg.DATA.NUM_FRAMES, 3) + frames.size()[-2:]).transpose(0,1) 

        label = self._video_records[index].label
        # commented by jiachen, if use slowfast network, then uncomment this line
        # frames = utils.pack_pathway_output(self.cfg, frames)
        metadata = self._video_records[index].metadata

        if not self.pretrain:
            # not pretrain, keep the original implementation
            return frames, label, index, metadata
        elif self.pretrain and not self.predict_preprocessed_flow:
            # if is pretrain but do not need to use flow images
            return frames, mask, label, index, metadata
        else:
            # pretrain and need flow images
            return frames, mask, flows, label, index, metadata

    def __len__(self):
        return len(self._video_records)

    def spatial_sampling(
            self,
            frames,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=224,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.random_crop(frames, crop_size)
            frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames


    
if __name__ == "__main__":
    pass