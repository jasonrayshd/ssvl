"""
Jiachen Lei, 2022.05.19

Reference 
https://github.com/EGO4D/hands-and-objects/tree/main/state-change-localization-classification/i3d-resnet50

"""

from builtins import NotImplemented, NotImplementedError, sorted
import os
import re
import json
import time
import csv
import random

import av
import cv2
import copy

from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import video_transforms as video_transforms 
import volume_transforms as volume_transforms
from random_erasing import RandomErasing
from ego4d_trim import _get_frames

# import detectron2.data.transforms as detection_transform
# from detectron2.data import detection_utils

import torch.nn.functional as F

import io
import random
import zipfile
from zipfile import ZipFile


def debug_functions(Dataset, **kwargs):
    dataset = Dataset(**kwargs)
    sample = dataset[0]

class Ego4dBase(torch.utils.data.Dataset):
    """
    Data loader for state change detection and key-frame localization.
    This data loader assumes that the user has alredy extracted the frames from
    all the videos using the `train.json`, `test_unnotated.json`, and
    'val.json' provided.
    """
    def __init__(self, mode, cfg, pretrain=False, flow_extractor=None, **kwargs):
        assert mode in [
            'train',
            'val',
            'test',
            'test_unannotated'
        ], "Split `{}` not supported.".format(mode)

        self.mode = mode
        self.cfg = cfg
        self.task = cfg.task            # [fho_oscc, fho_tl, fho_oscc_tl, fho_scod, fho_hands, egoclip]
        self.pretrain = pretrain        # bool, indicate pretrain or not
        self.flowExt = flow_extractor   # class, optical flow online extractor

        self.kwargs = kwargs            # for additional arguments

        self.mean = cfg.MEAN  # [0.485, 0.456, 0.406]
        self.std  = cfg.STD   # [0.229, 0.224, 0.225]

        # set required arguments for dataset
        self.init_dataset()
        # construct dataset for __getitem__
        self.build_dataset()
        # init transformation for each mode
        self.init_transformation()

    def init_dataset(self):
        raise NotImplementedError("init_dataset not implemented")

    def build_dataset(self):
        raise NotImplementedError("build_dataset not implemented")

    def init_transformation(self):
        raise NotImplementedError("init_transformation not implemented")

    def __len__(self):
        raise NotImplementedError("__len__ not implemented")

    def __getitem__(self, index):
        raise NotImplementedError("__getitem__ not implemented")

    def prepare_flow(self, clip, info, frame_idx):

        if self.load_flow == "online":
            assert self.flowExt != None, "flow extractor is None"
            flows = self.extract_flow(clip)
        elif self.load_flow == "local":
            flows = self.load_local_flow(info, frame_idx)
        elif self.load_flow == "none":
            flows = None
        else:
            raise ValueError(f"Unsupported flow mode: {self.load_flow}, available values are one of [oneline|local|none]")

        return flows

    def load_local_flow(self, info, frame_idx):
        
        flows = None
        self.clip_path = info["clip_path"]
        
        return flows

    def extract_flow(self, buffer):
        """
            Extract optical flow given frames
            Args:
                buffer: torch.Tensor, shape like (T x H x W x C)

            Return:
                flows: torch.Tensor
        """
        T, H, W, C = buffer.shape

        shifted_frames = torch.roll(buffer, -1, 0)
        concat_frames = torch.cat((buffer, shifted_frames), dim=1)

        concat_frames = torch.stack([concat_frames[i] for i in range(0, T, 2)], dim=0)

        # padding to 256x256
        concat_frames = F.pad(concat_frames, (16, 16, 16, 16), "constant", 0)
        flow_lst_dct = self.flowExt.ext(concat_frames)
        flows = np.stack([flow_dict["flow"] for flow_dict in flow_lst_dct], axis=0)
        T, H, W, C = flows.shape
        flows = flows[:, 16:240, 16:240, :]

        # noisy flows
        flows = flows.transpose(3, 0, 1, 2)
        flows = torch.from_numpy(flows)

        # standardization
        flows = (flows-flows.min()) / (flows.max() - flows.min())

        return flows

    def sample_frame(self, st_frame, end_frame, strategy="uniform"):

        num_frame = self.num_frame

        if strategy == "uniform":
            frames = np.linspace(st_frame, end_frame, num_frame).astype(int)

        elif strategy == "tsn":
            segments = num_frame
            intervals = np.linspace(st_frame, end_frame, segments+1).astype(int)
            # print(intervals)
            frames = []
            for i in range(1, len(intervals)):

                sampled_frame = random.randint(intervals[i-1], intervals[i])
                frames.append(sampled_frame)

        elif strategy == "limited_range":
            pass
        else:
            raise NotImplementedError(f"unkown sampling strategy for lta dataset:{strategy}")

        return frames

    def normalize_tensor(self, tensor, mean, std):
        """
        Normalize a given tensor by subtracting the mean and dividing the std.
        Args:
            tensor (tensor): tensor to normalize.
            mean (tensor or list): mean value to subtract.
            std (tensor or list): std to divide.
        """
        if tensor.dtype == torch.uint8:
            tensor = tensor.float()
            tensor = tensor / 255.0
        if type(mean) == list:
            mean = torch.tensor(mean)
        if type(std) == list:
            std = torch.tensor(std)
        tensor = tensor - mean
        tensor = tensor / std
        return tensor

    def spatial_sampling(
        self,
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
        random_horizontal_flip=True,
        inverse_uniform_sampling=False,
        aspect_ratio=None,
        scale=None,
        motion_shift=False,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `channels` x `num frames` x `height` x `width`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
            inverse_uniform_sampling (bool): if True, sample uniformly in
                [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
                scale. If False, take a uniform sample from [min_scale,
                max_scale].
            aspect_ratio (list): Aspect ratio range for resizing.
            scale (list): Scale range for resizing.
            motion_shift (bool): Whether to apply motion shift for resizing.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            if aspect_ratio is None and scale is None:
                frames, _ = video_transforms.random_short_side_scale_jitter(
                    images=frames,
                    min_size=min_scale,
                    max_size=max_scale,
                    inverse_uniform_sampling=inverse_uniform_sampling,
                )
                frames, _ = video_transforms.random_crop(frames, crop_size)
            else:
                transform_func = (
                    video_transforms.random_resized_crop_with_shift   # C T H W
                    if motion_shift
                    else video_transforms.random_resized_crop # C/T, T/C, H W
                )
                frames = transform_func(
                    images=frames,
                    target_height=crop_size,
                    target_width=crop_size,
                    scale=scale,
                    ratio=aspect_ratio,
                )
            if random_horizontal_flip:
                frames, _ = video_transforms.horizontal_flip(0.5, frames) # T C H W
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = video_transforms.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)

        return frames

    def _load_frame(self, frame_path):
        """
            Read a frame and do some pre-processing.
            Args:
                frame_path, str, path to the frame
            Returns:
                frames: ndarray, image in RGB format
        """
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    def assert_wtolerance(self, condition, message, retry=5):
        """
            asssert with tolerance
            Args:
                condition: bool, condition for assert
                message: str, message for assert
                retry: int, maximum number of retrying
        """

        for i in range(retry-1):
            try:
                assert condition, message
                return
            except AssertionError:
                continue
        assert condition, message
    
    def exec_wtolerance(self, func, retry, msg, **kwargs):
        """
            asssert with tolerance
            Args:
                func: function to be executed
                retry: int, maximum number of retrying
                msg: str, message to be printed when execution fails
                kwargs: arguments to be passed to func
        """

        for i in range(retry):
            try:
                return func(**kwargs)
            except Exception as e:
                if i == retry - 1:
                    print(msg)
                    raise e
                else:
                    continue


class Egoclip(Ego4dBase):

    def init_dataset(self):
        self.anno_path = os.path.join(self.cfg.ANN_DIR, "egoclip.csv")

        self.repeat_sample = self.cfg.repeat_sample
        self.input_size = self.cfg.input_size
        self.short_side_size = self.cfg.short_side_size

        self.load_flow = self.cfg.load_flow # str, [online, local]

        assert self.load_flow != "online", "Only support load flow locally while using egoclip"

    def build_dataset(self):
        """
            process egoclip csv annotation file

            Args:
                anno_path: str, path to egoclip csv annotation file
        """
        reader = csv.reader(open(self.anno_path, "r", encoding='utf-8'))
        # next(reader) # skip head for egoclip.csv
        rows = list(reader)
        self.package = []  # clips that are used for pretraining
        self.skip_lst = [] # clips that are ignored
        video2clipidx = {}
        skipped = 0
        for row in tqdm(rows):

            meta = row[0].split("\t")
            
            if meta[0] not in video2clipidx:
                video2clipidx[meta[0]] = 0
            else:
                video2clipidx[meta[0]] += 1
            
            # match verb and noun
            pattern1 = "\[(\d*)\].*?\[(\d*)"
            pattern2 = "\[(\d*).*?"
            result = re.findall(pattern1, row[0])
            if len(result) == 0:
                result = re.findall(pattern2, row[0])
                if len(result) == 0:
                    skipped += 1
                    continue # if still cannot match, then abadon
                verb = result[0]
                verb = -1 if verb == "" else int(verb)
                noun = -1
            else:
                verb, noun = result[0]
                verb = -1 if verb == "" else int(verb)
                noun = -1 if noun == "" else int(noun)

            if ("verb" in self.cfg.task and verb == -1) or ("noun" in self.cfg.task and noun == -1):
                continue # skip clip with missing label in egoclip action recognition

            pack = {
                    "video_uid": meta[0],
                    "video_dur": meta[1],
                    "narration_source": meta[2],
                    "narration_ind":meta[3],
                    "narration_time": meta[4],
                    "clip_start": meta[5],
                    "clip_end": meta[6],
                    "arration_info": "\t".join(meta[7:]),
                    "verb": verb,
                    "noun": noun,
                    "start_frame": int( float(meta[5])*30 ),
                    # NOTE 2022.11.09: end frame might do not exist due to the accuracy of float number
                    "end_frame": int( float(meta[6])*30 ),
                    "clip_idx": video2clipidx[meta[0]],
                    "spatial_temporal_idx": 0,
            }

            if pack["end_frame"] - pack["start_frame"] + 1 < self.cfg.NUM_FRAMES:
                # if the length of the clip is smaller than required number of frames to be sampled
                # then skip this clip
                self.skip_lst.append(pack)
                continue

            self.package.append(pack)
        print(f"Skipped {skipped} clips in total")

    def init_transformation(self):

        if self.pretrain:
            self.data_transform = self.kwargs["pretrain_transform"]
        else:
            self.data_transform = video_transforms.Compose([
                video_transforms.ShorterSideResize(self.short_side_size),
                video_transforms.CenterCrop(size=(self.input_size, self.input_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=self.mean,
                                            std=self.std)
            ])

    def prepare_clip_frames_flows(self, info):
        """
            Prepare training data and labels, return loaded frames and uvflows
            
        """
        # preprocess
        uid = info["video_uid"]
        clip_idx = info["clip_idx"]
        frame_zip_path = os.path.join(self.cfg.FRAME_DIR_PATH, uid, uid+"_" + "{:05d}".format(clip_idx), "frames.zip")
        flow_zip_path = os.path.join(self.cfg.FRAME_DIR_PATH, uid, uid+"_" + "{:05d}".format(clip_idx), "flows.zip")

        try:
            frame_zf_fp = zipfile.ZipFile(frame_zip_path, "r")
            flow_zf_fp = zipfile.ZipFile(flow_zip_path, "r")
            exist_frame_list = frame_zf_fp.namelist() # [frame_%010d_%010d.jpg, frame_%010d_%010d.jpg, ...]
            exist_flow_list = flow_zf_fp.namelist() # [u/frame_%010d_%010d.jpg, v/frame_%010d_%010d.jpg, ...]
        except:
            # zipfile does not exist or is corrupted
            return None

        # sample rgb and flows       
        ret = self.sample_frames(info, exist_frame_list, exist_flow_list)
        if ret is None:
            # actual frames number is less than required frame number
            return None

        frame_name_lst, uflow_name_lst, vflow_name_lst = ret

        # load frame content
        frame_lst = self.load_from_zip(frame_name_lst, frame_zf_fp)

        if self.load_flow == "local": # only load flow when needed
            uflow_lst = self.load_from_zip(uflow_name_lst, flow_zf_fp, isflow=True)
            vflow_lst = self.load_from_zip(vflow_name_lst, flow_zf_fp, isflow=True)
        else:
            uflow_lst = None
            vflow_lst = None

        # post process
        frame_zf_fp.close()
        flow_zf_fp.close()

        return frame_lst, uflow_lst, vflow_lst

    def sample_frames(self, info, exist_frame_lst, exist_flow_lst):
        """
            Sample frame and return list of sampled frame and corresponding flow file names

            Args:
            exist_frame_lst: list, exist frames in zip file (might be less than frame number in annotation)

            Return:
            frame_name_lst: list, list of sampled rgb file names
            uflow_name_lst: list, list of sampled uflow file names.
            vflow_name_lst: list, list of sampled vflow file names. 

        """
        start_frame = info["start_frame"]
        end_frame = info["end_frame"]

        exist_frame_lst = sorted( exist_frame_lst, key=lambda x: x.split(".")[0].split("_")[-1] )

        exist_uflow_lst = sorted( [flow for flow in exist_flow_lst if flow.startswith("u")], key=lambda x: x.split(".")[0].split("_")[-1]  )
        exist_vflow_lst = sorted( [flow for flow in exist_flow_lst if flow.startswith("v")], key=lambda x: x.split(".")[0].split("_")[-1]  )

        length = end_frame - start_frame + 1
        if length > len(exist_frame_lst):
            # mismatch frame number between annotation and zip file
            frame_name = exist_frame_lst[-1]
            end_frame  = int( frame_name.split(".")[0].split("_")[-1] )
        
        length = end_frame - start_frame + 1
        if length < self.cfg.NUM_FRAMES:
            return None

        frame_name_lst = []
        uflow_name_lst = []
        vflow_name_lst = []
        frame_idx_lst = self.sample_frames_idx(0, length, self.cfg.NUM_FRAMES)

        for i in range(0, len(frame_idx_lst), 2):
            idx = frame_idx_lst[i]
            frame_name_lst.append(exist_frame_lst[idx])
            frame_name_lst.append(exist_frame_lst[idx+1])

            uflow_name_lst.append(exist_uflow_lst[idx])
            vflow_name_lst.append(exist_vflow_lst[idx])

        return frame_name_lst, uflow_name_lst, vflow_name_lst

    def sample_frames_idx(self, start, end, num_frames):
        """
            return list of indexes of sampled frames, given start and end frame

            start: start frame idx
            end: end frame idx, should be start + clip_length
            num_frames: required number of frames

            e.g.:
                exist frames: [0.jpg, 1.jpg, ..., 100.jpg], then
                start = 0
                end = 0 + 101 = 101

        """
        start = max(0, start)

        intervals = np.linspace(start=start, stop=end, num=int(num_frames) + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))

        frame_idx_lst = [(x[0] + x[1]) // 2 for x in ranges]

        return frame_idx_lst


    def load_from_zip(self, frame_name_lst, zf_fp, isflow=False):
        """
            load frames (rgb or flow) from zip given frame list      
        """

        frame_lst = []
        for frame_name in frame_name_lst:
            img_fp = zf_fp.open(frame_name)
            buffer = io.BytesIO(img_fp.read())
            img = Image.open(buffer) if not isflow else Image.open(buffer).split()[0]
            frame_lst.append(img)

        return frame_lst

    def __getitem__(self, index):

        info = self.package[index]

        msg = f"fail to load frame for video_uid:{info['video_uid']} clip_id:{info['clip_idx']}"

        # load frames and label
        # load until succeed
        # some zip files may contain frames less than required number of frames
        ret = None
        while ret is None:
            ret = self.prepare_clip_frames_flows(info=info)
            if ret is not None:
                break
            print(f"Fail to read frames and flows for video:{info['video_uid']} clip_idx:{info['clip_idx']}, randomly choosing a new ")
            random_idx = random.randint(0, len(self.package)-1)
            info = self.package[random_idx]

        frames, uflows, vflows = ret
        flows = [uflows, vflows] if uflows is not None else None
        
        if frames is None:
            raise ValueError(msg + "," + "frame is None")

        if not self.pretrain:
            frames = [ np.array(frame) for frame in frames ]
            frames = self.data_transform(frames)
            label = info["verb"] if "verb" in self.cfg.task else info["noun"]
            return frames, flows, label
        else:
            frames, flows = self.data_transform((frames, flows))
            frames = frames.view((self.cfg.NUM_FRAMES, 3) + frames.size()[-2:]).transpose(0,1)
            return frames, flows


    def __len__(self):
        return len(self.package)


class Ego4dFhoOscc(Ego4dBase):

    def init_dataset(self):

        """
            init dataset specific paramters from self.cfg
        """

        self.short_side_size = self.cfg.short_side_size
        self.input_size = self.cfg.input_size
        self.save_as_zip = self.cfg.SAVE_AS_ZIP    # save frames in zip file

        # train
        self.repeat_sample = self.cfg.repeat_sample
        # train augmentation
        self.auto_augment = self.cfg.auto_augment
        self.rand_erase_count = self.cfg.rand_erase_count
        self.rand_erase_prob = self.cfg.rand_erase_prob
        self.rand_erase = self.rand_erase_prob > 0
        self.rand_erase_mode = self.cfg.rand_erase_mode
        self.train_interpolation = self.cfg.train_interpolation
        # test
        self.test_spatial_sample = self.cfg.test_spatial_sample

        self.load_flow = self.cfg.load_flow # str, [online, local]

        self.ann_path = ""

    def build_dataset(self):

        """
            build dataset from annotation file
        """

        self.ann_path = os.path.join(self.cfg.ANN_DIR, f'fho_oscc-pnr_{self.mode if self.mode != "test" else self.mode + "_unannotated"}.json')

        ann_err_msg = f"Wrong annotation path provided {self.ann_path}"
        assert os.path.exists(self.ann_path), ann_err_msg
        self.video_dir = self.cfg.VIDEO_DIR_PATH
        assert os.path.exists(self.video_dir), "Wrong videos path provided"
        self.positive_vid_dir = self.cfg.CLIPS_SAVE_PATH
        positive_vid_err_msg = "Wrong positive clips' frame path provided"
        assert os.path.exists(self.positive_vid_dir), positive_vid_err_msg
        self.negative_vid_dir = self.cfg.NO_SC_PATH
        negative_vid_err_msg = "Wrong negative clips' frame path provided"
        assert os.path.exists(self.negative_vid_dir), negative_vid_err_msg

        self.package = dict()
        self.ann_data = json.load(open(self.ann_path, 'r'))["clips"]

        for count, value in enumerate(
            tqdm(self.ann_data, desc='Preparing data')
        ):  

            clip_start_sec = value['parent_start_sec']
            clip_end_sec = value['parent_end_sec']
            clip_start_frame = value['parent_start_frame']
            clip_end_frame = value['parent_end_frame']
            video_id = value['video_uid']
            unique_id = value['unique_id']

            assert count not in self.package.keys()
            if self.mode in ['train', 'val']:
                state_change = value['state_change']
                if "parent_pnr_frame" in value.keys():
                    pnr_frame = value['parent_pnr_frame']
                else:
                    pnr_frame = value["pnr_frame"]
            else:
                state_change = None
                pnr_frame = None

            self.package[count] = {
                'unique_id': unique_id,
                'pnr_frame': pnr_frame,
                # 'state': 0 if not state_change else 1, # NOTE:state_change might be True, False or None
                'clip_start_sec': clip_start_sec,
                'clip_end_sec': clip_end_sec,
                'clip_start_frame': int(clip_start_frame),
                'clip_end_frame': int(clip_end_frame),
                'video_id': video_id,

                "video_path": os.path.join(self.video_dir, video_id+".mp4"),
                "clip_path": os.path.join(self.positive_vid_dir, unique_id) if state_change else os.path.join(self.negative_vid_dir, unique_id)
            }

        if self.mode == "test":
            self.tmp_package = dict()
            # Multiple spatial crop for testing
            for cp in range(self.test_spatial_sample):
                for k, v in self.package.items():
                    self.tmp_package[cp * len(self.package) + k] = {}
                    self.tmp_package[cp * len(self.package) + k].update(v)
                    self.tmp_package[cp * len(self.package) + k]["spatial_crop"] = cp

            self.package = self.tmp_package

        print(f'Number of clips for {self.mode}: {len(self.package)}')
 
    def init_transformation(self):

        self.normalize =  video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
        if self.mode == "train":
            self.data_transform = self.train_transformation

        elif self.mode == 'val':
            self.data_transform = video_transforms.Compose([
                video_transforms.ShorterSideResize(self.short_side_size),
                video_transforms.CenterCrop(size=(self.input_size, self.input_size)),
                volume_transforms.ClipToTensor(),
            ])

        elif self.mode == 'test':
            self.data_transform = video_transforms.Compose([
                video_transforms.ShorterSideResize(self.short_side_size),
                volume_transforms.ClipToTensor(),
            ])

    def __len__(self):
        return len(self.package)
    
    def __getitem__(self, index):

        info = self.package[index]

        try:
            # check existance of clip frames, if some frames are missing then re-extract from video
            self.check_extract_clip_frames(info, save_as_zip=self.save_as_zip)
        except Exception as e:
            print(f"error occurs while reading {info['video_id']}")
            raise e

        # labels is numpy.ndarray (label for oscc task or label for pnr task) 
        clip, labels, _, frame_idx = self.prepare_clip_frames_labels(info, from_zip=self.save_as_zip)

        if self.mode == "train":

            # prepare clip frames and flows at the same time
            clip, flows = self.data_transform(clip, info, frame_idx)
 
            # supports repeat sampling
            if self.repeat_sample > 1:
                clip, labels, flows = list(zip(*[ [ clip, labels, flows ] for _ in range(self.repeat_sample) ] ))

        elif self.mode == "val":
            clip = self.data_transform(clip)
            flows = self.prepare_flow(clip, info, frame_idx)
            clip = self.normalize(clip)

        elif self.mode =="test":
            # support multiple spatial crops
            assert "spatial_crop" in info.keys()

            clip = self.data_transform(clip)
            H, W, C = clip[0].shape
            spatial_step = 1.0 * (max(H, W) - self.short_side_size) \
                                 / (self.test_spatial_sample - 1)
            crop_num = info["spatial_crop"]
            spatial_start = int(crop_num * spatial_step)

            if H >= W:
                clip = [frame[spatial_start:spatial_start + self.short_side_size, :, :] for frame in clip]
            else:
                clip = [frame[:, spatial_start:spatial_start + self.short_side_size, :] for frame in clip]

            flows = self.prepare_flow(clip, info, frame_idx)
            clip = self.normalize(clip)

            return clip, flows, frame_idx

        return clip, flows, labels


    def check_extract_clip_frames(self, info, save_as_zip=False):
        """
            This method is used to extract and save frames for all the 8 seconds
            clips. If the frames are already saved, it does nothing.
        """

        clip_start_frame = info['clip_start_frame']
        clip_end_frame = info['clip_end_frame']
        unique_id = info['unique_id']
        video_path = info["video_path"]
        clip_path = info["clip_path"]
        # if info['pnr_frame'] is not None:
        #     clip_path = os.path.join(self.positive_vid_dir, unique_id)
        # else:
        #     clip_path = os.path.join(self.negative_vid_dir, unique_id)

        if os.path.exists(clip_path):

            if not save_as_zip:

                # The frames for this clip are already saved.
                num_frames = len(os.listdir(clip_path))
                # if frame number does not match annotation file and frames.zip does not exist
                # delete files in $clip_path$ and re-extract frames from video
                if num_frames < (clip_end_frame - clip_start_frame) and \
                                            ( save_as_zip and not os.path.exists(os.path.join(clip_path, "frames.zip"))):
                    # if save_as_zip and not os.path.exists(os.path.join(clip_path, "frames.zip")) :
                    #     print(f"Deleting {clip_path} as it does not have a zip file")
                    # else:
                    print(
                        f'Deleting {clip_path} as it has {num_frames} frames'
                    )

                    os.system(f'rm -r {clip_path}')
                else:
                    return None

            elif os.path.exists(os.path.join(clip_path, "frames.zip")):
                    return None

        print(f'Saving frames for {clip_path}...')
        os.makedirs(clip_path, exist_ok=True)

        start = time.time()

        # list of frame indexes
        frames_list = [ i for i in range(clip_start_frame, clip_end_frame + 1, 1) ]
        clip = self.read_frames_from_video(
            video_path,
            frames_list,
        )

        # desired_shorter_side = 384
        num_saved_frames = 0
        short_side_resize = video_transforms.ShorterSideResize(self.short_side_size)
        
        if save_as_zip:
            zf = ZipFile(f"{clip_path}/frames.zip", mode="a")

        clip = short_side_resize(clip)
        for frame, frame_count in zip(clip, frames_list):
            # save as single frame
            cv2.imwrite(
                os.path.join(
                    clip_path,
                    f'{frame_count}.jpeg'
                ),

                # NOTE: Frames are saved in BGR format
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
            )

            # if need save in a zip, then additionally add to a zip file
            if save_as_zip:
                imgByteArr = io.BytesIO()
                pil = Image.fromarray(frame)
                pil.save(imgByteArr, format="jpeg")
                zf.writestr(f'{frame_count}.jpeg', imgByteArr.getvalue())

            num_saved_frames += 1

        if save_as_zip:
            zf.close()

        print(f'Time taken: {time.time() - start}; {num_saved_frames} '
            f'frames saved; {clip_path}')


    def oscc_sample_frames(self,
        unique_id,
        clip_start_frame,
        clip_end_frame,
        num_frames_required,
        pnr_frame
    ):
        """
 
            Return sampled index of specific number of frames


            After execution, it might return a tuple like:
            ([66, 77, 88, 99, 110, 121, 132, 143, 154, 165, 176, 187, 198, 209, 220, 231], [134, 123, 112, 101, 90, 79, 68, 57, 46, 35, 24, 13, 2, 9, 20, 31])

            First list contains sampled frame index, 
            and the second list contains the relative distances (in frames) between pnr frame and corresponding frame in 1st list.

            if no state change occurs, then elements of the second list are zero
        """ 
        num_frames = clip_end_frame - clip_start_frame
        if num_frames < num_frames_required:
            print(f'Issue: {unique_id}; {num_frames}; {num_frames_required}')
        error_message = "Can\'t sample more frames than there are in the video"
        assert num_frames >= num_frames_required, error_message
        lower_lim = np.floor(num_frames/num_frames_required)
        upper_lim = np.ceil(num_frames/num_frames_required)
        lower_frames = list()
        upper_frames = list()
        # lower_keyframe_candidates_list = list()
        # upper_keyframe_candidates_list = list()
        for frame_count in range(clip_start_frame, clip_end_frame, 1):
            if frame_count % lower_lim == 0:
                lower_frames.append(frame_count)
                # if pnr_frame is not None:
                #     lower_keyframe_candidates_list.append(
                #         np.abs(frame_count - pnr_frame)
                #     )
                # else:
                #     lower_keyframe_candidates_list.append(0.0)
            if frame_count % upper_lim == 0:
                upper_frames.append(frame_count)
                # if pnr_frame is not None:
                #     upper_keyframe_candidates_list.append(
                #         np.abs(frame_count - pnr_frame)
                #     )
                # else:
                #     upper_keyframe_candidates_list.append(0.0)
        if len(upper_frames) < num_frames_required:
            return lower_frames[:num_frames_required]
            # lower_keyframe_candidates_list[:num_frames_required]
        else:
            return upper_frames[:num_frames_required]
            # upper_keyframe_candidates_list[:num_frames_required]


    def pnr_sample_frames(self,
        unique_id,
        clip_start_frame,
        clip_end_frame,
        num_frames_required,
        pnr_frame
    ):

        pass


    def read_frames_from_video(self, video_path, frames_list):
        """
            Code for decoding the video
        """

        cv2.setNumThreads(3)
        # official code where av == 6.0.0
        clip = []
        container = av.open(video_path)
        for frame in _get_frames(
                frames_list,
                container,
                include_audio=False,
                audio_buffer_frames=0
            ):  
            frame = frame.to_rgb().to_ndarray()
            clip.append(frame)

        return clip

    def prepare_clip_frames_labels(self, info, from_zip=False):
        """
            sample specified number of frames given clip
        """
        clip_path = info["clip_path"]
        message = f'Clip path {clip_path} does not exists...'
        assert os.path.isdir(clip_path), message
        # number of frames to be sampled
        num_frames_per_clip= (
            self.cfg.SAMPLING_FPS * self.cfg.CLIP_LEN_SEC
        )
        pnr_frame = info['pnr_frame']

        if self.mode == 'train':
            # Random clipping
            # Randomly choosing the duration of clip (between 5-8 seconds)
            random_length_seconds = np.random.uniform(5, 8)
            random_start_seconds = info['clip_start_sec'] + np.random.uniform(
                8 - random_length_seconds
            )
            random_start_frame = np.floor(
                random_start_seconds * 30
            ).astype(np.int32)
            random_end_seconds = random_start_seconds + random_length_seconds
            if random_end_seconds > info['clip_end_sec']:
                random_end_seconds = info['clip_end_sec']
            random_end_frame = np.floor(
                random_end_seconds * 30
            ).astype(np.int32)
            if pnr_frame is not None:
                keyframe_after_end = pnr_frame > random_end_frame
                keyframe_before_start = pnr_frame < random_start_frame
                if keyframe_after_end:
                    random_end_frame = info['clip_end_frame']
                if keyframe_before_start:
                    random_start_frame = info['clip_start_frame']
        elif self.mode in ['test', 'val']:

            random_start_frame = info['clip_start_frame']
            random_end_frame = info['clip_end_frame']

        if pnr_frame is not None:
            message = (f'Random start frame {random_start_frame} Random end '
                f'frame {random_end_frame} info {info} clip path {clip_path}')
            assert random_start_frame <= pnr_frame <= random_end_frame, message
        else:
            message = (f'Random start frame {random_start_frame} Random end '
                f'frame {random_end_frame} info {info} clip path {clip_path}')
            assert random_start_frame < random_end_frame, message

        labels = None
        if self.cfg.task == "oscc":
            candidate_frame_nums  = self.oscc_sample_frames(
                info['unique_id'],
                random_start_frame,
                random_end_frame,
                num_frames_per_clip,
                pnr_frame
            )

            labels = 1 if pnr_frame is not None else 0

        elif self.cfg.task == "pnr":
            candidate_frame_nums, keyframe_candidates_list = self.pnr_sample_frames(
                info['unique_id'],
                random_start_frame,
                random_end_frame,
                num_frames_per_clip,
                pnr_frame
            )

            if pnr_frame is not None:
                # if state change occurs, then prepare label for state change temporal localization
                keyframe_location = np.argmin(keyframe_candidates_list)
                # use hard labels by default
                hard_labels = np.zeros(len(candidate_frame_nums))
                hard_labels[keyframe_location] = 1
                labels = hard_labels
            else:
                labels = keyframe_candidates_list # all zero

        else:
            raise NotImplementedError(f"Not implemented task:{self.cfg.task} for osccpnr dataset")

        # Start sampling frames given frame index list
        clip = list()
        retry = 5
        if not from_zip:
            # load frames from folder that contains jpeg files
            for frame_num in candidate_frame_nums:
                frame_path = os.path.join(clip_path, f'{frame_num}.jpeg')
                message = f'Failed to find frames after trying {retry} times, {frame_path}; {candidate_frame_nums}; {os.listdir("/".join(frame_path.split("/")[:-1]))}'
                # tolerate missed read
                self.assert_wtolerance(os.path.exists(frame_path), message, retry=retry)
                clip.append(self._load_frame(frame_path))

        else:
            # load frames from zip file
            zip_file_path = os.path.join(clip_path, "frames.zip")
            message = f'Failed to find zip file after trying {retry} times, {zip_file_path}; {candidate_frame_nums};'
            self.assert_wtolerance(os.path.exists(zip_file_path), message, retry=retry)

            _zip_open_retry = 2
            for i in range(_zip_open_retry):
                try:
                    zf = ZipFile( zip_file_path, "r")
                    # if successfully opened zipfile then break loop
                    break
                except zipfile.BadZipFile:
                    os.system(f"rm {zip_file_path}")
                    # if reach maximum times of retrying
                    # then raise error and end program
                    if i == _zip_open_retry - 1:
                        raise Exception(f"Exception occurs while opening zip file: {zip_file_path}. Deleted it...\n\
                                        \rRaw expception: {e} \
                                        ")
                    # else
                    # extract frames again
                    self.check_extract_clip_frames(info, save_as_zip=True)

            for frame_num in candidate_frame_nums:
                try:
                    with zf.open(f"{frame_num}.jpeg") as f:
                        image_data = f.read()
                        pil = Image.open(io.BytesIO(image_data))
                        clip.append(np.array(pil))
                except Exception as e:
                    os.system(f"rm {zip_file_path}")
                    raise Exception(f"Exception occurs while reading frame {frame_num}.jpeg from file {zip_file_path}. Deleted it...\n\
                                    \rRaw expception: {e} \
                                    ")

        # Calculating the effective fps. In other words, the fps after sampling
        # changes when we are randomly clipping and varying the duration of the
        # clip
        final_clip_length = (random_end_frame/30) - (random_start_frame/30)
        effective_fps = num_frames_per_clip / final_clip_length
    
        return clip, np.array(labels), effective_fps, candidate_frame_nums

    def train_transformation(
        self,
        clip,
        info,
        frame_idx,
    ):
        """
            Parameters
            clip: np.ndarray
        
        """
        clip = [transforms.ToTensor()(img) for img in clip]
        clip = torch.stack(clip) # T C H W

        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        clip = self.spatial_sampling(
            clip,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.input_size,
            random_horizontal_flip = True,
            inverse_uniform_sampling = False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        ) # range in [0, 1]

        flows =  self.prepare_flow(clip, info, frame_idx)

        # buffer shape: T C H W
        aug_transform = video_transforms.create_random_augment(
            input_size=(self.input_size, self.input_size),
            auto_augment= self.auto_augment,
            interpolation= self.train_interpolation,
        )
        # print(f"frame raw shape: {buffer[0].shape}") # H, W, C
        clip = [transforms.ToPILImage()(frame) for frame in clip]
        clip = aug_transform(clip) # T, H, W, C
        clip = [transforms.ToTensor()(frame) for frame in clip]

        clip = torch.stack(clip).permute(0, 2, 3, 1) # T C H W -> T H W C
        clip = self.normalize_tensor(
            clip, self.mean, self.std
        ).permute(3, 0, 1, 2) # C T H W 

        if self.rand_erase:
            erase_transform = RandomErasing(
                self.rand_erase_prob,
                mode= self.rand_erase_mode,
                max_count= self.rand_erase_count,
                num_splits= self.rand_erase_count,
                device="cpu",
            )
            clip = clip.permute(1, 0, 2, 3)
            clip = erase_transform(clip)
            clip = clip.permute(1, 0, 2, 3)

        return clip, flows


# class Ego4dFhoScod(Ego4dBase):

#     def init_dataset(self):

#         self.rand_brightness = self.cfg.rand_brightness # by default (0.9, 1.1)
#         self.rand_flip_prob = self.cfg.rand_flip_prob # 0.5 by default 
#         self.input_size = self.cfg.input_size
#         self.short_side_size = self.cfg.short_side_size
#         self.anno_path = os.path.join(self.cfg.ANN_DIR, f"fho_scod_{self.mode}.json")

#         self.mean = torch.tensor(self.mean).view(3,1,1,1)
#         self.std = torch.tensor(self.std).view(3,1,1,1)

#     def build_dataset(self):

#         clips = json.load(open(self.anno_path, "r"))["clips"]
#         self.lst_dict = []
#         image_id = 1

#         for clip in clips:
#             data_dict = {}
#             data_dict["file_name"] = os.path.join(self.cfg.FRAME_DIR_PATH, clip["video_uid"], str(clip["pnr_frame"]["frame_number"])+".jpeg")
#             data_dict["pre_file_name"] = os.path.join(self.cfg.FRAME_DIR_PATH, clip["video_uid"], str(clip["pre_frame"]["frame_number"])+".jpeg")
#             data_dict["post_file_name"] = os.path.join(self.cfg.FRAME_DIR_PATH, clip["video_uid"], str(clip["post_frame"]["frame_number"])+".jpeg")
#             data_dict["height"] = clip["pnr_frame"]["height"]
#             data_dict["width"] = clip["pnr_frame"]["width"]
#             data_dict["image_id"] = image_id
#             data_dict["annotations"] = []

#             if self.mode == "test":
#                 image_id += 1
#                 self.lst_dict.append(data_dict)
#                 continue

#             for bbox in clip["pnr_frame"]["bbox"]:

#                 if bbox["object_type"] == "object_of_change":
#                     data_dict["annotations"].append({
#                         "segmentation": [],
#                         "category_id": 1,
#                         "bbox": [bbox["bbox"]["x"], bbox["bbox"]["y"], bbox["bbox"]["width"], bbox["bbox"]["height"]],
#                         "bbox_mode": 1, # XYWH_ABS
#                         "iscrowd": 0,
#                     })

#             image_id += 1
#             self.lst_dict.append(data_dict)

#     def init_transformation(self):
#         if self.mode == "train":
#             # See "Data Augmentation" tutorial for details usage
#             self.data_transform = detection_transform.AugmentationList([
#                         detection_transform.RandomBrightness(*self.rand_brightness),
#                         detection_transform.RandomFlip(prob=self.rand_flip_prob),
#                         detection_transform.ResizeShortestEdge(self.short_side_size, max_size=1920),
#                         # T.RandomCrop("absolute", (224, 224))
#                         detection_transform.Resize((self.input_size, self.input_size))
#                     ])
#         else:
#             self.data_transform = detection_transform.AugmentationList([
#                         detection_transform.ResizeShortestEdge(self.short_side_size, max_size=1920),
#                         # T.CenterCrop("absolute", (224, 224))
#                         detection_transform.Resize((self.input_size, self.input_size))
#                     ])
 
#     def __getitem__(self, index):

#         dataset_dict = self.lst_dict[index]
#         dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

#         image = self._load_frame(dataset_dict["file_name"])
#         pre_image = self._load_frame(dataset_dict["pre_file_name"])
#         post_image = self._load_frame(dataset_dict["post_file_name"])


#         auginput = detection_transform.AugInput(image)
#         transform = self.data_transform(auginput)
#         image = torch.from_numpy(auginput.image.transpose(2, 0, 1).copy())
#         pre_image = torch.from_numpy(transform.apply_image(pre_image).transpose(2, 0, 1).copy())
#         post_image = torch.from_numpy(transform.apply_image(post_image).transpose(2, 0, 1).copy())

#         # random pick combination of pnr frame with pre/post frame
#         _p = random.random()
#         vit_input = torch.stack([pre_image, image.clone()], dim=0) if _p > 0.5 else torch.stack([image.clone(), post_image], dim=0)
#         vit_input = vit_input.permute(1, 0, 2, 3) / 255.0 # C, T, H, W

#         vit_input = (vit_input - self.mean) / self.std

#         annos = [
#             detection_utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
#             for annotation in dataset_dict.pop("annotations")
#         ]
#         annos = detection_utils.annotations_to_instances(annos, image.shape[1:])

#         return image, vit_input, annos, dataset_dict


#     def visualize(np_rgb_image, xyxy_abs_box, name="detection_vis.png"):
#         # For debugging  

#         """
#             np_rgb_image: numpy.ndarray, in rgb format
#             xyxy_abs_box: list, length is 4
        
#         """

#         from detectron2.utils.visualizer import Visualizer
#         from detectron2.structures import Instances

#         H, W, C = np_rgb_image.shape
#         instance = Instances((H, W))
#         instance.pred_boxes = torch.tensor([xyxy_abs_box]) # the box should be XYXY_ABS
#         instance.scores = torch.tensor([1])
#         instance.pred_classes = torch.tensor([1])

#         vis = Visualizer(np_rgb_image, instance_mode=1)
#         vis_result = vis.draw_instance_predictions(instance)

#         vis_result.save(name)


class Ego4dFhoLTA(Ego4dBase):

    def init_dataset(self):

        self.anno_path = os.path.join(
            self.cfg.ANN_DIR, "fho_lta_{}.json".format(self.mode)
        )

        self.source = self.cfg.FRAME_DIR_PATH
        self.source = os.path.join(self.source , self.mode)

        # self.observation_time_second = self.cfg.observation_time_second
        # self.avail_frame_num = self.observation_time_second * 30
        self.short_side_size = self.cfg.short_side_size
        self.input_size = self.cfg.input_size
        # train
        self.num_frame = self.cfg.NUM_FRAMES

        self.repeat_sample = self.cfg.repeat_sample
        self.input_clip_num = self.cfg.input_clip_num
        self.num_action_predict = self.cfg.num_action_predict
        # test
        self.test_num_clips = self.cfg.test_spatial_crop_num * self.cfg.test_temporal_crop_num

        self.mean = self.cfg.MEAN
        self.std = self.cfg.STD

    def build_dataset(self):

        assert os.path.exists(self.anno_path), "annotation file not found"

        self.package = []
        self.info = {}
        with open(self.anno_path, "r") as anno_fp:
            clips = json.load(anno_fp)["clips"] # list[dict]

        # NOTE not used for now
        verb_freq = {} # compute frequency of each verb label
        noun_freq = {} # compute frequency of each noun label
        for i, clip in enumerate(clips):
            clip_uid = clip["clip_uid"]
            if clip_uid not in self.info.keys():
                self.info[clip_uid] = []

            verb_label = clip.get("verb_label", -1)
            noun_label = clip.get("noun_label", -1)
            if verb_label not in verb_freq:
                verb_freq[verb_label] = 0
            if noun_label not in noun_freq:
                noun_freq[noun_label] = 0

            verb_freq[verb_label] += 1
            noun_freq[noun_label] += 1

            self.info[clip_uid].append({
                "video_uid": clip["video_uid"],
                "clip_uid": clip["clip_uid"],

                "clip_parent_start_frame": clip["clip_parent_start_frame"],
                "clip_parent_end_frame": clip["clip_parent_end_frame"],
                "action_clip_start_frame": clip["action_clip_start_frame"],
                "action_clip_end_frame": clip["action_clip_end_frame"],

                "action_idx": clip["action_idx"],
                "verb_label": verb_label,
                "noun_label": noun_label,
            })

        for clip_uid in self.info.keys():
            clips = self.info[clip_uid]

            for i in range(len(clips) - self.input_clip_num - self.num_action_predict):
                self.package.append(
                    {
                        "input_clips": clips[i:i+self.input_clip_num],
                        "forecast_clips": clips[i+self.input_clip_num: i+self.input_clip_num+self.num_action_predict ]
                    }
                )

    def init_transformation(self):

        if self.mode == "train":
            # See "Data Augmentation" tutorial for details usage
            self.data_transform = self.train_transformation

        elif self.mode == 'val':
            self.data_transform = video_transforms.Compose([
                video_transforms.ShorterSideResize(self.short_side_size),
                video_transforms.CenterCrop(size=(self.input_size, self.input_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=self.mean, std=self.std),
            ])

        elif 'test' in self.mode:
            self.data_transform = video_transforms.Compose([
                video_transforms.ShorterSideResize(self.short_side_size),
                video_transforms.CenterCrop(size=(self.input_size, self.input_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=self.mean, std=self.std),
            ])

    def train_transformation(self, clip_np):

        spatial_crop = video_transforms.Compose([
                video_transforms.ShorterSideResize(self.short_side_size),
                video_transforms.RandomCrop(size=(self.input_size, self.input_size)),
                video_transforms.RandomHorizontalFlip(),
        ])

        rand_aug = video_transforms.create_random_augment(
            input_size=(self.input_size, self.input_size),
            auto_augment= "rand-m7-n4-mstd0.5-inc1",
            interpolation= "bicubic",
        )

        to_tensor = video_transforms.Compose([
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=self.mean, std=self.std),
        ])

        clip_np = spatial_crop(clip_np) # numpy
        clip_pil = [Image.fromarray(_clip) for _clip in clip_np]
        clip_pil = rand_aug(clip_pil)
        clip_tensor = to_tensor(clip_pil)

        return clip_tensor

    def sample_frames_from_clip(self, clip_info):
        parent_st_frame = clip_info["clip_parent_start_frame"]
        clip_st_frame = clip_info["action_clip_start_frame"]
        clip_end_frame = clip_info["action_clip_end_frame"]

        st_frame = parent_st_frame + clip_st_frame
        end_frame = parent_st_frame + clip_end_frame

        intervals = self.sample_frame(st_frame, end_frame, strategy="tsn")

        frames = []
        with zipfile.ZipFile(os.path.join(self.source, clip_info["clip_uid"], clip_info["clip_uid"]+"_{:05d}.zip".format(clip_info["action_idx"]) ) ,"r") as zipfp:
            for frame_idx in intervals:
                frame_fp = zipfp.open("{:010d}.jpg".format(frame_idx))
                pil = Image.open( frame_fp)
                frame = np.array(pil)
                frames.append(frame)

        return frames, intervals

    def __getitem__(self, index):

        info = self.package[index]

        input_clips = info["input_clips"]
        forecast_clips = info["forecast_clips"]

        frames = [] # list of numpy array
        frame_idx_lst = []
        target = [] # list of int

        for clip_info in input_clips:
            sampled_frames, frame_index = self.sample_frames_from_clip(clip_info)
            frames.extend(sampled_frames)
            frame_idx_lst.extend(frame_index)

        target = [input_clips[-1]["verb_label"]] if self.cfg.task == "lta_verb" else input_clips[-1]["noun_label"]
        for clip_info in forecast_clips:
            target.append( clip_info["verb_label"] if self.cfg.task == "lta_verb" else clip_info["noun_label"])

        frames = self.data_transform(frames)
        flows = None
 
        return frames, flows, target

    def __len__(self):
        return len(self.package)


class Ego4dFhoHands(Ego4dBase):
    """
        Hands prediction
    
    """
    def init_dataset(self):
        
        self.anno_path = os.path.join(
            self.cfg.ANN_DIR, "fho_hands_{}.json".format(self.mode)
        )

        self.source = self.cfg.FRAME_DIR_PATH
        self.source = os.path.join(self.source , self.mode)
        self.max_observation_time = 4
        self.max_observation_frame_num = 30 * self.max_observation_time
        # train
        self.repeat_sample = self.cfg.repeat_sample
        
        self.auto_augment = self.cfg.auto_augment
        self.rand_erase_count = self.cfg.rand_erase_count
        self.rand_erase_prob = self.cfg.rand_erase_prob
        self.rand_erase = self.rand_erase_prob > 0
        self.rand_erase_mode = self.cfg.rand_erase_mode
        self.train_interpolation = self.cfg.train_interpolation

        # test
        self.test_num_clips = self.cfg.test_spatial_crop_num * self.cfg.test_temporal_crop_num

        self.short_side_size = self.cfg.short_side_size
        self.input_size = self.cfg.input_size
        # train
        self.num_frame = self.cfg.NUM_FRAMES

    def build_dataset(self):

        """
            Construct the video loader.
        """

        assert os.path.exists(self.anno_path), "annotation file not found"

        self.package = []
        frame_types2index = {
            "pre_45": 0, 
            "pre_30": 4,
            "pre_15": 8,
            "pre_frame": 12,
            "contact_frame": 16,
        }

        with open(self.anno_path, "r") as anno_fp:
            clips = json.load(anno_fp)["clips"] # list[dict]

        for clip in clips:

            clip_meta = {
                "clip_id": clip['clip_id'],
                "clip_uid": clip['clip_uid'],
                "video_uid": clip['video_uid'],

                "spatial_temporal_index": 0,
            }

            for i, annot in enumerate(clip['frames']):
                
                if self.mode == "test":
                    
                    pre_45_frame = annot["pre_45"]["frame"]
                    pre_frame = annot["pre_frame"]["frame"] if "pre_frame" in annot else -1
                    st = max(0, pre_45_frame - self.max_observation_frame_num)
                    end = pre_frame + 30 if pre_frame == -1 else pre_45_frame + 45 + 30
                else:
                    st = annot["action_start_frame"]
                    end = annot["action_end_frame"]

                clip_name = str(clip['clip_uid']) + "_{:05d}.zip".format(i)
                clip_meta.update({
                    # distinguish segments in clip by index of frame pre_45
                    "idx": i,
                    "clip_name": clip_name,
                    "clip_path": os.path.join(self.source, clip["clip_uid"], clip_name),
                    "action_start_frame": st,
                    "action_end_frame": end,
                })

                # placeholder for the 1x20 hand gt vector (padd zero when GT is not available)
                # 5 frames have the following order: pre_45, pre_40, pre_15, pre, contact
                # GT for each frames has the following order: left_x,left_y,right_x,right_y
                label= [0.0]*20
                label_mask = [0.0]*20
                for frame_type, frame_annot in annot.items():

                    if frame_type in frame_types2index.keys():
                        # if len(frame_annot)==2:
                        #     print(frame_annot)
                        #     continue
                        lh_idx = frame_types2index[frame_type]
                        rh_idx = frame_types2index[frame_type] + 2

                        frame_gt = frame_annot['boxes']

                        for single_hand in frame_gt:
                            if 'left_hand' in single_hand:
                                label_mask[lh_idx]=1.0
                                label_mask[lh_idx+1]=1.0
                                label[lh_idx]= single_hand['left_hand'][0]
                                label[lh_idx+1]= single_hand['left_hand'][1]
                            if 'right_hand' in single_hand:
                                label_mask[rh_idx]=1.0
                                label_mask[rh_idx+1]=1.0
                                label[rh_idx]= single_hand['right_hand'][0]
                                label[rh_idx+1]= single_hand['right_hand'][1]   

                clip_meta.update({
                    "label": label,
                    "label_mask": label_mask,
                })

                if self.mode == "test":
                    for idx in range(self.test_num_clips):
                        clip_meta["spatial_temporal_index"] = idx
                        self.package.append(clip_meta)
                else:
                    self.package.append(clip_meta)

    def init_transformation(self):

        if self.mode == "train":
            # See "Data Augmentation" tutorial for details usage
            self.data_transform = self.train_transform
        elif self.mode == 'val':
            self.data_transform = video_transforms.Compose([
                video_transforms.ShorterSideResize(self.short_side_size),
                video_transforms.CenterCrop(size=(self.input_size, self.input_size)),
                volume_transforms.ClipToTensor(),
            ])

        elif self.mode == 'test':
            self.data_transform = video_transforms.Compose([
                video_transforms.ShorterSideResize(self.short_side_size),
                volume_transforms.ClipToTensor(),
            ])


    def sample_frames_from_clip(self, clip_info):

        clip_path = clip_info["clip_path"]
        st_frame = clip_info["action_start_frame"]
        end_frame = clip_info["action_end_frame"]

        intervals = self.sample_frame(st_frame, end_frame, strategy="tsn")

        frames = []
        with zipfile.ZipFile(clip_path ,"r") as zipfp:
            for frame_idx in intervals:
                try:
                    frame_fp = zipfp.open("{:010d}.jpg".format(frame_idx))
                except:
                    print("error when reading image from zipfile")
                    print(zipfp.namelist())
                    print(clip_path, clip_info, "{:010d}.jpg".format(frame_idx))

                pil = Image.open(frame_fp)
                frame = np.array(pil)
                frames.append(frame)

        return frames, intervals

    def __getitem__(self, index):

        clip_info = self.package[index]
    
        label = clip_info["label"]
        label = torch.tensor(label)
        mask = clip_info["label_mask"]
        mask = torch.tensor(mask)

        frames, intervals = self.sample_frames_from_clip(clip_info) # list[PIL]
        frames = self.data_transform(frames) # C, T, H, W

        flows = None
        if self.repeat_sample > 1:
            frames = [frames for i in range(self.repeat_sample)]
            flows = [flows for i in range(self.repeat_sample)]
            label = [label for i in range(self.repeat_sample)]
            mask = [mask for i in range(self.repeat_sample)]

        return frames, flows, label, mask

    def __len__(self):
        return len(self.package)

    def train_transform(self, clip):
        """
            Args:
                clip: list[PIL.Image, numpy.ndarray]
                info: dict, contains information of current clip
                frame_idx: list, frame indexes of current clip
        """

        # clip = [transforms.ToTensor()(img) for img in clip]
        # clip = torch.stack(clip, dim=0) # T C H W

        # scl, asp = (
        #     [0.08, 1.0],
        #     [0.75, 1.3333],
        # )

        # clip = self.spatial_sampling(
        #     clip,
        #     spatial_idx=-1,
        #     min_scale=256,
        #     max_scale=320,
        #     crop_size=self.input_size,
        #     random_horizontal_flip = True,
        #     inverse_uniform_sampling = False,
        #     aspect_ratio=asp,
        #     scale=scl,
        #     motion_shift=False
        # ) # T C H W, range in [0, 1]

        # # flows =  self.prepare_flow(clip, info, frame_idx)

        # clip = clip.permute(0, 2, 3, 1) # T C H W -> T H W C
        # clip = self.normalize_tensor(
        #     clip, self.mean, self.std
        # ).permute(3, 0, 1, 2) # C T H W 

        clip = [transforms.ToTensor()(img) for img in clip]
        clip = torch.stack(clip) # T C H W

        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        clip = self.spatial_sampling(
            clip,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.input_size,
            random_horizontal_flip = True,
            inverse_uniform_sampling = False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        ) # range in [0, 1]

        # buffer shape: T C H W
        aug_transform = video_transforms.create_random_augment(
            input_size=(self.input_size, self.input_size),
            auto_augment= self.auto_augment,
            interpolation= self.train_interpolation,
        )
        # print(f"frame raw shape: {buffer[0].shape}") # H, W, C
        clip = [transforms.ToPILImage()(frame) for frame in clip]
        clip = aug_transform(clip) # T, H, W, C
        clip = [transforms.ToTensor()(frame) for frame in clip]

        clip = torch.stack(clip).permute(0, 2, 3, 1) # T C H W -> T H W C
        clip = self.normalize_tensor(
            clip, self.mean, self.std
        ).permute(3, 0, 1, 2) # C T H W 

        if self.rand_erase:
            erase_transform = RandomErasing(
                self.rand_erase_prob,
                mode= self.rand_erase_mode,
                max_count= self.rand_erase_count,
                num_splits= self.rand_erase_count,
                device="cpu",
            )
            clip = clip.permute(1, 0, 2, 3)
            clip = erase_transform(clip)
            clip = clip.permute(1, 0, 2, 3)

        return clip


def get_basic_config_for(task):
    """
        Return basic configuration for each task
        Args:
            task: str, name of task
        Return:
            cfg: argparse.Namespace
    """
    if task == "oscc-pnr":
         cfg = {
            # path used
            "ANN_DIR": "/data/shared/ssvl/ego4d/v1/annotations",
            "VIDEO_DIR_PATH": "/data/shared/ssvl/ego4d/v1/full_scale",
            "CLIPS_SAVE_PATH": "/data/shared/ssvl/ego4d/v1/pos",
            "NO_SC_PATH": "/data/shared/ssvl/ego4d/v1/neg",
            "SAVE_AS_ZIP": True,
            "CLIP_LEN_SEC": 8,
            "SAMPLING_FPS": 2, 
            "MEAN":[0.485, 0.456, 0.406],
            "STD": [0.229, 0.224, 0.225],
            # "FRAME_FORMAT": "{:10d}.jpeg",

            "short_side_size": 256,
            "input_size": 224,
            # train
            "repeat_sample": 1,
            # train augmentation
            "auto_augment": 'rand-m7-n4-mstd0.5-inc1',
            "rand_erase_count": 1,
            "rand_erase_prob": 0.25,
            "rand_erase_mode": "pixel",
            "train_interpolation": "bicubic",
            # test
            "test_spatial_sample": 3,

            "task": "fho_oscc_tl",
            "load_flow": "none", # [online, local, none] 
        }

    elif task == "scod":

        cfg = {
            "ANN_DIR": "/data/shared/ssvl/ego4d/v1/annotations",
            "FRAME_DIR_PATH": "/data/shared/ssvl/ego4d/v1/fho_scod/pre_pnr_post_frames",
            "SAVE_AS_ZIP": True,
            "MEAN":[0.485, 0.456, 0.406],
            "STD": [0.229, 0.224, 0.225],
            "FRAME_FORMAT": "{:010d}.jpg",  # image frame format

            "task": "fho_scod",

            "rand_brightness": (0.9, 1.1),
            "rand_flip_prob": 0.5,
            "input_size" : 224,
            "short_side_size" : 256,
        }

    elif task == "hands":
        cfg = {
            "ANN_DIR": "/data/shared/ssvl/ego4d/v1/annotations",
            "FRAME_DIR_PATH": "/data/shared/ssvl/ego4d/v1/fho_hands",
            "MEAN":[0.485, 0.456, 0.406],
            "STD": [0.229, 0.224, 0.225],
            "FRAME_FORMAT": "{:010d}.jpg",  # image frame format

            "observation_time_second": 4,  # observation time of baseline method is 2s

            "repeat_sample": 1, 
            "test_spatial_crop_num": 3,
            "test_temporal_crop_num": 1,

            "task": "fho_hands"
        }

    elif task == "egoclip":

        cfg = {
            "ANN_DIR": "/mnt/shuang/Data/ego4d/data/v1/annotations",
            "FRAME_DIR_PATH": "/mnt/shuang/Data/ego4d/preprocessed_data/egoclip",
            "NUM_FRAMES": 16,

            "MEAN":[0.485, 0.456, 0.406],
            "STD": [0.229, 0.224, 0.225],

            "FRAME_FORMAT": "frame_{:10d}_{:010d}.jpg",  # image frame format

            "repeat_sample": 1, 
            "test_spatial_crop_num": 3,
            "test_temporal_crop_num": 1,

            "load_flow": "local",

            "task": "fho_hands"
        }

    elif task == "lta":

        cfg = {
            "ANN_DIR": "/data/shared/ssvl/ego4d/v1/annotations",
            "FRAME_DIR_PATH": "/data/shared/ssvl/ego4d/v1/fho_lta/test",
            "NUM_FRAMES": 16,

            "MEAN":[0.485, 0.456, 0.406],
            "STD": [0.229, 0.224, 0.225],

            "FRAME_FORMAT": "{:010d}.jpg",  # image frame format

            "repeat_sample": 1, 
            "input_clip_num": 1,
            "num_action_predict": 20,
            "short_side_size": 256,
            "input_size": 224,

            "test_spatial_crop_num": 3,
            "test_temporal_crop_num": 1,

            "load_flow": "local",
            "task": "lta"
        }
    return Namespace(**cfg)


if __name__ == "__main__":
    from argparse import Namespace


    cfg = get_basic_config_for("lta")
    kwargs = {
        "mode": "test_unannotated",
        "cfg": cfg,
        "pretrain": True,
        "flow_extractor": None,
        "pretrain_transform": None,
    }

    debug_functions(Ego4dFhoLTA, **kwargs)