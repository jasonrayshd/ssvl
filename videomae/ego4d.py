"""
Jiachen Lei, 2022.05.19

Reference 
https://github.com/EGO4D/hands-and-objects/tree/main/state-change-localization-classification/i3d-resnet50

"""

import os
import json
import time

import av
import cv2

from PIL import Image
from tblib import Frame
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
import video_transforms as video_transforms 
import volume_transforms as volume_transforms
from random_erasing import RandomErasing
from ego4d_trim import _get_frames

import io
import decord
import zipfile
from zipfile import ZipFile

"""
Used configuration:      Jiachen, 2022.05.19

DATA.ANN_DIR
DATA.VIDEO_DIR_PATH
DATA.CLIPS_SAVE_PATH
DATA.NO_SC_PATH

DATA.CLIP_LEN_SEC
DATA.CROP_SIZE: width and height of resized result
DATA.SAMPLING_FPS

"""

##### copied from kinetics.py, edited by Jiachen Lei
def tensor_normalize(tensor, mean, std):
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
            dimension is `num frames` x `height` x `width` x `channel`.
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
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames

#####

class StateChangeDetectionAndKeyframeLocalisation(torch.utils.data.Dataset):
    """
    Data loader for state change detection and key-frame localization.
    This data loader assumes that the user has alredy extracted the frames from
    all the videos using the `train.json`, `test_unnotated.json`, and
    'val.json' provided.
    """
    def __init__(self, cfg, mode, args, pretrain=False, transform=None, flow_extractor=None):
        assert mode in [
            'train',
            'val',
            'test'
        ], "Split `{}` not supported for Keyframe detection.".format(mode)
        self.mode = mode
        self.cfg = cfg
        self.args = args
        self.pretrain = pretrain
        self.pretrain_transform = transform
        self.crop_size = args.input_size

        self.flowExt = flow_extractor
        self.save_as_zip = cfg.DATA.SAVE_AS_ZIP    # save frames in zip file

        # Added by Jiachen
        self.rand_erase = False
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        # commented by Jiachen for debugging
        self.ann_path = os.path.join(cfg.DATA.ANN_DIR, f'fho_oscc-pnr_{self.mode if self.mode != "test" else self.mode + "_unannotated"}.json')

        # NOTE line below should be modified when start finetuning
        # self.ann_path = os.path.join(cfg.DATA.ANN_DIR, f'partial_train.json')

        ann_err_msg = f"Wrong annotation path provided {self.ann_path}"
        assert os.path.exists(self.ann_path), ann_err_msg
        self.video_dir = self.cfg.DATA.VIDEO_DIR_PATH
        assert os.path.exists(self.video_dir), "Wrong videos path provided"
        self.positive_vid_dir = self.cfg.DATA.CLIPS_SAVE_PATH
        positive_vid_err_msg = "Wrong positive clips' frame path provided"
        assert os.path.exists(self.positive_vid_dir), positive_vid_err_msg
        self.negative_vid_dir = self.cfg.DATA.NO_SC_PATH
        negative_vid_err_msg = "Wrong negative clips' frame path provided"
        assert os.path.exists(self.negative_vid_dir), negative_vid_err_msg

        self._construct_loader()
        self._init_trans_for_mode()

    def _init_trans_for_mode(self):

        self.normalize =  video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])

        if self.mode == "train":
            self.data_transform = self._aug_frame

        elif self.mode == 'val':
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.args.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
            ])

        elif self.mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(self.args.short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
            ])


    def _construct_loader(self):
        self.package = dict()
        # NOTE ann_data should be list of clips or dictionary of which key "clips" contains a list of clips
        self.ann_data = json.load(open(self.ann_path, 'r'))
        if isinstance(self.ann_data, dict):
            self.ann_data = self.ann_data["clips"]

        for count, value in enumerate(
            tqdm(self.ann_data, desc='Preparing data')
        ):  
            # edited by Jiachen Lei
            # read by clips for debugging or read by full_scale video for training
            if self.cfg.DATA.READ_BY_CLIPS:
                try:
                    clip_start_sec = value['clip_start_sec']
                    clip_end_sec = value['clip_end_sec']
                    clip_start_frame = value['clip_start_frame']
                    clip_end_frame = value['clip_end_frame']
                    video_id = value['clip_uid']
                except KeyError as e:
                    # in debug, part of all clips are downloaded
                    continue
            else:
                # Codes below are offcial implementation
                clip_start_sec = value['parent_start_sec']
                clip_end_sec = value['parent_end_sec']
                clip_start_frame = value['parent_start_frame']
                clip_end_frame = value['parent_end_frame']
                video_id = value['video_uid']

            unique_id = value['unique_id']
            assert count not in self.package.keys()
            if self.mode in ['train', 'val']:
                state_change = value['state_change']
                # edited by Jiachen Lei
                if self.cfg.DATA.READ_BY_CLIPS:
                    if "clip_pnr_frame" in value.keys():
                        pnr_frame = value['clip_pnr_frame']
                    else:
                        pnr_frame = value["pnr_frame"]
                else:
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
                'state': 0 if not state_change else 1, # NOTE:state_change might be True, False or None
                'clip_start_sec': clip_start_sec,
                'clip_end_sec': clip_end_sec,
                'clip_start_frame': int(clip_start_frame),
                'clip_end_frame': int(clip_end_frame),
                'video_id': video_id,
            }

        if self.mode == "test":
            self.tmp_package = dict()
            for cp in range(self.args.test_num_crop):
                for k, v in self.package.items():
                    self.tmp_package[cp * len(self.package) + k] = {}
                    self.tmp_package[cp * len(self.package) + k].update(v)
                    self.tmp_package[cp * len(self.package) + k]["crop"] = cp

            self.package = self.tmp_package

        print(f'Number of clips for {self.mode}: {len(self.package)}')

    def __len__(self):
        return len(self.package)

    def __getitem__(self, index):
        info = self.package[index]
        state = info['state']                                                                   # Indiate whether state change occurs in the clip 
        try:
            self._extract_clip_frames(info, save_as_zip=self.save_as_zip)                       # Extract frames from videos, if frames not exist
        except Exception as e:
            print(f"error occurs while reading {info['video_id']}")
            raise e

        frames, labels, _, frame_idx = self._sample_frames_gen_labels(info, from_zip=self.save_as_zip)     # Sample given number of frames
                                                                                            # Return contains two numpy.ndarray that contains sampeld frames and label
                                                                                            # original transformations include: 
                                                                                            #     (1) random start frame
                                                                                            #     (2) resize to square (was commented)
        # prepare label for state change localization
        if labels.sum() != 0:
            labels = labels.nonzero()[0].item()
        else:
            labels = len(frames)

        clip_len = info['clip_end_sec'] - info['clip_start_sec']
        clip_frame = info['clip_end_frame'] - info['clip_start_frame'] + 1
        fps = clip_frame / clip_len

        if self.mode == "train":
            frame_list = []
            label_list = []
            state_list = []
            flow_list = []
            for _ in range(self.args.num_sample):

                new_frames, flows = self.data_transform(frames)

                frame_list.append(new_frames)
                label_list.append(labels)
                state_list.append(state)
                flow_list.append(flows)

            if len(frame_list) == 1:
                frames = frame_list[0]
                labels = label_list[0]
                state = state_list[0]
                flows = flow_list[0] if len(flow_list) > 0 else None
            else:
                frames = frame_list
                labels = label_list
                state = state_list
                flows = flow_list

        elif self.mode == "val":
            frames = self.data_transform(frames)
            flows = None
            if self.args.flow_mode == "online":
                flows = self.flowExt(frames)

            frames = self.normalize(frames)

        elif self.mode =="test":
            assert "crop" in info.keys()

            frames = self.data_resize(frames)
            H, W, C = frames[0].shape
            spatial_step = 1.0 * (max(H, W) - self.args.short_side_size) \
                                 / (self.args.test_num_crop - 1)
            crop_num = info["crop"]
            spatial_start = int(crop_num * spatial_step)

            if H >= W:
                frames = [frame[spatial_start:spatial_start + self.args.short_side_size, :, :] for frame in frames]
            else:
                frames = [frame[:, spatial_start:spatial_start + self.args.short_side_size, :] for frame in frames]

            frames = self.data_transform(frames)
    
            flows = None
            if self.args.flow_mode == "online":
                flows = self.flowExt(frames)

            frames = self.normalize(frames)

            # edited by jiachen
            # Ego4d challenge is currently undergoing, annotations of state change test set are not published.
            return frames, flows, info, frame_idx


        if self.pretrain:
            # left for future development
            pass
        else:
            if self.args.nb_classes == -1:
                GT =  [labels, state]
            elif self.args.nb_classes == 2:
                GT = state
            else:
                GT = labels

            return frames, GT, flows, fps, info

    # _aug_frame edited by Jiachen Lei
    def _aug_frame(
        self,
        buffer,
    ):
        """
            Parameters
            buffer: np.ndarray
        
        """
        # aug_transform = video_transforms.create_random_augment(
        #     input_size=(self.crop_size, self.crop_size),
        #     auto_augment= self.args.aa,
        #     interpolation= self.args.train_interpolation,
        # )
        # print(f"frame raw shape: {buffer[0].shape}") # H, W, C
        # buffer = [
        #     transforms.ToPILImage()(frame) for frame in buffer
        # ]

        # buffer = aug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        # print(buffer.shape)
        # T C H W -> T H W C
        # buffer = buffer.permute(0, 2, 3, 1)
        # print(buffer.shape)
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip = False if  self.args.data_set == 'SSV2' else True ,
            inverse_uniform_sampling = False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

        flows = None
        if self.args.flow_mode == "online":
            assert self.flowExt is not None, "flow extractor is None"
            flows = self.flowExt(buffer)

        buffer = buffer.permute(0, 2, 3, 1) # T H W C 
        # print(buffer.shape)
        # T H W C 
        buffer = tensor_normalize(
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ).permute(0, 3, 1, 2) # T C H W

        # print(buffer.shape)
        buffer = transforms.ColorJitter(brightness=0, saturation=0, hue=0)(buffer).permute(1, 0, 2, 3)

        # if self.rand_erase:
        #     erase_transform = RandomErasing(
        #         self.args.reprob,
        #         mode= self.args.remode,
        #         max_count= self.args.recount,
        #         num_splits= self.args.recount,
        #         device="cpu",
        #     )
        #     buffer = buffer.permute(1, 0, 2, 3)
        #     buffer = erase_transform(buffer)
        #     buffer = buffer.permute(1, 0, 2, 3)

        return buffer, flows


    def _extract_clip_frames(self, info, save_as_zip=False):
        """
        This method is used to extract and save frames for all the 8 seconds
        clips. If the frames are already saved, it does nothing.
        """
        clip_start_frame = info['clip_start_frame']
        clip_end_frame = info['clip_end_frame']
        unique_id = info['unique_id']
        video_path = os.path.join(
            self.video_dir,
            info['video_id']+".mp4",
        )

        if info['pnr_frame'] is not None:
            clip_save_path = os.path.join(self.positive_vid_dir, unique_id)
        else:
            clip_save_path = os.path.join(self.negative_vid_dir, unique_id)
        # We can do do this fps for canonical data is 30.
        num_frames_per_video = 30 * self.cfg.DATA.CLIP_LEN_SEC
        if os.path.exists(clip_save_path):
            # The frames for this clip are already saved.
            num_frames = len(os.listdir(clip_save_path))
            if num_frames < (clip_end_frame - clip_start_frame) or \
                                        ( save_as_zip and not os.path.exists(os.path.join(clip_save_path, "frames.zip"))):
                if save_as_zip and not os.path.exists(os.path.join(clip_save_path, "frames.zip")) :
                    print(f"Deleting {clip_save_path} as it does not have a zip file")
                else:
                    print(
                        f'Deleting {clip_save_path} as it has {num_frames} frames'
                    )

                os.system(f'rm -r {clip_save_path}')
            else:
                return None

        print(f'Saving frames for {clip_save_path}...')
        os.makedirs(clip_save_path)
        start = time.time()
        # We need to save the frames for this clip.
        frames_list = [
            i for i in range(clip_start_frame, clip_end_frame + 1, 1)
        ]
        frames = self.get_frames_for(
            video_path,
            frames_list,
        )

        desired_shorter_side = 384
        num_saved_frames = 0

        if save_as_zip:
            zf = ZipFile(f"{clip_save_path}/frames.zip", mode="a")

        for frame, frame_count in zip(frames, frames_list):
            original_height, original_width, _ = frame.shape
            if original_height < original_width:
                # Height is the shorter side
                new_height = desired_shorter_side
                new_width = np.round(
                    original_width*(desired_shorter_side/original_height)
                ).astype(np.int32)
            elif original_height > original_width:
                # Width is the shorter side
                new_width = desired_shorter_side
                new_height = np.round(
                    original_height*(desired_shorter_side/original_width)
                ).astype(np.int32)
            else:
                # Both are the same
                new_height = desired_shorter_side
                new_width = desired_shorter_side
            assert np.isclose(
                new_width/new_height,
                original_width/original_height,
                0.01
            )
            frame = cv2.resize(
                frame,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )
            cv2.imwrite(
                os.path.join(
                    clip_save_path,
                    f'{frame_count}.jpeg'
                ),

                # NOTE: Frames are saved in BGR format
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
            )

            if save_as_zip:
                imgByteArr = io.BytesIO()
                pil = Image.fromarray(frame)
                pil.save(imgByteArr, format="jpeg")
                zf.writestr(f'{frame_count}.jpeg', imgByteArr.getvalue())

            num_saved_frames += 1

        if save_as_zip:
            zf.close()

        print(f'Time taken: {time.time() - start}; {num_saved_frames} '
            f'frames saved; {clip_save_path}')
        return None

    def _sample_frames(
        self,
        unique_id,
        clip_start_frame,
        clip_end_frame,
        num_frames_required,
        pnr_frame
    ):
        """
            Edited by Jiachen
            
            Return sampled index of specific number of frames

            e.g.:
            _sample_frames(
                unique_id="",
                clip_start_frame=np.array(((8-random_length_seconds)*30)).astype(np.int32),
                clip_end_frame=np.array(240).astype(np.int32),
                num_frames_required=16,
                pnr_frame=200,
            )

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
        lower_keyframe_candidates_list = list()
        upper_keyframe_candidates_list = list()
        for frame_count in range(clip_start_frame, clip_end_frame, 1):
            if frame_count % lower_lim == 0:
                lower_frames.append(frame_count)
                if pnr_frame is not None:
                    lower_keyframe_candidates_list.append(
                        np.abs(frame_count - pnr_frame)
                    )
                else:
                    lower_keyframe_candidates_list.append(0.0)
            if frame_count % upper_lim == 0:
                upper_frames.append(frame_count)
                if pnr_frame is not None:
                    upper_keyframe_candidates_list.append(
                        np.abs(frame_count - pnr_frame)
                    )
                else:
                    upper_keyframe_candidates_list.append(0.0)
        if len(upper_frames) < num_frames_required:
            return (
                lower_frames[:num_frames_required],
                lower_keyframe_candidates_list[:num_frames_required]
            )
        return (
            upper_frames[:num_frames_required],
            upper_keyframe_candidates_list[:num_frames_required]
        )

    def _load_frame(self, frame_path):
        """
        This method is used to read a frame and do some pre-processing.

        Args:
            frame_path (str): Path to the frame
        
        Returns:
            frames (ndarray): Image as a numpy array
        """
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Commented by Jiachen Lei
        # frame = cv2.resize(frame,(
        #     self.cfg.DATA.CROP_SIZE,
        #     self.cfg.DATA.CROP_SIZE
        # ))
        # frame = np.expand_dims(frame, axis=0).astype(np.float32)

        return frame

    def _sample_frames_gen_labels(self, info, from_zip=False):
        if info['pnr_frame'] is not None:
            clip_path = os.path.join(
                self.positive_vid_dir,
                info['unique_id']
            )
        else:
            # Clip path for clips with no state change
            clip_path = os.path.join(
                self.negative_vid_dir,
                info['unique_id']
            )
        message = f'Clip path {clip_path} does not exists...'
        assert os.path.isdir(clip_path), message

        num_frames_per_video = (
            self.cfg.DATA.SAMPLING_FPS * self.cfg.DATA.CLIP_LEN_SEC
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

        candidate_frame_nums, keyframe_candidates_list = self._sample_frames(
            info['unique_id'],
            random_start_frame,
            random_end_frame,
            num_frames_per_video,
            pnr_frame
        )

        # Start sampling frames given frame index list
        frames = list()
        retry = 5
        if not from_zip:
            # load frames from folder that contains jpeg files
            for frame_num in candidate_frame_nums:
                frame_path = os.path.join(clip_path, f'{frame_num}.jpeg')
                message = f'Failed to find frames after trying {retry} times, {frame_path}; {candidate_frame_nums}; {os.listdir("/".join(frame_path.split("/")[:-1]))}'
                # tolerate missed read
                self.assert_exist_wtolerance(frame_path, message, retry=retry)
                frames.append(self._load_frame(frame_path))

        else:
            # load frames from zip file
            zip_file_path = os.path.join(clip_path, "frames.zip")
            message = f'Failed to find zip file after trying {retry} times, {zip_file_path}; {candidate_frame_nums};'
            self.assert_exist_wtolerance(zip_file_path, message, retry=retry)

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
                    self._extract_clip_frames(info, save_as_zip=True)

            # zf.printdir()
            for frame_num in candidate_frame_nums:
                try:
                    with zf.open(f"{frame_num}.jpeg") as f:
                        image_data = f.read()
                        pil = Image.open(io.BytesIO(image_data))
                        frames.append(np.array(pil))
                except Exception as e:
                    os.system(f"rm {zip_file_path}")
                    raise Exception(f"Exception occurs while reading frame {frame_num}.jpeg from file {zip_file_path}. Deleted it...\n\
                                    \rRaw expception: {e} \
                                    ")

        if pnr_frame is not None:
            keyframe_location = np.argmin(keyframe_candidates_list)
            hard_labels = np.zeros(len(candidate_frame_nums))
            hard_labels[keyframe_location] = 1
            labels = hard_labels
        else:
            labels = keyframe_candidates_list # all zero

        # Calculating the effective fps. In other words, the fps after sampling
        # changes when we are randomly clipping and varying the duration of the
        # clip
        final_clip_length = (random_end_frame/30) - (random_start_frame/30)
        effective_fps = num_frames_per_video / final_clip_length

        # Commented by Jiachen Lei
        # return np.concatenate(frames), np.array(labels), effective_fps
        return frames, np.array(labels), effective_fps, candidate_frame_nums


    def assert_exist_wtolerance(self, path, message, retry=5):
        if not os.path.exists(path):
            flag = False
            for i in range(retry):
                if os.path.exists(path):
                    flag = True
            if not flag:
                assert False, message


    def get_frames_for(self, video_path, frames_list):
        """
        Code for decoding the video
        """
        # frames = list()

        # Codes by Jiachen Lei, since we only care about frames, audio is ignored
        # Frames are converted into RGB format

        # version 1, write with opencv
        # cap = cv2.VideoCapture(video_path)
        # assert cap != None, "Fail to open video"
        # ret = cap.set(cv2.CAP_PROP_POS_FRAMES, frames_list[0]-1)
        # assert ret == True, "Fail to set video to given frame"
        # for fidx in frames_list:
        #     ret, frame = cap.read()
        #     # print(frame)
        #     assert ret == True, "Fail to read videos before finish extracting frames"
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     frames.append(frame)

        # cap.release()
 
        # version 2, write with decord
        # vr = decord.VideoReader(video_path)
        # frames = vr.get_batch(frames_list).asnumpy() # return RGB format numpy.ndarray

        # Codes below are official implementation av==8.0.3
        # with av.open(video_path) as container:
        #     for frame in _get_frames(
        #         frames_list,
        #         container,
        #         include_audio=False,
        #         audio_buffer_frames=0
        #     ):  
        #         frame = frame.to_rgb().to_ndarray()
        #         frames.append(frame)

        cv2.setNumThreads(3)
        # official code where av == 6.0.0
        frames = []
        container = av.open(video_path)
        for frame in _get_frames(
                frames_list,
                container,
                include_audio=False,
                audio_buffer_frames=0
            ):  
            frame = frame.to_rgb().to_ndarray()
            frames.append(frame)

        return frames

def _debug(**kwargs):
    def _sample_frames(
        unique_id,
        clip_start_frame,
        clip_end_frame,
        num_frames_required,
        pnr_frame
    ):
        num_frames = clip_end_frame - clip_start_frame
        if num_frames < num_frames_required:
            print(f'Issue: {unique_id}; {num_frames}; {num_frames_required}')
        error_message = "Can\'t sample more frames than there are in the video"
        assert num_frames >= num_frames_required, error_message
        lower_lim = np.floor(num_frames/num_frames_required)
        upper_lim = np.ceil(num_frames/num_frames_required)
        print(f"lower lim: {lower_lim} upper_lim: {upper_lim}")

        lower_frames = list()
        upper_frames = list()
        lower_keyframe_candidates_list = list()
        upper_keyframe_candidates_list = list()
        for frame_count in range(clip_start_frame, clip_end_frame, 1):
            if frame_count % lower_lim == 0:
                lower_frames.append(frame_count)
                if pnr_frame is not None:
                    lower_keyframe_candidates_list.append(
                        np.abs(frame_count - pnr_frame)
                    )
                else:
                    lower_keyframe_candidates_list.append(0.0)
            if frame_count % upper_lim == 0:
                upper_frames.append(frame_count)
                if pnr_frame is not None:
                    upper_keyframe_candidates_list.append(
                        np.abs(frame_count - pnr_frame)
                    )
                else:
                    upper_keyframe_candidates_list.append(0.0)
        if len(upper_frames) < num_frames_required:
            return (
                lower_frames[:num_frames_required],
                lower_keyframe_candidates_list[:num_frames_required]
            )
        return (
            upper_frames[:num_frames_required],
            upper_keyframe_candidates_list[:num_frames_required]
        )
    
    return _sample_frames(**kwargs)


if __name__ == "__main__":

    def get_frame_for(video_path, frames_list):
        frames = []
        container = av.open(video_path)
        for frame in _get_frames(
                frames_list,
                container,
                include_audio=False,
                audio_buffer_frames=0
            ):  
            frame = frame.to_rgb().to_ndarray()
            frames.append(frame)

        return frames

    video_path = "/mnt/ego4d/v1/full_scale/6c03be74-4692-4e3e-8eab-01f9f8e0d3ba.mp4"
    frame_list = [8907, 8908, 8909, 8910, 8911, 8912, 8913, 8914, 8915, 8916, 8917, 8918, 8919, 8920, 8921, 8922, 8923, 8924, 8925, 8926, 8927, 8928, 8929, 8930, 8931, 8932, 8933, 8934, 8935, 8936, 8937, 8938, 8939, 8940, 8941]

    frames = get_frame_for(video_path, frame_list)

    print(frames[0].shape)