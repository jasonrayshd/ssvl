import os
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator
from kinetics import VideoClsDataset, VideoMAE
from ssv2 import SSVideoClsDataset


from ego4d import StateChangeDetectionAndKeyframeLocalisation
from epickitchens import Epickitchens
from argparse import Namespace


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_dataset(is_train, test_mode, args):

    if "Ego4d-statechange" in args.data_set:
        mode = None
        
        if is_train is True:
            mode = 'train'
        elif test_mode is True:
            mode = 'test'
        else:  
            mode = 'val'

        """
            Used configuration in ego4d dataset (StateChangeDetectionAndKeyframeLocalisation.py)      by Jiachen, 2022.05.19

            DATA.ANN_DIR
            DATA.VIDEO_DIR_PATH
            DATA.NO_SC_PATH
            DATA.CLIP_LEN_SEC
            DATA.CROP_SIZE: width and height of resized result
            DATA.SAMPLING_FPS
        """
        # Edited by Jiachen Lei, 2022.05.24
        # Refer to official file state-change-localization-classification/i3d-resnet50/configs/2021-09-18_keyframe_loc_release1-v2_main-experiment.yaml

        # ann_dir = args.data_path
        # video_dir_path = os.path.join(args.data_path, "clips")
        # clips_save_path = "/mnt/ego4d/v1/pos"
        # no_sc_path = "/mnt/ego4d/v1/neg"

        cfg = Namespace(**{
            "DATA": Namespace(**{
                # Data Loading
                "ANN_DIR": args.anno_path,
                "VIDEO_DIR_PATH": args.data_path,
                "CLIPS_SAVE_PATH": args.pos_clip_save_path, # "/mnt/shuang/Data/ego4d/preprocessed_data/pos"
                "NO_SC_PATH": args.neg_clip_save_path, # "/mnt/shuang/Data/ego4d/preprocessed_data/neg"

                "SAVE_AS_ZIP": True,                # save frames in zip file for efficient data loading
                "READ_BY_CLIPS": args.debug,        # read by clips or full_scale video

                # Data Sampling
                "CLIP_LEN_SEC": args.clip_len,      # Duration time in second of clip
                "CROP_SIZE": args.input_size,
                "SAMPLING_FPS": args.sampling_rate, # Sampled frames per second for training
            })
        })

        # cfg is of type namespace
        dataset = StateChangeDetectionAndKeyframeLocalisation(cfg, mode, args=args)
        """
            Jiachen 2022.05.25
            dataset.__getitem__() will return 
            (1) torch.Tensor, frames of shape (Channels, T, H, W). e.g. Tensor(ch,t,h,w)
            (2) list, 1-D numpy.ndarray label that indicates whether each frame is pnr or not (if yes, the value is 1 else the value is 0)
                and boolean value indicates whether stage change occurs in the clip or not
            (3) float, fps of current sampled frames
            (4) dict, info of the clip
        """

        if "localization" in args.data_set and "classification" in args.data_set:
            nb_classes = -1
            args.two_head = True # make sure two_head is set
        elif "localization" in args.data_set:
            nb_classes = cfg.DATA.SAMPLING_FPS * cfg.DATA.CLIP_LEN_SEC + 1
        elif "classification" in args.data_set:
            nb_classes = 2

    elif args.data_set == "Epic-kichen":
        pass

    elif args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 400

    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'test.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
