from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator, AgnosticMaskingGenerator

from ego4d import StateChangeDetectionAndKeyframeLocalisation
from epickitchens import Epickitchens

class DataAugmentationForVideoMAE(object):
    def __init__(self, args, flow_mode=False):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD

        # flow image should be the same given normalized images
        # thus it do not need to be normalized
        normalize = GroupNormalize(self.input_mean, self.input_std, flow_mode=flow_mode)
        # This contains random process and is rewritten for flow image processing
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66], flow_mode=flow_mode)

        # accpet pil image
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False, flow_mode=flow_mode),
            ToTorchFormatTensor(div=True, flow_mode=flow_mode),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        elif args.mask_type == "agnostic":
            self.masked_position_generator = AgnosticMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        else:
            raise ValueError(f"Unknown mask type {args.mask_type}")

    def __call__(self, images, flow_mode=""):
        process_data, flows_or_none = self.transform(images)

        if flow_mode == "local":
            return process_data, flows_or_none, self.masked_position_generator()
        else:
            return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args, **kwargs):
    
    transform = DataAugmentationForVideoMAE(args, flow_mode=args.flow_mode)
    mode = "train"

    if "egoclip" in args.task:
        pass
        # dataset = StateChangeDetectionAndKeyframeLocalisation(args.cfg, mode, args=args, pretrain=True, pretrain_transform=transform)

    elif "epic-kitchen" in args.data_set:
        dataset = Epickitchens(args.cfg, mode,
                                pretrain = True, pretrain_transform=transform, 
                                flow_mode=args.flow_mode,
                                flow_extractor = kwargs["flow_extractor"] if "flow_extractor" in kwargs.keys() else None,
                                )

    else:
        raise NotImplementedError()

    print("Data Aug = %s" % str(transform))
    return dataset


# build finetuning dataset
def build_dataset(mode, args, flow_extractor=None):

    if args.task == "oscc-pnr":

        dataset = StateChangeDetectionAndKeyframeLocalisation(args.cfg, mode, args=args, pretrain=False, flow_extractor=flow_extractor)

        if args.task == "oscc-pnr":
            nb_classes = -1
            # args.two_head = True # make sure two_head is set
        elif args.task == "pnr":
            nb_classes = args.cfg.DATA.SAMPLING_FPS * args.cfg.DATA.CLIP_LEN_SEC + 1
        elif args.task == "oscc":
            nb_classes = 2

    elif args.task == "lta":
        pass
    else:
        raise NotImplementedError()

    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
