from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator, AgnosticMaskingGenerator

from ego4d import Ego4dFhoOscc, Ego4dFhoLTA
from epickitchens import Epickitchens

class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD

        # flow image should be the same given normalized images
        # thus it do not need to be normalized
        normalize = GroupNormalize(self.input_mean, self.input_std)
        # This contains random process and is rewritten for flow image processing
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])

        # accpet pil image
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
        elif args.mask_type == "agnostic":
            self.masked_position_generator = AgnosticMaskingGenerator(
                args.window_size, args.mask_ratio
            )
        else:
            raise ValueError(f"Unknown mask type {args.mask_type}")

    def __call__(self, images):
        process_data, flows_or_none = self.transform(images)
        return process_data, flows_or_none, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args, **kwargs):
    
    transform = DataAugmentationForVideoMAE(args)
    mode = "train"

    if args.cfg.task == "egoclip":
        pass
        # dataset = StateChangeDetectionAndKeyframeLocalisation(args.cfg, mode, args=args, pretrain=True, pretrain_transform=transform)

    elif args.cfg.task == "epic-kitchens":
        dataset = Epickitchens(args.cfg, mode, pretrain_transform=transform)

    else:
        raise NotImplementedError()

    print("Data Aug = %s" % str(transform))
    return dataset


# build finetuning dataset
def build_dataset(mode, args, flow_extractor=None):

    if "oscc" in args.cfg.task or "pnr" in args.cfg.task:

        dataset = Ego4dFhoOscc(mode, args.cfg, pretrain=False, flow_extractor=flow_extractor)

    elif args.cfg.task == "lta":
        
        dataset = Ego4dFhoLTA(mode, args.cfg, pretrain=False, flow_extractor=flow_extractor)
    else:
        raise NotImplementedError()


    return dataset
