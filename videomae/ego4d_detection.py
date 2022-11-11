import copy
import random
from functools import partial
from omegaconf import OmegaConf

import torch
from torch import nn

from modeling_detection import VitDet, Ego4dGeneralizedRCNN

from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
    detection_utils
)
import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

from detectron2.layers import ShapeSpec

from detectron2.modeling.backbone.vit import SimpleFeaturePyramid

from detectron2.modeling.anchor_generator import DefaultAnchorGenerator

from detectron2.modeling.backbone import BasicStem, FPN, ResNet
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead
from detectron2.modeling.roi_heads import (
    StandardROIHeads,
    FastRCNNOutputLayers,
    MaskRCNNConvUpsampleHead,
    FastRCNNConvFCHead,
)

from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.layers.batch_norm import get_norm


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1

    return lr_decay_rate ** (num_layers + 1 - layer_id)


def Ego4dMapper(dataset_dict, is_train=True):

    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

    # can use other ways to read image
    image = detection_utils.read_image(dataset_dict["file_name"], format="RGB")
    pre_image = detection_utils.read_image(dataset_dict["pre_file_name"], format="RGB")
    post_image = detection_utils.read_image(dataset_dict["post_file_name"], format="RGB")

    if is_train:
    # See "Data Augmentation" tutorial for details usage
        augs = T.AugmentationList([
                    T.RandomBrightness(0.9, 1.1),
                    T.RandomFlip(prob=0.5),
                    T.ResizeShortestEdge(224, max_size=1920),
                    # T.RandomCrop("absolute", (224, 224))
                    T.Resize((224, 224))
                ])
    else:
        augs = T.AugmentationList([
                    T.ResizeShortestEdge(224, max_size=1920),
                    # T.CenterCrop("absolute", (224, 224))
                    T.Resize((224, 224))
                ])

    auginput = T.AugInput(image)
    transform = augs(auginput)
    # print(auginput.image.shape)
    image = torch.from_numpy(auginput.image.transpose(2, 0, 1).copy())
    pre_image = torch.from_numpy(transform.apply_image(pre_image).transpose(2, 0, 1).copy())
    post_image = torch.from_numpy(transform.apply_image(post_image).transpose(2, 0, 1).copy())

    # random pick combination of pnr frame with pre/post frame
    _p = random.random()
    vit_input = torch.stack([pre_image, image.clone()], dim=0) if _p > 0.5 else torch.stack([image.clone(), post_image], dim=0)
    vit_input = vit_input.permute(1, 0, 2, 3) / 255.0

    # print(vit_input)
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1,1)
    vit_input = (vit_input - mean) / std

    annos = [
        detection_utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]

    return {
       # create the format that the model expects
       "image": image,
       "vit_input": vit_input,
       "width": dataset_dict["width"],
       "height": dataset_dict["height"],
       "image_id": dataset_dict["image_id"],
       "instances": detection_utils.annotations_to_instances(annos, image.shape[1:])
    }


imagenet_rgb256_mean=[123.675, 116.28, 103.53],
imagenet_rgb256_std=[58.395, 57.12, 57.375],
imagenet_bgr256_mean=[103.530, 116.280, 123.675],
# When using pre-trained models in Detectron1 or any MSRA models,
# std has been absorbed into its conv1 weights, so the std needs to be set 1.
# Otherwise, you can use [57.375, 57.120, 58.395] (ImageNet std)
imagenet_bgr256_std=[1.0, 1.0, 1.0],


model = L(Ego4dGeneralizedRCNN)(
    backbone=L(FPN)(
        bottom_up=L(ResNet)(
            stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
            stages=L(ResNet.make_default_stages)(
                depth=50,
                stride_in_1x1=True,
                norm="FrozenBN",
            ),
            out_features=["res2", "res3", "res4", "res5"],
        ),
        in_features="${.bottom_up.out_features}",
        out_channels=256,
        top_block=L(LastLevelMaxPool)(),
    ),
    proposal_generator=L(RPN)(
        in_features=["p2", "p3", "p4", "p5", "p6"],
        head=L(StandardRPNHead)(in_channels=256, num_anchors=3),
        anchor_generator=L(DefaultAnchorGenerator)(
            sizes=[[32], [64], [128], [256], [512]],
            aspect_ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            offset=0.0,
        ),
        anchor_matcher=L(Matcher)(
            thresholds=[0.3, 0.7], labels=[0, -1, 1], allow_low_quality_matches=True
        ),
        box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_topk=(2000, 1000),
        post_nms_topk=(1000, 1000),
        nms_thresh=0.7,
    ),
    roi_heads=L(StandardROIHeads)(
        num_classes=80,
        batch_size_per_image=512,
        positive_fraction=0.25,         # fraction of positive (foreground) proposals to sample for training.
        proposal_matcher=L(Matcher)(
            thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
        ),
        box_in_features=["p2", "p3", "p4", "p5"],
        box_pooler=L(ROIPooler)(
            output_size=7,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        box_head=L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[],
            fc_dims=[1024, 1024],
        ),
        box_predictor=L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
            num_classes="${..num_classes}",
        ),
        mask_in_features=["p2", "p3", "p4", "p5"],
        mask_pooler=L(ROIPooler)(
            output_size=14,
            scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        ),
        mask_head=L(MaskRCNNConvUpsampleHead)(
            input_shape=ShapeSpec(channels=256, width=14, height=14),
            num_classes="${..num_classes}",
            conv_dims=[256, 256, 256, 256, 256],
        ),
    ),
    pixel_mean=imagenet_bgr256_mean,
    pixel_std=imagenet_bgr256_std,
    input_format="BGR",
)


model.pixel_mean = imagenet_rgb256_mean
model.pixel_std = imagenet_rgb256_std
model.input_format = "RGB"

# Base
embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
# Large
# embed_dim, depth, num_heads, dp = 1024, 24, 16, 0.4

# Creates Simple Feature Pyramid from ViT backbone
model.backbone = L(SimpleFeaturePyramid)(
    net=L(VitDet)(  # Single-scale ViT backbone
        img_size=224,
        patch_size=16,
        num_frames = 2,
        tubelet_size = 2,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 2, 5, 8 11 for global attention
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        residual_block_indexes=[],
        # use_rel_pos=True,
        out_feature="last_feat",
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=224,
)

model.roi_heads.box_head.conv_norm = "LN"
model.roi_heads.num_classes = 1

model.roi_heads.mask_in_features = None
model.roi_heads.mask_pooler  = None
model.roi_heads.mask_head  = None

# 2conv in RPN:
model.proposal_generator.head.conv_dims = [-1, -1]

# 4conv1fc box head
model.roi_heads.box_head.conv_dims = [256, 256, 256, 256]
model.roi_heads.box_head.fc_dims = [1024]

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = "ckpt/pretrain_vitb_800_kinetics400.pth"
train.checkpointer = {
    "period": 20,
}
train.name = "temp"
train.output_dir = f"/data/jiachen/ssvl_output/{train.name}"
train.log_period = 20


# Schedule
# 100 ep = 184375 iters * 64 images/iter / 118000 images/ep
train.max_iter = 270000

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[210000, 250000],
        num_updates=train.max_iter,
    ),
    warmup_length = 1,
    warmup_factor = 0.0001,
)

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.lr = 5.0e-10
optimizer.weight_decay = 0.01
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.75)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}

# dataset

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="ego4dv1_pnr_objects_train"),
    mapper=partial(Ego4dMapper, is_train=True),
    total_batch_size=64,
    num_workers=8,
)

dataloader.evaluator = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="ego4dv1_pnr_objects_val", filter_empty=False),
    mapper=partial(Ego4dMapper, is_train=False),
    total_batch_size=64,
    num_workers=8,
)

# dataloader.test = L(build_detection_test_loader)(
#     dataset=L(get_detection_dataset_dicts)(names="ego4dv1_pnr_objects_test", filter_empty=False),
#     mapper=partial(Ego4dMapper, is_train=False),
#     total_batch_size=2,
#     num_workers=8,
# )