#ego_baseline

This baseline can be only train on single GPU, there exists a situation of label not match info on multi GPUs

How to use it for training:

    python -m train --cfg configs/keyframe_loc_baseline.yaml
    
so you only need to change the config file: keyframe_loc_baseline.yaml in configs
