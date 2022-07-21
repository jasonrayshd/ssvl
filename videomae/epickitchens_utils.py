import logging
import numpy as np
import time
import torch
from PIL import Image
import cv2
import os

logger = logging.getLogger(__name__)

import time
import zipfile
from zipfile import ZipFile
import tarfile

def extract_zip(path_to_save, ext="tar"):

    # num_frames = len(os.listdir(path_to_save)) # existing frames in the directory
    message = f"Zip file does not exists: {path_to_save}"
    assert os.path.exists(path_to_save + "." + ext), message
    os.makedirs(path_to_save, exist_ok=True)

    print(f"Start extracting frame from zip file:{path_to_save}.{ext} ...")
    start_time = time.time()

    # if ext == "zip":
    #     try:
    #         zf = ZipFile( path_to_save + "." + ext, "r")
    #     except zipfile.BadZipFile:       
    #         raise Exception(f"Exception occurs while opening zip file: {path_to_save}.zip, file might be corrupted")

    #     if len(frame_lst) != 0:
    #         namelist = zf.get
    #         for frame in frame_lst:
                
    #     else:
    #         zf.extractall(path_to_save)
    #         zf.close()

    if ext == "tar":
        try:
            tf = tarfile.open( path_to_save + "." + ext, "r")
        except Exception as e:
            raise Exception(f"Exception occurs while opening tar file: {path_to_save}.tar, file might be corrupted \
                            \r Raw exception: {e}")

        tf.extractall(path_to_save)
        tf.close()
    else:
        raise ValueError(f"Unsupported compressed file type: {ext}, expect one of [zip, tar]")

    end_time = time.time()
    print(f"Finish processing zipfile {path_to_save}, time taken: {end_time-start_time}")


def retry_load_images(image_paths, retry=10, backend="pytorch", as_pil=False, path_to_compressed=""):
    """
    This function is to load images with support of retrying for failed load.
    Args:
        image_paths (list): paths of images needed to be loaded.
        retry (int, optional): maximum time of loading retrying. Defaults to 10.
        backend (str): `pytorch` or `cv2`.
    Returns:
        imgs (list): list of loaded images.
    """

    for image_path in image_paths:
        if not os.path.exists(image_path):
            # if one frame does not exist then extract all frames specified in image_paths from the zip
            assert os.path.exists(path_to_compressed), f"image file {image_paths} not exists while compressed file does not exist: {path_to_compressed}"
            extract_zip(path_to_compressed)    
            break

    for i in range(retry):
        # edited by jiachen, read image and convert to RGB format
        if not as_pil:
            imgs = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) for image_path in image_paths]
        else:
            imgs = [Image.open(image_path) for image_path in image_paths]

        # imgs = [cv2.imread(image_path) for image_path in image_paths]

        if all(img is not None for img in imgs):
            if (as_pil == False ) and backend == "pytorch":
                imgs = torch.as_tensor(np.stack(imgs))
            return imgs
        else:
            logger.warn("Reading failed. Will retry.")
            time.sleep(1.0)
        if i == retry - 1:
            raise Exception("Failed to load images {}".format(image_paths))


def get_sequence(center_idx, half_len, sample_rate, num_frames):
    """
    Sample frames among the corresponding clip.
    Args:
        center_idx (int): center frame idx for current clip
        half_len (int): half of the clip length
        sample_rate (int): sampling rate for sampling frames inside of the clip
        num_frames (int): number of expected sampled frames
    Returns:
        seq (list): list of indexes of sampled frames in this clip.
    """
    seq = list(range(center_idx - half_len, center_idx + half_len, sample_rate))

    for seq_idx in range(len(seq)):
        if seq[seq_idx] < 0:
            seq[seq_idx] = 0
        elif seq[seq_idx] >= num_frames:
            seq[seq_idx] = num_frames - 1
    return seq

def pack_pathway_output(cfg, frames):
    """
    Prepare output as a list of tensors. Each tensor corresponding to a
    unique pathway.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `channel` x `num frames` x `height` x `width`.
    Returns:
        frame_list (list): list of tensors with the dimension of
            `channel` x `num frames` x `height` x `width`.
    """
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        frame_list = [frames]
    elif cfg.MODEL.ARCH in cfg.MODEL.MULTI_PATHWAY_ARCH:
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // cfg.SLOWFAST.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        raise NotImplementedError(
            "Model arch {} is not in {}".format(
                cfg.MODEL.ARCH,
                cfg.MODEL.SINGLE_PATHWAY_ARCH + cfg.MODEL.MULTI_PATHWAY_ARCH,
            )
        )
    return frame_list