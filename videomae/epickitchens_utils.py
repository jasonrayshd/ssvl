import logging
import numpy as np
import time
import torch
from PIL import Image
import cv2
import os
import io
logger = logging.getLogger(__name__)

import time
import zipfile
from zipfile import ZipFile
import tarfile
import shutil

import socket
import pickle
from multiprocessing.managers import SyncManager
from multiprocessing import Lock
import multiprocessing as mlp
import threading as th
from collections import defaultdict

class CacheManager(object):

    def __init__(self, address:tuple, local_world_size:int, log_path="./"):
        self.address = address
        self.manager_address = ( self.address[0], self.address[1] + 1)
        self.local_world_size = local_world_size
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.log_path = log_path
        self.buffer = 5

        self.LOCK_DCT = {
            "zip": Lock(),
            "check_board": Lock(),
        }
        # self.m = None
        # m = SyncManager()
        # m.start()
        # self.zip_dct = m.dict()
        # self.check_dct = m.dict()
        self.logger = None
        self.m = SyncManager(address=self.manager_address)
        self.zip_dct = None
        self.check_dct = None

    def start(self):
        self.s.bind(self.address)
        self.s.listen()
        self.m.start()

        zip_dct = self.m.dict()
        check_dct = self.m.dict()

        pool = []
        for i in range(self.local_world_size):
            conn, addr = self.s.accept()
            logger.debug(f"Accept connection from {addr}")

            with conn:
                # client_thread = th.Thread(target=self.on_new_client, args=(conn, ))
                # client_thread.start()
                client_sub_process = mlp.Process(target=self.on_new_client, args=(conn, zip_dct, check_dct))
                client_sub_process.start()
                pool.append(client_sub_process)

        for p in pool:
            p.join()

        logger.debug("Exit main cache manager process")



    def on_new_client(self, conn, zip_dct, check_dct):
        self.zip_dct = zip_dct
        self.check_dct = check_dct
        retry = 5

        for i in range(retry):
            try:
                self.m.connect()
            except Exception as e:
                time.sleep(1)

        while True:
            self.recv(conn)

    def connect(self):
        self.s.connect(self.address)

    def socket_recv(self, conn):
        BUFFER = self.buffer

        binary_length = conn.recv(BUFFER)
        conn.sendall(b"t")

        length = pickle.loads(binary_length)
        logger.warn(f"in socket_recv - received object length: {length}")

        binary_obj = conn.recv(length)
        conn.sendall(b"t")

        return pickle.loads(binary_obj)

    def socket_send(self, obj, conn):

        BUFFER = self.buffer

        binary_obj = pickle.dumps(obj)
        length = len(binary_obj)
        logger.warn(f"in socket_send - object to be sent length:{length}")
        binary_length = pickle.dumps(length)
        logger.warn(f"in socket_send - bit length of length:{len(binary_length)}")

        conn.sendall(binary_length)
        msg = conn.recv(1)
        assert msg == b"t", f"incorrect message received: {pickle.dumps(msg)}"

        conn.sendall(binary_obj)
        msg = conn.recv(1)
        assert msg == b"t", f"incorrect message received: {pickle.dumps(msg)}"

        return True

    def recv(self, conn):
        BUFFER = self.buffer
        # first commuincation: meta head
        # meta = self.socket_recv(conn)

        # length = meta["length"]
        # data_type = meta["type"]
        # second commuincation: data

        com_lst = self.socket_recv(conn)
        com, args = com_lst

            # then result should be string
        
        if com == "exists_auto_acquire":
            ret = self.exists_auto_acquire(*args)
        elif com == "release_and_check":
            ret = self.release_and_check(*args)
        elif com == "length_of_zip_dct":
            ret = self.length_of_zip_dct()
        elif com == "init_zip_dct":
            ret = self.init_zip_dct(*args)
        else:
            raise ValueError(f"Unkown command: {com}")

        self.socket_send(ret, conn)
        # conn.sendall(binary_length)
        # conn.recv(len(b"t"))
        # conn.sendall(ret_binary)

    def call(self, com_lst:list):

        BUFFER = self.buffer
        # first commuincation: meta head
        # binary_com = pickle.dumps(com_lst)
        # length = len(binary_com)
        # com_lst = {
        #     "type":"command",
        #     "length": length
        # }
        # binary_meta = pickle.dumps(meta)

        # self.socket_send(com_lst, self.s)
        # # send command and arguments
        # self.s.sendall(binary_meta)
        # msg = self.s.recv(len(b"t"))

        self.socket_send(com_lst, self.s)

        # receive return values
        ret = self.socket_recv(self.s)

        return ret

    def read_keys(self, lock):
        self.acquire(lock)
        if lock == "zip":
            keys = self.zip_dct.keys()
        else:
            keys = self.check_dct.keys()
        self.release(lock)

        return keys

    def acquire(self, lock):
        self.LOCK_DCT[lock].acquire()
        logger.debug(f"Acquire: {lock}")

    def release(self, lock):
        self.LOCK_DCT[lock].release()
        logger.debug(f"Release: {lock}")

    def length_of_zip_dct(self):
        self.acquire(lock="zip")
        length = len(self.zip_dct.keys())
        self.release(lock="zip")

        return length

    def init_zip_dct(self, path):
        # logger.debug("initializing zip dictionary")
        self.acquire(lock="zip")
        self.acquire(lock="check_board")
        # logger.debug(f"")
        if len(self.zip_dct.keys()) == 0:
            logger.debug(f"{os.getpid()} length of zip dictionary is 0")
            zf = os.listdir(path)
            for f in zf:
                self.zip_dct[os.path.join(path, f)] = self.m.Lock()
                self.check_dct[os.path.join(path, f)] = True
            logger.debug("Finish initializing zip dictionary")
        # else:
            # logger.debug(f"length of zip dictionary is not 0")
        self.release(lock="check_board")
        self.release(lock="zip")

        return True

    def exists_auto_acquire(self, path):
        self.acquire(lock="zip")

        if path in self.zip_dct.keys():
            logger.debug(f"{path} is in zip dict {self.zip_dct.keys()}")
            self.release(lock="zip")
            keys = self.read_keys(lock="check_board")
            if path in keys:
                logger.debug(path+" has been processed")
                return True
            else: # some processs have not finished transfering zip file yet
                logger.debug(path+f" is transferring, lock:{self.zip_dct[path]}")
                self.zip_dct[path].acquire()
                self.zip_dct[path].release()
                return True
        else:
            # if not exists, then lock until finish caching
            assert path not in self.zip_dct.keys(), "Conflict occurs when transferring zip file"
            logger.debug(f"{path} is not in zip dict {self.zip_dct.keys()}")
            self.zip_dct[path] = self.m.Lock()
            logger.debug(path+" is created")
            self.zip_dct[path].acquire()
            logger.debug(path+f" is acquired, {self.zip_dct[path]}")
            self.release(lock="zip")

            return False

    def release_and_check(self, path):
        self.zip_dct[path].release()
        logger.debug(path+" is released")

        self.acquire(lock="check_board")
        self.check_dct[path] = True
        with open(os.path.join(self.log_path,"cache.log"), "a+") as logf:
            logf.write(path + "\n")
        self.release(lock="check_board")
        # release lock of zip file
        
        return True


def cache_tar_to_local(zip_file_path, raw_dest, cache_log_file = "cache.log", flow=False, cache_manager=None):

    assert os.path.exists(zip_file_path), "Zip file not found when caching it locally"
    zip_file_name = zip_file_path.split("/")[-1]

    # if already cached, then return
    dest = os.path.join(raw_dest, "flow" if flow else "rgb")
    # assert cache_manager is not None, "Cache mananger is None"

    # length = cache_manager.call(["length_of_zip_dct", ["", ]])
    # logger.debug(f"length:{length}")
    # if length == 0:
    # cache_manager.call([ "init_zip_dct", [dest, ] ])

    # if cache_manager.call(["exists_auto_acquire", [os.path.join(dest, zip_file_name)]]):
    #     logger.debug(f"{os.path.join(dest, zip_file_name)} exists, using cached file")
    #     return True


    if os.path.exists(os.path.join(dest, zip_file_name)):
        # if exists, then check whether the compressed file has been transferred
        if os.path.getsize(zip_file_path) == os.path.getsize(os.path.join(dest, zip_file_name)):
            return True
        else:
            logger.debug(f"Zip file is transferring, read from remote.. {zip_file_path}")
            return False

    # else copy file and handle potential error
    os.makedirs(dest, exist_ok=True)
    
    retry = 10
    for i in range(retry):
        # keep trying caching tar file
        try:
            logger.debug(f"Shutil - start transferring {zip_file_path}")
            ret_dest = shutil.copy(zip_file_path, dest)
            # write to cache log file
            # cache_log_fbar = open(cache_log_file, "a+")
            # cache_log_fbar.write(os.path.join(dest, zip_file_name) + "\n")
            # cache_log_fbar.close()
            logger.debug(f"Finish transfer zip file {zip_file_path}")
            # cache_manager.call(["release_and_check", [os.path.join(dest, zip_file_name)]])
            return True

        except OSError as e:
            logger.warn(f"Caching tar file to local directory failed:\nRaw Exception:\n{e}")
            # cache_manager.call(["release_and_check", [os.path.join(dest, zip_file_name)]])
            return False

            # assume not enough space and delete pre-cached tar file
            # cache_log_fbar = open(cache_log_file, "r")
            # # ATTENTION: with \n at tail of each element in the list
            # # each element in the list is a absolute path of previously cached zip file
            # cached_file_lst = cache_log_fbar.readlines()
            # cache_log_fbar.close()

            # if len(cached_file_lst) != 0:

            #     zip_file_path = cached_file_lst[0].strip("\n")
            #     cached_file_lst.pop(0)

            #     cache_log_fbar = open(cache_log_file, "w")
            #     cache_log_fbar.write("".join(cached_file_lst))
            #     cache_log_fbar.close()
            # else:
            #     return False

            # # remove earliest cached file
            # try:
            #     os.remove(zip_file_path)
            # except:
            #     print(f"Fail to delete cached file:{zip_file_path}, continue removing next tar files...")
            #     continue

            # print(f"Deleted previously cached file:{zip_file_path} and try again...")

        except Exception as e:
            # cache_manager.call(["release_and_check", [os.path.join(dest, zip_file_name)]])
            logger.warn(f"Caching tar file to local directory failed:\nRaw Exception:\n{e}")
            return False

    logger.warn(f"Reach maximum caching attempts... zip_file_path:{zip_file_path}")

def extract_zip(path_to_save, ext="tar", frame_list = [], flow=False, cache_dest="/data/jiachen/temp", cache_manager=None, force=False):

    # num_frames = len(os.listdir(path_to_save)) # existing frames in the directory
    message = f"Zip file does not exists: {path_to_save}"
    assert os.path.exists(path_to_save + "." + ext), message
    os.makedirs(path_to_save, exist_ok=True)

    logger.info(f"Start extracting frame from zip file:{path_to_save}.{ext} ...")
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
            if len(frame_list) != 0:

                if cache_dest == "":
                    # specify where to cache compressed file
                    cache_dest = os.getcwd()
                # if only extract several frames from the tar file then to ensure reading efficiency
                # cache tar file locally
                ret = cache_tar_to_local(path_to_save + "." + ext, raw_dest=cache_dest, flow=flow, cache_manager=cache_manager)
                # print(f"caching file return: {ret}")
                if ret:
                    zip_file_name = path_to_save.split("/")[-1] + "." + ext
                    # read from local directory
                    tf = tarfile.open( os.path.join(cache_dest, "flow" if flow else "rgb", zip_file_name), "r")
                    # print("opened local compressed file")
                else:
                    # fail to cache tar file or the file is transferring, read from original path
                    tf = tarfile.open( path_to_save + "." + ext, "r")
            else:
                tf = tarfile.open( path_to_save + "." + ext, "r")

        except Exception as e:
            raise Exception(f"Exception occurs while opening tar file: {path_to_save}.tar, file might be corrupted \
                            \rRaw exception:\n{e}")

        if len(frame_list) != 0:
            dir_name = path_to_save.split("/")[-1]
            retry = 5         
            if flow:
                # Obtain existing flow image list to prevent duplicate writing
                if os.path.exists(os.path.join(path_to_save, "u")):
                    exist_uflow_list = os.listdir(os.path.join(path_to_save, "u"))
                else:
                    exist_uflow_list = []
                if os.path.exists(os.path.join(path_to_save, "v")):
                    exist_vflow_list = os.listdir(os.path.join(path_to_save, "v"))
                else:
                    exist_vflow_list= []

                for frame_idx in frame_list:
                    for i in range(retry):
                        try:
                            if not frame_idx in exist_uflow_list or force:
                                tf.extract(f"./u/{frame_idx}", path_to_save)
                            if not frame_idx in exist_vflow_list or force:
                                tf.extract(f"./v/{frame_idx}", path_to_save)
                            break
                        except KeyError as e:
                            raise Exception(f"Key error raised tf.names:{tf.getnames()[:20]}... frame_idx:{frame_idx} frame_list:{frame_list} path_to_save:{path_to_save}")
                        except FileExistsError as e:
                            logger.warn(f"When extracting {path_to_save} {frame_idx}, file eixsts, retrying...")
                            continue
            else:
                if os.path.exists(path_to_save):
                    exist_frame_list = os.listdir(path_to_save)
                else:
                    exist_frame_list = []

                for frame_idx in frame_list:
                    for i in range(retry):
                        try:
                            if not frame_idx in exist_frame_list or force:
                                tf.extract("./"+frame_idx, path_to_save)
                            break
                        except KeyError as e:
                            raise Exception(f"Key error raised tf.names:{tf.getnames()[:50]}... frame_idx:{frame_idx} frame_list:{frame_list} path_to_save:{path_to_save}")
                        except FileExistsError as e:
                            logger.warn(f"When extracting {path_to_save} {frame_idx}, file eixsts, retrying...")
                            continue
        else:
            tf.extractall(path_to_save)

        tf.close()
    else:
        raise ValueError(f"Unsupported compressed file type: {ext}, expect one of [zip, tar]")

    end_time = time.time()
    logger.info(f"Finish processing zipfile {path_to_save}, time taken: {end_time-start_time}")


def read_from_tarfile(source, name, frame_idx, as_pil=False, flow=False):
    frame_list = [] if not flow else [[], []]
    _debug_shape = []
    with tarfile.open(os.path.join(source,f"{name}.tar")) as tf:
        for i in range(0, len(frame_idx), 1 if not flow else 2):
            idx = frame_idx[i]
            if flow:
                idx = idx // 2 + 1

            img_name = "frame_{:010d}.jpg".format(idx)

            if flow:
                uflow_bytes = tf.extractfile(f"u/{img_name}").read()
                vflow_bytes =  tf.extractfile(f"v/{img_name}").read()
                uflow = Image.open(io.BytesIO(uflow_bytes))
                vflow = Image.open(io.BytesIO(vflow_bytes))
                if not as_pil:
                    uflow = np.array(uflow)
                    vflow = np.array(vflow)

                frame_list[0].append(uflow)
                frame_list[1].append(vflow)
 
            else:
                try:
                    rgb_bytes = tf.extractfile(img_name).read()
                    rgb = Image.open(io.BytesIO(rgb_bytes))
                    if not as_pil:
                        rgb = np.array(rgb)
                    _debug_shape.append(rgb.size)
                    frame_list.append(rgb)
                except Exception as e:
                    print(source, name, frame_idx, img_name)
                    print(e)
    
    print(_debug_shape)
    return frame_list


def retry_load_images(image_paths, retry=10, backend="pytorch", 
            as_pil=False, path_to_compressed="", online_extracting=False,
            flow=False, video_record=None, cache_manager=None,
            read_from_zip=True,
        ):
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
        try:
            Image.open(image_path)
        except:
        # if image does not exist or is corrupted will raise an error
        # we assume the file is missing instead of corrupted here
        # we will handle corruption latter if any
            assert os.path.exists(path_to_compressed), f"image file {image_paths} not exists while compressed file does not exist: {path_to_compressed}"
            if online_extracting:
                img_tmpl = "frame_{:010d}.jpg"

                if video_record is None:
                    # flst = [image_path.split("/")[-1] for image_path in image_paths]

                    st = image_paths[0].split("_")[-1].split(".")[0]
                    end = image_paths[-1].split("_")[-1].split(".")[0]
                    flst = [ img_tmpl.format(idx) for idx in range(int(st), int(end)+1, 1) ]

                else:
                    st = int(video_record.start_frame)
                    n = int(video_record.num_frames)
                    if flow:
                        if st % 2 == 0:
                            st += 1
                        flst = [img_tmpl.format(idx//2 + 1) for idx in range(st, st+n+1, 2)]
                    else:
                        flst = [img_tmpl.format(idx) for idx in range(st, st+n+1, 1)]

                extract_zip(path_to_compressed, frame_list=flst, flow=flow, cache_manager=cache_manager)
            else:
                extract_zip(path_to_compressed)

            break

    for i in range(retry):
        # edited by jiachen, read image and convert to RGB format
        
        imgs = []
        for image_path in image_paths:
            try:
                if not as_pil:
                    imgs.append(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
                else:
                    imgs.append(Image.open(image_path))

            except Exception as e:
                logger.warn(f"PIL reading error:{image_path}, extracting image file again.\nRaw exception:{e}")
                assert os.path.exists(path_to_compressed), f"image file {image_paths} not exists while compressed file does not exist: {path_to_compressed}"
                if online_extracting:
                    # flst = [image_path.split("/")[-1] for image_path in image_paths]
                    extract_zip(path_to_compressed, frame_list=[image_path.split("/")[-1]], flow=flow, cache_manager=cache_manager, force=True)
                else:
                    extract_zip(path_to_compressed)

                # break inner image reading loop and read from start
                break
 
        if len(imgs) == len(image_paths) and all(img is not None for img in imgs):
            if (as_pil == False ) and backend == "pytorch":
                imgs = torch.as_tensor(np.stack(imgs))
            return imgs

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