import numpy as np


def shuffle_along_axis(a, axis):
    # uniform sampling
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)


class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Tube Mask: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self, batch_size):
        """
            Args:
                batch_size: int

            Return:
                batch_mask: list[numpy.ndarray]
        """
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        mask_per_frame = np.expand_dims(mask_per_frame, axis=0).repeat(batch_size, axis=0)
        mask_per_frame = shuffle_along_axis(mask_per_frame, 1)

        mask = np.tile(mask_per_frame.copy(), (1, self.frames))

        batch_mask = [mask[i] for i in range(batch_size)]
        # for i in range(batch_size):
        #     np.random.shuffle(mask_per_frame)
        #     mask = np.tile(mask_per_frame.copy(), (self.frames,1)).flatten()
        #     batch_mask.append(mask)

        return batch_mask


class AgnosticMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.total_masks = int(mask_ratio * self.total_patches)

    def __repr__(self):
        repr_str = "Agnostic Mask: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self, batch_size):
        """
            Args:
                batch_size: int

            Return:
                batch_mask: list[numpy.ndarray]
        """
        mask = np.hstack([
            np.zeros(self.total_patches - self.total_masks),
            np.ones(self.total_masks),
        ]) # (1568, )
        mask = np.expand_dims(mask, axis=0).repeat(batch_size, axis=0)
        mask = shuffle_along_axis(mask, 1)

        batch_mask = [mask[i] for i in range(batch_size)]
        # for i in range(batch_size):
        #     np.random.shuffle(mask)

        #     batch_mask.append(mask.copy())

        return batch_mask

class MultiModalMaskingGenerator:
    """
        Masking generator for rgb and flow input where visible tokens of each modality is predetermined
    """
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame

        self.total_masks = int(mask_ratio * self.total_patches)
        # self.total_masks_per_modality = int(mask_ratio * self.total_patches)
        self.visible_patches = ( self.total_patches - self.total_masks ) * 2

        self.min_visible_patches = 20


    def __repr__(self):
        repr_str = "Multimodal Mask: total patches {}, mask patches {}, minimum visible patches {}".format(
            self.total_patches, self.total_masks, self.min_visible_patches
        )
        return repr_str

    def __call__(self, batch_size):
        """
            Args:
                batch_size: int

            Return:
                batch_mask: list[numpy.ndarray], length: batch_size
        """

        mask = np.arange(0, self.total_patches) 
        mask = np.expand_dims(mask, axis=0).repeat(batch_size, axis=0)

        rgb_visible_num = np.random.randint(self.min_visible_patches, self.visible_patches+1-self.min_visible_patches, size=(1,)).repeat(batch_size, axis=0)
        # print(rgb_visible_num)
        rgb_visible_num = np.expand_dims(rgb_visible_num, axis=1).repeat(self.total_patches, axis=1)
        flow_visible_num = self.visible_patches - rgb_visible_num

        rgb_mask = np.where(mask < rgb_visible_num, 0, 1)
        flow_mask = np.where(mask < flow_visible_num, 0, 1)

        rgb_mask = shuffle_along_axis(rgb_mask, 1)
        flow_mask = shuffle_along_axis(flow_mask, 1)

        mask = np.concatenate((rgb_mask, flow_mask), axis=1)

        mask_lst = [mask[i] for i in range(batch_size)]

        return mask_lst

class MaskGenerator:

    def __init__(self, mask_type_lst, input_size_lst, mask_ratio_lst):
        """

            Args:
                mask_type_lst: list[str, ...]
                input_size_list: list[tuple, ...]
                mask_ratio_list: list[float, ...]
        """
        # accept multiple mask types, input sizes and corresponding mask ratios
        self.mask_type_lst = mask_type_lst
        self.input_size_lst = input_size_lst
        self.mask_ratio_lst = mask_ratio_lst

        self._generator_fn = {
            "agnostic": AgnosticMaskingGenerator,
            "tube": TubeMaskingGenerator,
            "multimodal": MultiModalMaskingGenerator,
        }
        self.generators = []

        for i, mask_type in enumerate(self.mask_type_lst):
            self.generators.append(
                self._generator_fn[mask_type](self.input_size_lst[i], float(self.mask_ratio_lst[i]))
            )

    def __call__(self, batch_size):

        mask_lst = [ gen(batch_size) for gen in self.generators]

        return mask_lst

    def __repr__(self):
        _repr = ""
        for gen in self.generators:
            _repr += repr(gen)
        return _repr


if __name__ == "__main__":

    g = TubeMaskingGenerator((2, 4, 4), 0.5)
    print(g(1))