import numpy as np

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
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

        batch_mask = []
        for i in range(batch_size):
            np.random.shuffle(mask_per_frame)
            mask = np.tile(mask_per_frame.copy(), (self.frames,1)).flatten()
            batch_mask.append(mask)

        return batch_mask

class AgnosticMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.total_masks = int(mask_ratio * self.total_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
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

        batch_mask = []
        for i in range(batch_size):
            np.random.shuffle(mask)

            batch_mask.append(mask.copy())

        return batch_mask



if __name__ == "__main__":

    g = TubeMaskingGenerator((1, 14, 14), 0.9)
    g(2)
    print(g(2))