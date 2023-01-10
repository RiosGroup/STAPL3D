import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class Dataset_STAPL3D(Dataset):
    """Dataset for STAPL3D blocks."""

    def __init__(self, blocks, ids_image, ids_labels, augmenter=None):

        self._blocks = blocks
        self.ids_image = ids_image
        self.ids_labels = ids_labels
        self.augmenter = augmenter

        axlab = self._blocks[0].axlab
        self._axlab = ''.join([al for al in axlab if al in 'zyx'])

    def __len__(self):

        return len(self._blocks)

    def __getitem__(self, index):

        block = self._blocks[index]

        # FIXME: assuming uint16 here for a bit
        # factors = {'z': 1, 'y': 4, 'x':4}

        img = self.read_blockdata(block, self.ids_image, imtype='Image')
        # img = img.downsampled(factors)
        image = np.array(img.ds, dtype='int32')

        lab = self.read_blockdata(block, self.ids_labels, imtype='Label')
        # lab = lab.downsampled(factors, ismask=True)
        labels = np.array(lab.ds, dtype='int64')

        if self.augmenter is not None:
            augs = self.augmenter({"image": image, "mask": labels})
            image, labels = augs['image'], augs['mask']

        image = np.expand_dims(image, axis=0)
        labels = np.expand_dims(labels, axis=0)

        # timg = nn.functional.interpolate(torch.from_numpy(image).float(), scale_factor=[1, 4, 4], mode='trilinear')
        # tlab = nn.functional.interpolate(torch.from_numpy(labels).float(), scale_factor=[1, 4, 4], mode='nearest').float()
        # return timg, tlab, index

        return torch.from_numpy(image).float(), torch.from_numpy(labels).float(), index

    def read_blockdata(self, block, ids, imtype=''):
        """Read a block-dataset."""

        props = dict(axlab=self._axlab, imtype=imtype)
        block.create_dataset(ids, **props)
        block_ds = block.datasets[ids]
        block_ds.read(from_block=True)

        return block.datasets[ids].image


