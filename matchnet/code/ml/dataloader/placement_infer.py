"""The placement network dataset and dataloader.
"""

import glob
import logging
import multiprocessing
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from skimage.draw import circle
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tools.matrix import gen_rot_mtx_anticlockwise
from matchnet import config
from matchnet.code.utils import misc
from matchnet.code.utils.mask import get_kit

class PlacementDataset(Dataset):
    """The placement network dataset.
    """

    def __init__(self, root, sample_ratio, stateless, augment, background_subtract, num_channels, radius):
        """Initializes the dataset.

        Args:
            root: (str) Root directory path.
            sample_ratio: (float) The ratio of negative to positive
                labels.
            stateless: (bool) Whether to use just the current placement
                point and ignore all the ones from the previous objects
                in the sequence.
            augment: (bool) Whether to apply data augmentation.
            background_subtract: (bool) Whether to apply background subtraction.
            num_channels: (int) 4 clones the grayscale image to produce an RGB image.
        """
        self._root = root
        self._sample_ratio = sample_ratio
        self._augment = augment
        self._stateless = stateless
        self._background_subtract = background_subtract
        self._num_channels = num_channels
        self.radius = radius

        # figure out how many data samples we have
        self._get_filenames()

        stats = pickle.load(open(os.path.join(Path(self._root).parent, "mean_std.p"), "rb"))
        if self._num_channels == 4:
            self._c_norm = transforms.Normalize(mean=stats[0][0] * 3, std=stats[0][1] * 3)
        else:
            self._c_norm = transforms.Normalize(mean=stats[0][0], std=stats[0][1])
        self._d_norm = transforms.Normalize(mean=stats[1][0], std=stats[1][1])
        self._to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self._filenames)

    def _get_filenames(self):
        self._filenames = glob.glob(os.path.join(self._root, "*/"))
        self._filenames.sort(key=lambda x: int(x.split("/")[-2]))

    def _load_state(self, name):
        """Loads the raw state variables.
        """
        # load heightmaps
        # c_height = np.asarray(Image.open(os.path.join(name, "final_color_height.png")))
        # d_height = np.asarray(Image.open(os.path.join(name, "final_depth_height.png")))
        c_height = np.asarray(Image.open(os.path.join(name, "init_color_height.png")))
        d_height = np.asarray(Image.open(os.path.join(name, "init_depth_height.png")))

        return c_height, d_height

    def _split_heightmap(self, height):
        """Splits a heightmap into a source and target.

        For placement, we just need the source heightmap.
        """
        half = height.shape[1] // 2
        self._half = half
        height_s = height[:, 0:half].copy()
        return height_s

  
    def __getitem__(self, idx):
        name = self._filenames[idx]

        # load state
        c_height, d_height= self._load_state(name)

        # split heightmap into source and target
        c_height = self._split_heightmap(c_height)
        d_height = self._split_heightmap(d_height)
        self._H, self._W = c_height.shape[:2]

              
        #四张图全都是右半边有盒子的图，所以现在只是分出盒子
        if self._background_subtract :
            c_height,d_height = get_kit(c_height,d_height)
            # idxs = np.vstack(np.where(d_height > self._background_subtract[0])).T
            # mask = np.zeros_like(d_height)
            # mask[idxs[:, 0], idxs[:, 1]] = 1
            # mask = misc.largest_cc(np.logical_not(mask))
            # idxs = np.vstack(np.where(mask == 1)).T
            # c_height[idxs[:, 0], idxs[:, 1]] = 0
            # d_height[idxs[:, 0], idxs[:, 1]] = 0
            # idxs = np.vstack(np.where(d_height_i > self._background_subtract[0])).T
            # mask = np.zeros_like(d_height)
            # mask[idxs[:, 0], idxs[:, 1]] = 1
            # mask = misc.largest_cc(np.logical_not(mask))
            # idxs = np.vstack(np.where(mask == 1)).T
            # c_height_i[idxs[:, 0], idxs[:, 1]] = 0
            # d_height_i[idxs[:, 0], idxs[:, 1]] = 0

        # convert depth to meters
        d_height = (d_height * 1e-3).astype("float32")

        if self._num_channels == 2:
            c_height = c_height[..., np.newaxis]
        else:  # clone the gray channel 3 times
            c_height = np.repeat(c_height[..., np.newaxis], 3, axis=-1)

        # convert heightmaps tensors
        c_height = self._c_norm(self._to_tensor(c_height))
        d_height = self._d_norm(self._to_tensor(d_height[..., np.newaxis]))

        # concatenate height and depth into a 4-channel tensor
        # img_tensor = torch.cat([c_height, d_height], dim=0)
        img_tensor = torch.cat([c_height, d_height], dim=0)#shape为[2,480,424]
        img_tensor = torch.stack([img_tensor, img_tensor], dim=0)#shape为[2,2,480,424]

        return img_tensor


def get_placement_loader(
    foldername,
    dtype="train",
    batch_size=1,
    sample_ratio=1.0,
    shuffle=True,
    stateless=True,
    augment=False,
    background_subtract=None,
    num_channels=2,
    radius=2,
    num_workers=4,
    use_cuda=True,
):
    """Returns a dataloader over the `Placement` dataset.

    Args:
        foldername: (str) The name of the folder containing the data.
        dtype: (str) Whether to use the train, validation or test partition.
        batch_size: (int) The number of data samples in a batch.
        sample_ratio: (float) The ratio of negative to positive
            labels.
        shuffle: (bool) Whether to shuffle the dataset at the end
            of every epoch.
        stateless: (bool) Whether to use just the current placement
            point and ignore all the ones from the previous objects
            in the sequence.
        augment: (bool) Whether to apply data augmentation.
        background_subtract: (bool) Whether to apply background subtraction.
        num_workers: (int) How many processes to use. Each workers
            is responsible for loading a batch.
        use_cuda: (bool) Whether to use the GPU.
    """

    def _collate_fn(batch):
        """A custom collate function.

        This is to support variable length suction labels.
        """
        # imgs = [b[0] for b in batch]
        # labels = [b[1] for b in batch]
        # imgs = torch.stack(imgs, dim=0)
        # return [imgs, labels]
        imgs = [b for b in batch]
        imgs = torch.cat(imgs, dim=0)
        return imgs

    num_workers = min(num_workers, multiprocessing.cpu_count())
    root = os.path.join(config.ml_data_dir, foldername, dtype)

    dataset = PlacementDataset(
        root,
        sample_ratio,
        stateless,
        augment,
        background_subtract,
        num_channels,
        radius,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_fn,
        pin_memory=True,
        num_workers=num_workers,
    )

    return loader