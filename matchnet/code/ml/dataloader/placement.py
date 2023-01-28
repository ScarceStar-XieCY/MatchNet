"""The placement network dataset and dataloader.
"""

import glob
import logging
import multiprocessing
import os
import pickle

import cv2
import numpy as np
import torch
from tools.image_mask.mask_process import mask2coord
from pathlib import Path
from PIL import Image
from skimage.draw import circle_perimeter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tools.matrix import gen_rot_mtx_anticlockwise
from matchnet.code.utils import misc


class PlacementDataset(Dataset):
    """The placement network dataset.
    """

    def __init__(self, root, sample_ratio, stateless, augment, background_subtract,use_color, num_channels, radius):
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
            use_color: (bool) Whether to use color image.
            num_channels: (int) 4 clones the grayscale image to produce an RGB image.
        """
        self._root = root
        self._sample_ratio = sample_ratio
        self._augment = augment
        self._stateless = stateless
        self._background_subtract = background_subtract
        self._num_channels = num_channels
        self._radius = radius
        self._use_color = use_color

        # figure out how many data samples we have
        self._get_filenames()

         # load per-channel mean and std
        norm_info = pickle.load(open(os.path.join(Path(self._root).parent, "mean_std.pkl"), "rb"))
        if use_color:
            color_key = "color"
            self._c_norm = transforms.Normalize(mean=norm_info[color_key]["mean"], std=norm_info[color_key]["std"])
        else:
            color_key = "gray"
            if num_channels == 2:
                self._c_norm = transforms.Normalize(mean=norm_info[color_key]["mean"], std=norm_info[color_key]["std"])
            else:
                self._c_norm = transforms.Normalize(mean=norm_info[color_key]["mean"]*3, std=norm_info[color_key]["std"]*3)
        self._d_norm = transforms.Normalize(mean=norm_info["depth"]["mean"], std=norm_info["depth"]["std"])
        self._transform = transforms.ToTensor()

    def __len__(self):
        return len(self._filenames)

    def _get_filenames(self):
        """Returns a list of filenames to process.
        """
        self._filenames = glob.glob(os.path.join(self._root, "*/"))
        self._filenames.sort(key=lambda x: int(x.split("/")[-2]))

    def _load_state(self, name):
        """Loads the raw state variables.
        """

        # load visual
        if self._use_color:
            color_name = "color.png"
        else:
            color_name = "gray.png"
        depth_name = "depth.png"
            
        c_height_i = cv2.imread(os.path.join(name, "init_" + color_name),cv2.IMREAD_UNCHANGED)
        d_height_i = cv2.imread(os.path.join(name, "init_" + depth_name),cv2.IMREAD_UNCHANGED)
        c_height_f = cv2.imread(os.path.join(name, "final_" + color_name),cv2.IMREAD_UNCHANGED)
        d_height_f = cv2.imread(os.path.join(name, "final_" + depth_name),cv2.IMREAD_UNCHANGED)

        # # convert depth to meters
        # d_height_i = (d_height_i * 1e-3).astype("float32")
        # d_height = (d_height * 1e-3).astype("float32")

        # load the list of suction points
        # load info_dict
        info_dict = pickle.load(open(os.path.join(name, "info_dict.pkl"),"rb"))

        # we just want the current timestep place point
        placement_points = info_dict["init_point"]
        if self._stateless:
            placement_points = placement_points[-1:]

        # load kit mask
        kit_mask = info_dict["kit_no_hole"][-1]
        return c_height_f, d_height_f, placement_points, kit_mask, c_height_i, d_height_i

    def _split_heightmap(self, height):
        """Splits a heightmap into a source and target.

        For placement, we just need the source heightmap.
        """
        half = height.shape[1] // 2
        self._half = half
        height_s = height[:, half:].copy()
        return height_s

    def _sample_negative(self, positives):
        """Randomly samples negative pixel indices.
        """
        max_val = self._H * self._W
        num_pos = len(positives)
        num_neg = int(num_pos * self._sample_ratio)
        positives = np.round(positives).astype("int")
        positives = positives[:, :2]
        positives = np.ravel_multi_index((positives[:, 0], positives[:, 1]), (self._H, self._W))
        if self._sample_ratio < 70:
            negative_indices = []
            while len(negative_indices) < num_neg:
                negative = np.random.randint(0, max_val)
                if negative not in positives:
                    negative_indices.append(negative)
        else:
            allowed = list(set(np.arange(0, max_val)) - set(positives.ravel()))
            np.random.shuffle(allowed)
            negative_indices = allowed[:num_neg]
        negative_indices = np.unravel_index(negative_indices, (self._H, self._W))
        return negative_indices

    def _sample_free_negative(self, kit_mask):
        """Randomly samples negative pixel indices.
        """
        max_val = self._H * self._W
        num_neg = int(100 * self._sample_ratio)
        negative_indices = []
        while len(negative_indices) < num_neg:
            negative_indices.append(np.random.randint(0, max_val))
        negative_indices = np.vstack(np.unravel_index(negative_indices, (self._H, self._W))).T
        idxs = np.random.choice(np.arange(len(kit_mask)), size=30, replace=False)
        inside = kit_mask[idxs]
        negative_indices = np.vstack([negative_indices, inside])
        return negative_indices

    def _sample_translation(self, corrz, angle):
        aff_1 = np.eye(3)
        aff_1[:2, 2] = [-self._uc, -self._vc]
        aff_2 = gen_rot_mtx_anticlockwise(-angle)
        aff_3 = np.eye(3, 3)
        aff_3[:2, 2] = [self._uc, self._vc]
        affine = aff_3 @ aff_2 @ aff_1
        affine = affine[:2, :]
        corrs = []
        for corr in corrz:
            ones = np.ones((len(corr), 1))
            corrs.append((affine @ np.hstack((corr, ones)).T).T)
        max_vv = corrs[0][:, 1].max()
        max_vu = corrs[0][corrs[0][:, 1].argmax()][0]
        min_vv = corrs[0][:, 1].min()
        min_vu = corrs[0][corrs[0][:, 1].argmin()][0]
        max_uu = corrs[0][:, 0].max()
        max_uv = corrs[0][corrs[0][:, 0].argmax()][1]
        min_uu = corrs[0][:, 0].min()
        min_uv = corrs[0][corrs[0][:, 0].argmin()][1]
        for t in corrs[1:]:
            if t[:, 1].max() > max_vv:
                max_vv = t[:, 1].max()
                max_vu = t[t[:, 1].argmax()][0]
            if t[:, 1].min() < min_vv:
                min_vv = t[:, 1].min()
                min_vu = t[t[:, 1].argmin()][0]
            if t[:, 0].max() > max_uu:
                max_uu = t[:, 0].max()
                max_uv = t[t[:, 0].argmax()][1]
            if t[:, 0].min() < min_uu:
                min_uu = t[:, 0].min()
                min_uv = t[t[:, 0].argmin()][1]
        tu = np.random.uniform(-min_vv + 10, self._W - max_vv - 10)
        tv = np.random.uniform(-min_uu + 10, self._H - max_uu - 10)
        return tu, tv
    
    def get_circle_point(self, uv_center, radius):
        """Get all point of one circle."""
        mask = np.zeros((self._H, self._W),dtype = "uint8")
        rr, cc = circle_perimeter(uv_center[0], uv_center[1], radius)
        mask[rr,cc] = 255
        coord = np.stack([rr,cc],axis = 1)
        cv2.fillConvexPoly(mask,coord[:,::-1],255)
        circle_coord = mask2coord(mask, need_xy=False)
        return circle_coord


    def __getitem__(self, idx):
        name = self._filenames[idx]

        # load state
        c_height, d_height, positives, kit_mask, c_height_i, d_height_i = self._load_state(name)

        # split heightmap into source and target
        c_height = self._split_heightmap(c_height)
        d_height = self._split_heightmap(d_height)
        c_height_i = self._split_heightmap(c_height_i)
        d_height_i = self._split_heightmap(d_height_i)
        self._H, self._W = c_height.shape[:2]

        pos_placement = []
        for pos in positives:
            if isinstance(pos, tuple):
                pos = np.array(pos)
            pos[1] -= self._half
            circle_points = self.get_circle_point(pos, self._radius)
            pos_placement.append(circle_points)
        pos_placement = np.concatenate(pos_placement)

        # offset placement point to adjust for splitting
        # pos_placement[:, 1] = pos_placement[:, 1] - self._half
        kit_mask[:, 1] = kit_mask[:, 1] - self._half

        # center of rotation is the center of the kit
        self._uc = int((kit_mask[:, 0].max() + kit_mask[:, 0].min()) // 2)
        self._vc = int((kit_mask[:, 1].max() + kit_mask[:, 1].min()) // 2)

        if self._augment:
            shape = (self._W, self._H)
            angle = np.radians(np.random.uniform(0, 360))
            tu, tv = self._sample_translation([kit_mask], angle)
            aff_1 = np.eye(3)
            aff_1[:2, 2] = [-self._vc, -self._uc]
            aff_2 = gen_rot_mtx_anticlockwise(angle)
            aff_2[:2, 2] = [tu, tv]
            aff_3 = np.eye(3, 3)
            aff_3[:2, 2] = [self._vc, self._uc]
            affine = aff_3 @ aff_2 @ aff_1
            affine = affine[:2, :]
            c_height = cv2.warpAffine(c_height, affine, shape, flags=cv2.INTER_NEAREST)
            d_height = cv2.warpAffine(d_height, affine, shape, flags=cv2.INTER_NEAREST)

            aff_1[:2, 2] = [-self._uc, -self._vc]
            aff_2 = gen_rot_mtx_anticlockwise(-angle)
            aff_2[:2, 2] = [tv, tu]
            aff_3[:2, 2] = [self._uc, self._vc]
            affine = aff_3 @ aff_2 @ aff_1
            affine = affine[:2, :]
            pos_placement = (affine @ np.hstack((pos_placement, np.ones((len(pos_placement), 1)))).T).T
            kit_mask = (affine @ np.hstack((kit_mask, np.ones((len(kit_mask), 1)))).T).T

        # update center of rotation
        self._uc = int((kit_mask[:, 0].max() + kit_mask[:, 0].min()) // 2)
        self._vc = int((kit_mask[:, 1].max() + kit_mask[:, 1].min()) // 2)

        if self._background_subtract is not None:
            idxs = np.vstack(np.where(d_height > self._background_subtract[0])).T
            mask = np.zeros_like(d_height)
            mask[idxs[:, 0], idxs[:, 1]] = 1
            mask = misc.largest_cc(np.logical_not(mask))
            idxs = np.vstack(np.where(mask == 1)).T
            c_height[idxs[:, 0], idxs[:, 1]] = 0
            d_height[idxs[:, 0], idxs[:, 1]] = 0
            idxs = np.vstack(np.where(d_height_i > self._background_subtract[0])).T
            mask = np.zeros_like(d_height)
            mask[idxs[:, 0], idxs[:, 1]] = 1
            mask = misc.largest_cc(np.logical_not(mask))
            idxs = np.vstack(np.where(mask == 1)).T
            c_height_i[idxs[:, 0], idxs[:, 1]] = 0
            d_height_i[idxs[:, 0], idxs[:, 1]] = 0

        # expand to proper dim
        if not self._use_color:
            assert c_height.ndim == 2
            if self._num_channels == 2:
                c_height = c_height[..., np.newaxis]
                c_height_i = c_height_i[..., np.newaxis]
            else:  # clone the gray channel 3 times
                c_height = np.repeat(c_height[..., np.newaxis], 3, axis=-1)
                c_height_i = np.repeat(c_height_i[..., np.newaxis], 3, axis=-1)
        else:
            assert c_height.ndim == 3 and c_height.shape[2] == 3
        d_height = d_height[..., np.newaxis]
        d_height_i = d_height_i[..., np.newaxis]

        # convert heightmaps tensors
        c_height = self._c_norm(self._transform(c_height))
        d_height = self._d_norm(self._transform(d_height))
        c_height_i = self._c_norm(self._transform(c_height_i))
        d_height_i = self._d_norm(self._transform(d_height_i))

        # concatenate height and depth into a 4-channel tensor
        # img_tensor = torch.cat([c_height, d_height], dim=0)
        img_tensor_i = torch.cat([c_height_i, d_height_i], dim=0)
        img_tensor = torch.cat([c_height, d_height], dim=0)
        img_tensor = torch.stack([img_tensor_i, img_tensor], dim=0)  # TODO:is img_tensor_i is nesssarry? batch_size = 2?

        # add columns of 1 (positive labels)
        pos_label = np.hstack((pos_placement, np.ones((len(pos_placement), 1))))

        # generate negative labels
        neg_placement = np.vstack(self._sample_negative(pos_label)).T
        neg_label = np.hstack((neg_placement, np.zeros((len(neg_placement), 1))))

        # stack positive and negative into a single array
        label = np.vstack((pos_label, neg_label))

        neg_placement_i = self._sample_free_negative(kit_mask)
        neg_label_i = np.hstack((neg_placement_i, np.zeros((len(neg_placement_i), 1))))

        label_tensor_i = torch.LongTensor(neg_label_i) # TODO：shape？
        label_tensor_f = torch.LongTensor(label)
        label_tensor = [label_tensor_i, label_tensor_f]

        # convert suction points to tensors
        # label_tensor = torch.LongTensor(label)

        return img_tensor, label_tensor


def get_placement_loader(
    foldername,
    dtype="train",
    batch_size=1,
    sample_ratio=1.0,
    shuffle=True,
    stateless=True,
    augment=False,
    background_subtract=None,
    use_color=True,
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
        imgs = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        imgs = torch.cat(imgs, dim=0)
        labels = [l for sublist in labels for l in sublist]
        return [imgs, labels]

    num_workers = min(num_workers, multiprocessing.cpu_count())
    root = os.path.join(foldername, dtype)

    dataset = PlacementDataset(
        root,
        sample_ratio,
        stateless,
        augment,
        background_subtract,
        use_color,
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