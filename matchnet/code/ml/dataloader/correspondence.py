"""The correspondence network dataloader.
"""

import glob
import multiprocessing
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# from functools import reduce
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from matchnet.code.utils import misc, viz
from matchnet.code.utils import sampling
# from tools.matrix import gen_rot_mtx_anticlockwise


class CorrespondenceDataset(Dataset):
    """The correspondence network dataset.
    """

    def __init__(
        self,
        root,
        sample_ratio,
        num_rotations,
        stateless,
        augment,
        background_subtract:bool,
        use_color:bool,
        num_channels:int,
    ):
        """Initializes the dataset.

        Args:
            root: (str) Root directory path.
            sample_ratio: (float) The ratio of negative to positive labels.
            num_rotations: (int) The number of discrete rotation levels to consider.
            stateless: (bool) If `True`, only consider correspondences
                from the current timestep. Else, use correspondences
                from all previous and current timestep.
            augment: (bool) Whether to apply data augmentation.
            background_subtract: (bool) Whether to apply background subtraction.
            use_color: (bool) Whether to use color image.
            num_channels: (int) valid when not use_color, 
                4 means clones the grayscale image to produce an RGB image.
        """
        self._root = root
        self._num_rotations = num_rotations
        self._stateless = stateless
        self._sample_ratio = sample_ratio
        self._augment = augment
        self._background_subtract = background_subtract
        self._num_channels = num_channels
        self._use_color = use_color

        # figure out how many data samples we have
        self._get_filenames()

        # generate rotation increments
        self._rot_step_size = 360 / num_rotations
        self._rotations = np.array([self._rot_step_size * i for i in range(num_rotations)])
        # load per-channel mean and std
        self._load_norm_info()
        self._c_norm = transforms.Normalize(mean=self.c_mean, std=self.c_std)
        self._d_norm = transforms.Normalize(mean=self.d_mean, std=self.d_std)
        self._to_tensor = transforms.ToTensor()

    def _load_norm_info(self,):
        norm_info = pickle.load(open(os.path.join(Path(self._root).parent, "mean_std.pkl"), "rb"))
        self.norm_info = norm_info
        # color
        if self._use_color:
            color_key = "color"
        else:
            color_key = "gray"
        self.c_mean = norm_info[color_key]["mean"]
        self.c_std = norm_info[color_key]["std"]
        if not self._use_color and self._num_channels == 4:
            self.c_mean = self.c_mean*3
            self.c_std = self.c_std *3
        # depth
        self.d_mean = norm_info["depth"]["mean"]
        self.d_std = norm_info["depth"]["std"]

    def __len__(self):
        return len(self._filenames)

    def _get_filenames(self):
        """Returns a list of filenames to process.
        """
        self._filenames = glob.glob(os.path.join(self._root, "*/"))
        # self._filenames.sort(key=lambda x: int(x.split("/")[-2]))

    def _load_state(self, name):
        # load visual
        if self._use_color:
            color_name = "color.png"
        else:
            color_name = "gray.png"
        depth_name = "depth.png"
            
        # c_height_i = cv2.imread(os.path.join(name, "init_" + color_name),cv2.IMREAD_UNCHANGED)
        # d_height_i = cv2.imread(os.path.join(name, "init_" + depth_name),cv2.IMREAD_UNCHANGED)
        c_height_f = cv2.imread(os.path.join(name, "final_" + color_name),cv2.IMREAD_UNCHANGED)
        d_height_f = cv2.imread(os.path.join(name, "final_" + depth_name),cv2.IMREAD_UNCHANGED)

        # # convert depth to meters
        # d_height_i = (d_height_i * 1e-3).astype("float32")
        # d_height_f = (d_height_f * 1e-3).astype("float32")

        # load info_dict
        info_dict = pickle.load(open(os.path.join(name, "info_dict.pkl"),"rb"))
        info_dict["c_height_f"] = c_height_f
        info_dict["d_height_f"] = d_height_f
        
        if self._stateless:
            info_dict["corres"] = info_dict["corres"][-1:]

        return info_dict

    def _split_heightmap(self, info_dict, key):
        """Splits a heightmap into a source and target.
        """
        height = info_dict[key]
        half = height.shape[1] // 2
        key_prefix = key[:-1]
        info_dict[f"{key_prefix}_s"] = height[:, half:]
        info_dict[f"{key_prefix}_t"] = height[:, :half]

        return info_dict, half

    def _compute_relative_rotation(self, pose_i, pose_f):
        """Computes the relative z-axis rotation between two poses.

        Returns:
            (float) The angle in degrees.
        """
        transform = pose_f @ np.linalg.inv(pose_i)
        rotation = np.rad2deg(misc.rotz2angle(transform))
        return rotation

    def _quantize_rotation(self, true_rot):
        """Bins the true rotation into one of `num_rotations`.
        Returns:
            (int) An index from 0 to `num_rotations` - 1.
        """
        angle = true_rot - (360 * np.floor(true_rot * (1 / 360)))
        # angle = (true_rot % 360 + 360) % 360

        # since 0 = 360 degrees, we need to remap
        # any indices in the last quantization
        # bracket to 0 degrees.
        if angle > (360 - (0.5 * self._rot_step_size)) and angle <= 360:
            return 0

        return np.argmin(np.abs(self._rotations - angle))

    def _process_correspondences(self, corrs, rot_idx, append=True):
        """Processes correspondences for a given rotation. 
            Rotate source label as quanti rot angle, filter valid point.
        """
        # split correspondences into source and target
        source_corrs = corrs[:, 0:2]
        target_corrs = corrs[:, 2:4]

        # rotate source indices clockwise
        source_idxs = misc.rotate_uv(source_corrs, -self._rot_step_size * rot_idx, self._H, self._W, (kit_uc, kit_vc))
        target_idxs = np.round(target_corrs)

        # remove any repetitions
        _, unique_idxs = np.unique(source_idxs, return_index=True, axis=0)
        source_idxs_unique = source_idxs[unique_idxs]
        target_idxs_unique = target_idxs[unique_idxs]
        _, unique_idxs = np.unique(target_idxs_unique, return_index=True, axis=0)
        source_idxs_unique = source_idxs_unique[unique_idxs]
        target_idxs_unique = target_idxs_unique[unique_idxs]

        # remove indices that exceed image bounds
        valid_idxs = np.logical_and(
            target_idxs_unique[:, 0] < self._H,
            np.logical_and(
                target_idxs_unique[:, 1] < self._W,
                np.logical_and(
                    target_idxs_unique[:, 0] >= 0, target_idxs_unique[:, 1] >= 0
                ),
            ),
        )
        target_idxs = target_idxs_unique[valid_idxs]
        source_idxs = source_idxs_unique[valid_idxs]
        valid_idxs = np.logical_and(
            source_idxs[:, 0] < self._H,
            np.logical_and(
                source_idxs[:, 1] < self._W,
                np.logical_and(source_idxs[:, 0] >= 0, source_idxs[:, 1] >= 0),
            ),
        )
        target_idxs = target_idxs[valid_idxs].astype("int")
        source_idxs = source_idxs[valid_idxs].astype("int")

        if append:
            self._features_source.append(source_idxs)
            self._features_target.append(target_idxs)
            self._rot_idxs.append(np.repeat([rot_idx], len(source_idxs)))
            self._is_match.append(np.ones(len(source_idxs)))
        else:
            return np.hstack([source_idxs, target_idxs])

    # def _get_valid_idxs(self, corr, rows, cols):
    #     positive_cond = np.logical_and(corr[:, 0] >= 0, corr[:, 1] >= 0)
    #     within_cond = np.logical_and(corr[:, 0] < rows, corr[:, 1] < cols)
    #     valid_idxs = reduce(np.logical_and, [positive_cond, within_cond])
    #     return valid_idxs

    # def _sample_translation(self, corrz, angle,kit_uc, kit_vc ):
    #     # calculate valid offset range of translation, then selected one from it.
    #     # vailid range [10:-10]
    #     aff_1 = np.eye(3)
    #     aff_1[:2, 2] = [-kit_uc, -kit_vc]
    #     aff_2 = gen_rot_mtx_anticlockwise(-angle)
    #     aff_3 = np.eye(3, 3)
    #     aff_3[:2, 2] = [kit_uc, kit_vc]
    #     affine = aff_3 @ aff_2 @ aff_1
    #     affine = affine[:2, :]
    #     corrs = []
    #     for corr in corrz:
    #         ones = np.ones((len(corr), 1))
    #         corrs.append((affine @ np.hstack((corr, ones)).T).T)
    #     max_vv = corrs[0][:, 1].max()
    #     max_vu = corrs[0][corrs[0][:, 1].argmax()][0]
    #     min_vv = corrs[0][:, 1].min()
    #     min_vu = corrs[0][corrs[0][:, 1].argmin()][0]
    #     max_uu = corrs[0][:, 0].max()
    #     max_uv = corrs[0][corrs[0][:, 0].argmax()][1]
    #     min_uu = corrs[0][:, 0].min()
    #     min_uv = corrs[0][corrs[0][:, 0].argmin()][1]
    #     for t in corrs[1:]:
    #         if t[:, 1].max() > max_vv:
    #             max_vv = t[:, 1].max()
    #             max_vu = t[t[:, 1].argmax()][0]
    #         if t[:, 1].min() < min_vv:
    #             min_vv = t[:, 1].min()
    #             min_vu = t[t[:, 1].argmin()][0]
    #         if t[:, 0].max() > max_uu:
    #             max_uu = t[:, 0].max()
    #             max_uv = t[t[:, 0].argmax()][1]
    #         if t[:, 0].min() < min_uu:
    #             min_uu = t[:, 0].min()
    #             min_uv = t[t[:, 0].argmin()][1]
    #     #  
    #     tv = np.random.uniform(-min_vv + 10, self._W - max_vv - 10)
    #     tu = np.random.uniform(-min_uu + 10, self._H - max_uu - 10)
    #     return tu, tv

    def mask_left_half_move(self, info_dict, half):
        # no need to move mask on the left
        # three mask on the right
        for name in ["hole", "kit_with_hole", "kit_no_hole"]:
            info_dict[name][:,1] = info_dict[name][:,1] - half
        # corres only needs change v coord of kit]
        for name in ["hole", "kit_with_hole", "kit_no_hole","corres"]:
            if isinstance(info_dict[name], np.ndarray):
                info_dict[name][:,1] = info_dict[name][:,1] - half
            elif isinstance(info_dict[name], list) and isinstance(info_dict[name][0], np.ndarray):
                for i, mask in enumerate(info_dict[name]):
                    info_dict[name][i][:,1] = mask[:,1] - half
            else:
                raise RuntimeError("Expect info_dict[%s] is array or list of array, but get %s",name, type(info_dict[name]))
        return info_dict



    def __getitem__(self, idx):
        name = self._filenames[idx]

        # load states
        info_dict = self._load_state(self._filenames[idx])
        
        
        # split heightmap into source and target
        info_dict, half = self._split_heightmap(info_dict,"c_height_f")
        info_dict, _ = self._split_heightmap(info_dict,"d_height_f")
        self._H, self._W = c_height_t.shape[:2]

        info_dict = self.mask_left_half_move(info_dict, half)

        # # partition correspondences into current and previous
        # curr_corrs = all_corrs[-1]
        # prev_corrs = all_corrs[:-1]

        # center of rotation is the center of the kit
        kit_uc = int((info_dict["kit_no_hole"][:, 0].max() + info_dict["kit_no_hole"][:, 0].min()) // 2)
        kit_vc = int((info_dict["kit_no_hole"][:, 1].max() + info_dict["kit_no_hole"][:, 1].min()) // 2)

        if self._augment:
            # source_corrs = curr_corrs[:, 0:2].astype("float64")
            # target_corrs = curr_corrs[:, 2:4].astype("float64")

            # # determine bounds on translation for source and target
            # sources = [info_dict["kit_no_hole"]]

            # # random rotation & translation on image
            # angle_s = np.radians(np.random.uniform(0, 360))
            # tu_s, tv_s = self._sample_translation(sources, angle_s)
            # aff_1 = np.eye(3)
            # aff_1[:2, 2] = [-kit_vc, -kit_uc]
            # aff_2 = gen_rot_mtx_anticlockwise(angle_s) # anticlockwise in uv coord, but clockwise in opencv xy coord
            # aff_2[:2, 2] = [tv_s, tu_s]
            # aff_3 = np.eye(3, 3)
            # aff_3[:2, 2] = [kit_vc, kit_uc]
            # affine_s = aff_3 @ aff_2 @ aff_1
            # affine_s = affine_s[:2, :]
            # c_height_s = cv2.warpAffine(c_height_s, affine_s, shape, flags=cv2.INTER_NEAREST)
            # d_height_s = cv2.warpAffine(d_height_s, affine_s, shape, flags=cv2.INTER_NEAREST)

            # # random rotation & translation on label
            # aff_1[:2, 2] = [-kit_uc, -kit_vc]
            # aff_2 = gen_rot_mtx_anticlockwise(-angle_s)
            # aff_2[:2, 2] = [tv_s, tu_s]
            # aff_3[:2, 2] = [kit_uc, kit_vc]
            # affine_s = aff_3 @ aff_2 @ aff_1
            # affine_s = affine_s[:2, :]
            # source_corrs = (affine_s @ np.hstack((source_corrs, np.ones((len(source_corrs), 1)))).T).T

            # # remove invalid indices
            # valid_target_idxs = self._get_valid_idxs(target_corrs, self._H, self._W)
            # target_corrs = target_corrs[valid_target_idxs].astype("int64")
            # source_corrs = source_corrs[valid_target_idxs].astype("int64")
            # curr_corrs = np.hstack((source_corrs, target_corrs))

            # # apply affine transformation to masks in source
            # masks = [hole_mask, info_dict["kit_no_hole"], kit_minus_hole_mask]
            # for i in range(len(masks)):
            #     ones = np.ones((len(masks[i]), 1))
            #     masks[i] = (affine_s @ np.hstack((masks[i], ones)).T).T
            # hole_mask, info_dict["kit_no_hole"], kit_minus_hole_mask = masks

            # # update prev_corrs if not stateless
            # if not self._stateless:
            #     for i, corrs in enumerate(prev_corrs):
            #         corrs = corrs[:, 0:2].astype("float64")
            #         corrs = (affine_s @ np.hstack((corrs, np.ones((len(corrs), 1)))).T).T
            #         prev_corrs[i][:, 0:2] = corrs

            # # update gd_truth_rot
            # gd_truth_rot = gd_truth_rot - np.degrees(angle_s)

        # reupdate kit mask center
        kit_uc = int((info_dict["kit_no_hole"][:, 0].max() + info_dict["kit_no_hole"][:, 0].min()) // 2)
        kit_vc = int((info_dict["kit_no_hole"][:, 1].max() + info_dict["kit_no_hole"][:, 1].min()) // 2)

        
        # quantize rotation
        curr_rot_idx = self._quantize_rotation(gd_truth_rot)

        self._features_source = []
        self._features_target = []
        self._rot_idxs = []
        self._is_match = []

        # sample matches from all previous timesteps if not stateless
        if not self._stateless:
            for rot_idx, corrs in zip(self._pre_rot_quant_indices, prev_corrs):
                self._process_correspondences(corrs, rot_idx)

        # sample matches from the current timestep
        self._process_correspondences(curr_corrs, curr_rot_idx, depth=d_height_t)

        # determine the number of non-matches to sample per rotation
        num_matches = 0
        for m in self._is_match:
            num_matches += len(m)
        num_non_matches = int(self._sample_ratio * num_matches / self._num_rotations)

        # convert masks to linear indices for sampling
        all_idxs_1d = np.arange(0, self._H * self._W)
        object_target_1d = sampling.make1d(object_mask, self._W)
        background_target_1d = np.array(list((set(all_idxs_1d) - set(object_target_1d))))
        hole_source_1d = sampling.make1d(hole_mask, self._W)
        kit_minus_hole_source_1d = sampling.make1d(kit_minus_hole_mask, self._W)
        kit_plus_hole_source_1d = sampling.make1d(info_dict["kit_no_hole"], self._W)
        background_source_1d = np.array(list(set(all_idxs_1d) - set(kit_plus_hole_source_1d)))
        # remove the part of box of kit
        background_source_1d = sampling.remove_outliers(background_source_1d, info_dict["kit_no_hole"], self._W)

        # sample non-matches
        temp_idx = 0
        div_factor = 5
        for rot_idx in range(self._num_rotations):
            non_matches = []

            # source: anywhere
            # target: anywhere but the object
            non_matches.append(sampling.sample_non_matches(
                1 * num_non_matches // div_factor,
                (self._H, self._W),
                -self._rotations[rot_idx],
                mask_target=background_target_1d,
                rotate=False)
            )

            # source: anywhere but the kit
            # target: on the object
            nm_idxs = sampling.sample_non_matches(
                1 * num_non_matches // div_factor,
                (self._H, self._W),
                -self._rotations[rot_idx],
                background_source_1d,
                object_target_1d,
                rotate=False,
            )
            non_matches.append(nm_idxs)

            # source: on the kit but not in the hole
            # target: on the object
            nm_idxs = sampling.sample_non_matches(
                1 * num_non_matches // div_factor,
                (self._H, self._W),
                -self._rotations[rot_idx],
                kit_minus_hole_source_1d,
                object_target_1d,
                cxcy=(kit_uc, kit_vc),
            )
            non_matches.append(nm_idxs)

            # here, I want to explicity samples matches
            # for the incorrect rotations to teach
            # the network that in fact, this is
            # the incorrect rotation.
            # This is especially useful for the
            # 180 degree rotated version of the
            # correct rotation.
            if rot_idx != curr_rot_idx:
                nm_idxs = self._process_correspondences(curr_corrs, rot_idx, False)
                subset_mask = np.random.choice(np.arange(len(nm_idxs)), replace=False, size=(1 * num_non_matches // div_factor))
                nm_idxs = nm_idxs[subset_mask]
                non_matches.append(nm_idxs)

            # source: in the hole
            # target: on the object
            if self._stateless:
                if rot_idx == curr_rot_idx:
                    nm_idxs = sampling.non_matches_from_matches(
                        1 * num_non_matches // div_factor,
                        (self._H, self._W),
                        -self._rotations[rot_idx],
                        hole_source_1d,
                        self._features_source[0],
                        self._features_target[0],
                        cxcy=(kit_uc, kit_vc),
                    )
                    non_matches.append(nm_idxs)
                else:
                    nm_idxs = sampling.sample_non_matches(
                        1 * num_non_matches // div_factor,
                        (self._H, self._W),
                        -self._rotations[rot_idx],
                        hole_source_1d,
                        object_target_1d,
                        cxcy=(kit_uc, kit_vc),
                    )
                    non_matches.append(nm_idxs)
            else:
                if rot_idx in self._pre_rot_quant_indices:
                    non_matches.append(
                        sampling.non_matches_from_matches(
                            num_non_matches // div_factor,
                            (self._H, self._W),
                            -self._rotations[rot_idx],
                            hole_source_1d,
                            self._features_source[temp_idx],
                            self._features_target[temp_idx],
                        )
                    )
                    temp_idx += 1
                else:
                    non_matches.append(
                        sampling.sample_non_matches(
                            num_non_matches // div_factor,
                            (self._H, self._W),
                            -self._rotations[rot_idx],
                            hole_source_1d,
                            object_target_1d,
                        )
                    )
            non_matches = np.vstack(non_matches)
            self._features_source.append(non_matches[:, :2])
            self._features_target.append(non_matches[:, 2:])
            self._rot_idxs.append(np.repeat([rot_idx], len(non_matches)))
            self._is_match.append(np.repeat([0], len(non_matches)))

        # convert lists to numpy arrays
        self._features_source = np.concatenate(self._features_source)
        self._features_target = np.concatenate(self._features_target)
        self._rot_idxs = np.concatenate(self._rot_idxs)[..., np.newaxis]
        self._is_match = np.concatenate(self._is_match)[..., np.newaxis]

        # concatenate into 1 big array
        label = np.hstack(
            (
                self._features_source,
                self._features_target,
                self._rot_idxs,
                self._is_match,
            )
        )

        # expand to proper dim
        if not self._use_color:
            assert c_height_s.ndim == 2
            if self._num_channels == 2:
                c_height_s = c_height_s[..., np.newaxis]
                c_height_t = c_height_t[..., np.newaxis]
            else:  # clone the gray channel 3 times
                c_height_s = np.repeat(c_height_s[..., np.newaxis], 3, axis=-1)
                c_height_t = np.repeat(c_height_t[..., np.newaxis], 3, axis=-1)
        else:
            assert c_height_s.ndim == 3 and c_height_s.shape[2] == 3
        d_height_s = d_height_s[..., np.newaxis]
        d_height_t = d_height_t[..., np.newaxis]


        # ndarray -> tensor
        label_tensor = torch.LongTensor(label)

        # heightmaps -> tensor
        c_height_s = self._c_norm(self._to_tensor(c_height_s))
        c_height_t = self._c_norm(self._to_tensor(c_height_t))
        d_height_s = self._d_norm(self._to_tensor(d_height_s))
        d_height_t = self._d_norm(self._to_tensor(d_height_t))

        # concatenate height and depth into a 4-channel tensor
        source_img_tensor = torch.cat([c_height_s, d_height_s], dim=0)
        target_img_tensor = torch.cat([c_height_t, d_height_t], dim=0)

        # concatenate source and target into a 8-channel tensor
        img_tensor = torch.cat([source_img_tensor, target_img_tensor], dim=0)

        kit_center = (kit_uc, kit_vc)
        return img_tensor, label_tensor, kit_center, obj_center


def get_corr_loader(
    foldername,
    dtype="train",
    batch_size=1,
    shuffle=True,
    sample_ratio=1.0,
    num_rotations=20,
    stateless=True,
    augment=False,
    background_subtract=None,
    use_color = True,
    num_channels=2,
    num_workers=8,
):
    """Returns a dataloader over the correspondence dataset.

    Args:
        foldername: (str) The name of the folder containing the data.
        dtype: (str) Whether to use the train, validation or test partition.
        shuffle: (bool) Whether to shuffle the dataset at the end
            of every epoch.
        sample_ratio: (float) The ratio of negative to positive
            labels.
        num_rotations: (int) The number of discrete rotation levels
            to consider.
        stateless: (bool) If `True`, only consider correspondences
            from the current timestep. Else, use correspondences
            from all previous and current timestep.
        background_subtract: (bool) Whether to apply background subtraction.
        num_channels: (int) 4 clones the grayscale image to produce an RGB image.
        num_workers: (int) How many processes to use. Each workers
            is responsible for loading a batch.
    """

    def _collate_fn(batch):
        """A custom collate function.

        This is to support variable length correspondence labels.
        """
        imgs = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        kit_centers = [b[2] for b in batch]
        obj_centers = [b[3] for b in batch]
        # mask = [b[2] for b in batch]
        # kit_mask = [b[3] for b in batch]
        imgs = torch.stack(imgs, dim=0)
        max_num_label = labels[0].shape[0]
        for l in labels[1:]:
            if l.shape[0] > max_num_label:
                max_num_label = l.shape[0]
        new_labels = []
        for l in labels:
            if l.shape[0] < max_num_label:
                l_pad = torch.cat([l, torch.LongTensor([999]).repeat(max_num_label - l.shape[0], 6)], dim=0)
                new_labels.append(l_pad)
            else:
                new_labels.append(l)
        labels = torch.stack(new_labels, dim=0)
        return [imgs, labels, kit_centers,obj_centers]

    num_workers = min(num_workers, multiprocessing.cpu_count())
    root = os.path.join(foldername,dtype)

    dataset = CorrespondenceDataset(
        root,
        sample_ratio,
        num_rotations,
        stateless,
        augment,
        background_subtract,
        use_color,
        num_channels,
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
