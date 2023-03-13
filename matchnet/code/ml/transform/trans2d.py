import torch
import numpy as np
import cv2
from tools.matrix import gen_rot_mtx_anticlockwise
from functools import reduce
# from abc import ABC

class RanRotTranslation():
    def __init__(self):
        """init"""
        pass

    def _get_valid_idxs(self, corr, rows, cols):
        positive_cond = np.logical_and(corr[:, 0] >= 0, corr[:, 1] >= 0)
        within_cond = np.logical_and(corr[:, 0] < rows, corr[:, 1] < cols)
        valid_idxs = reduce(np.logical_and, [positive_cond, within_cond])
        return valid_idxs
    
    def _corres_rot_translation(self, info_dict, affine_matrix):
        # random rotation & translation on label

        for corrs in info_dict["corres"]:
            source_corrs = corrs[:,0:2].astype("float64")
            target_corrs = corrs[:,2:4].astype("float64")
            source_corrs = (affine_matrix @ np.hstack((source_corrs, np.ones((len(source_corrs), 1)))).T).T
            # remove invalid indices
            valid_target_idxs = self._get_valid_idxs(target_corrs, self._H, self._W)
            target_corrs = target_corrs[valid_target_idxs].astype("int64")
            source_corrs = source_corrs[valid_target_idxs].astype("int64")
            corrs = np.hstack((source_corrs, target_corrs))
        return info_dict

    def _mask_rot_translation(self, info_dict, affine_matrix):
        for name in ["hole", "kit_with_hole", "kit_no_hole"]:
            mask = info_dict[name]
            ones = np.ones((len(mask), 1))
            masks = (affine_matrix @ np.hstack((masks, ones)).T).T
            info_dict[name] = mask
        return info_dict


    def __call__(self,info_dict, kit_vc, kit_uc,shape):
        """Update mask ,corres and angle for random rot and translation."""
        # determine bounds on translation for source and target
        sources = [info_dict["kit_no_hole"]]
        
        # random rotation & translation on image
        angle_s = np.radians(np.random.uniform(0, 360))
        tu_s, tv_s = self._sample_translation(sources, angle_s)
        aff_1 = np.eye(3)
        aff_1[:2, 2] = [-kit_vc, -kit_uc]
        aff_2 = gen_rot_mtx_anticlockwise(angle_s) # anticlockwise in uv coord, but clockwise in opencv xy coord
        aff_2[:2, 2] = [tv_s, tu_s]
        aff_3 = np.eye(3, 3)
        aff_3[:2, 2] = [kit_vc, kit_uc]
        affine_s = aff_3 @ aff_2 @ aff_1
        affine_s = affine_s[:2, :]
        c_height_s = cv2.warpAffine(c_height_s, affine_s, shape, flags=cv2.INTER_NEAREST)
        d_height_s = cv2.warpAffine(d_height_s, affine_s, shape, flags=cv2.INTER_NEAREST)

        aff_1[:2, 2] = [-kit_uc, -kit_vc]
        aff_2 = gen_rot_mtx_anticlockwise(-angle_s)
        aff_2[:2, 2] = [tv_s, tu_s]
        aff_3[:2, 2] = [kit_uc, kit_vc]
        affine_s = aff_3 @ aff_2 @ aff_1
        affine_s = affine_s[:2, :]
        # update corres
        info_dict = self._corres_rot_translation(info_dict, affine_s)

        # apply affine transformation to masks in source
        self._mask_rot_translation(info_dict, affine_s)

        # update gd_truth_rot
        for i, rot in enumerate(info_dict["delta_angle"]):
            info_dict["delta_angle"][i] = rot - np.degrees(angle_s)

        return info_dict

    def _sample_translation(self, corrz, angle,kit_uc, kit_vc):
        # calculate valid offset range of translation, then selected one from it.
        # vailid range [10:-10]
        aff_1 = np.eye(3)
        aff_1[:2, 2] = [-kit_uc, -kit_vc]
        aff_2 = gen_rot_mtx_anticlockwise(-angle)
        aff_3 = np.eye(3, 3)
        aff_3[:2, 2] = [kit_uc, kit_vc]
        affine = aff_3 @ aff_2 @ aff_1
        affine = affine[:2, :]
        corrs = []
        for corr in corrz:
            ones = np.ones((len(corr), 1))
            corrs.append((affine @ np.hstack((corr, ones)).T).T)
        max_vv = corrs[0][:, 1].max()
        # max_vu = corrs[0][corrs[0][:, 1].argmax()][0]
        min_vv = corrs[0][:, 1].min()
        # min_vu = corrs[0][corrs[0][:, 1].argmin()][0]
        max_uu = corrs[0][:, 0].max()
        # max_uv = corrs[0][corrs[0][:, 0].argmax()][1]
        min_uu = corrs[0][:, 0].min()
        # min_uv = corrs[0][corrs[0][:, 0].argmin()][1]
        for t in corrs[1:]:
            if t[:, 1].max() > max_vv:
                max_vv = t[:, 1].max()
                # max_vu = t[t[:, 1].argmax()][0]
            if t[:, 1].min() < min_vv:
                min_vv = t[:, 1].min()
                # min_vu = t[t[:, 1].argmin()][0]
            if t[:, 0].max() > max_uu:
                max_uu = t[:, 0].max()
                # max_uv = t[t[:, 0].argmax()][1]
            if t[:, 0].min() < min_uu:
                min_uu = t[:, 0].min()
                # min_uv = t[t[:, 0].argmin()][1]
        #  
        tv = np.random.uniform(-min_vv + 10, self._W - max_vv - 10)
        tu = np.random.uniform(-min_uu + 10, self._H - max_uu - 10)
        return tu, tv
        

class ChangeBG():
    def __init__():
        """"""
        
    def __call__(fg_mask, ):
        """"""

class ColorJit():
    def __init__():
        """"""
        
    def __call__():
        """"""

class TensorNorm():
    def __init__():
        """"""
        
    def __call__():
        """"""