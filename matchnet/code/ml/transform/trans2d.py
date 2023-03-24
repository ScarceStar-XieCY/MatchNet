import torch
import numpy as np
import cv2
from tools.matrix import gen_rot_mtx_anticlockwise
from functools import reduce
import pickle
from torchvision import transforms
# from abc import ABC
import os
from pathlib import Path
from tools.image_mask.mask_process import apply_mask_to_img,coord2mask



def _key_list_hook(key_list):
    "Convert string to list if needed"
    if isinstance(key_list, list):
        return key_list
    if isinstance(key_list, str):
        return [key_list]


class SplitTarSour():
    """Split image to traget and source. And Update 
    all types of mask.
    
    
    c_height_f -> c_height_s c_height_t
    d_height_f -> d_height_s d_height_t
    """
    def __init__(self):
        pass
    
    @staticmethod
    def mask_left_half_move(info_dict):
        """move all types of mask to the left."""
        # no need to move mask on the left
        # three mask on the right
        half = info_dict["half"]
        for name in ["hole", "kit_with_hole", "kit_no_hole","corres"]:
            if isinstance(info_dict[name], np.ndarray):
                info_dict[name][:,1] -= half
            elif isinstance(info_dict[name], list) and isinstance(info_dict[name][0], np.ndarray):
                for i, mask in enumerate(info_dict[name]):
                    if mask[:,1].max() <= half:
                        continue
                    info_dict[name][i][:,1] -= half
            else:
                raise RuntimeError("Expect info_dict[%s] is array or list of array, but get %s",name, type(info_dict[name]))
        return info_dict
    
    @staticmethod
    def split_heightmap(info_dict, key):
        """Splits a heightmap into a source and target.
        Get half and save in info_dict.
        key:key of image to be splited.
        """
        height = info_dict[key]
        key_prefix = key[:-2]
        if isinstance(height,np.ndarray): # HWC
            half = height.shape[1] // 2
            info_dict[f"{key_prefix}_s"] = height[:, half:]
            info_dict[f"{key_prefix}_t"] = height[:, :half]
        elif isinstance(height,torch.Tensor): #CHW
            half = height.shape[2] // 2
            info_dict[f"{key_prefix}_s"] = height[:, :,half:]
            info_dict[f"{key_prefix}_t"] = height[:, :,:half]
        info_dict["half"] = half
        return info_dict
    

    def __call__(self,info_dict):
        # have to split first then move mask
        info_dict = self.split_heightmap(info_dict,"c_height_f")
        info_dict = self.split_heightmap(info_dict,"d_height_f")
        info_dict = self.mask_left_half_move(info_dict)
        return info_dict


class MergeTarSour():
    """Merge image from traget and source. And Update 
    all types of mask.
    
    
    c_height_f <- c_height_s c_height_t
    d_height_f <- d_height_s d_height_t
    """
    def __init__(self):
        pass
    
    @staticmethod
    def mask_right_half_move(info_dict):
        """move all types of mask to the right."""
        # no need to move mask on the left
        # three mask on the right
        half = info_dict["half"]
        # corres only needs change v coord of kit]
        for name in ["hole", "kit_with_hole", "kit_no_hole","corres"]:
            if isinstance(info_dict[name], np.ndarray):
                info_dict[name][:,1] += half
            elif isinstance(info_dict[name], list) and isinstance(info_dict[name][0], np.ndarray):
                for i, mask in enumerate(info_dict[name]):
                    if mask[:,1].min() >= half:
                        continue
                    info_dict[name][i][:,1] += half
            else:
                raise RuntimeError("Expect info_dict[%s] is array or list of array, but get %s",name, type(info_dict[name]))
        return info_dict
    
    @staticmethod
    def merge_heightmap(info_dict, key):
        """Splits a heightmap into a source and target.
        key: key of merged image.
        """
        key_prefix = key[:-2]

        right_half = info_dict[f"{key_prefix}_s"]
        left_half = info_dict[f"{key_prefix}_t"]
        info_dict[f"{key_prefix}_f"] = np.hstack((left_half, right_half))
        # check shape
        if info_dict[f"{key_prefix}_f"].shape[1] != 2 * info_dict["half"]:
            raise RuntimeError("shape dismatch.")
        
        return info_dict
    
    def __call__(self,info_dict):
        # have to split first then move mask
        info_dict = self.merge_heightmap(info_dict,"c_height_f")
        info_dict = self.merge_heightmap(info_dict,"d_height_f")
        info_dict = self.mask_right_half_move(info_dict)
        return info_dict


class RanRotTranslation():
    """Randomly rotation and ranslation kit on right half image and update
    other mask in info_dict.
    """
    def __init__(self,shape):
        """init"""
        # center of rotation is the center of the kit
        self.split_process = SplitTarSour()
        self.merge_process = MergeTarSour()
        self._H = shape[0]
        self._W = shape[1]
        pass

    def _get_valid_idxs(self, corr, rows, cols):
        positive_cond = np.logical_and(corr[:, 0] >= 0, corr[:, 1] >= 0)
        within_cond = np.logical_and(corr[:, 0] < rows, corr[:, 1] < cols)
        valid_idxs = reduce(np.logical_and, [positive_cond, within_cond])
        return valid_idxs
    
    def _corres_rot_translation(self, info_dict, affine_matrix):
        # random rotation & translation on label

        for i,corrs in enumerate(info_dict["corres"]):
            source_corrs = corrs[:,0:2].astype("float64")
            target_corrs = corrs[:,2:4].astype("float64")
            source_corrs = (affine_matrix @ np.hstack((source_corrs, np.ones((len(source_corrs), 1)))).T).T
            # remove invalid indices
            valid_target_idxs = self._get_valid_idxs(source_corrs, self._H, self._W * 2)
            target_corrs = target_corrs[valid_target_idxs].astype("int64")
            source_corrs = source_corrs[valid_target_idxs].astype("int64")
            corrs = np.hstack((source_corrs, target_corrs))
            info_dict[i] = corrs
        return info_dict

    def _mask_rot_translation(self, info_dict, affine_matrix):
        for name in ["hole", "kit_with_hole", "kit_no_hole"]:
            mask = info_dict[name][-1]
            ones = np.ones((len(mask), 1))
            mask = (affine_matrix @ np.hstack((mask, ones)).T).T
            valid_mask_idxs = self._get_valid_idxs(mask, self._H, self._W * 2)
            mask = mask[valid_mask_idxs].astype("int64")
            info_dict[name][-1] = mask
        return info_dict


    def __call__(self,info_dict):
        """Update mask ,corres and angle for random rot and translation."""
        # split image and updata mask
        info_dict = self.split_process(info_dict)
        # determine bounds on translation for source and target
        sources = info_dict["kit_no_hole"][-1:]
        
        kit_uc = int((info_dict["kit_no_hole"][-1][:, 0].max() + info_dict["kit_no_hole"][-1][:, 0].min()) // 2)
        kit_vc = int((info_dict["kit_no_hole"][-1][:, 1].max() + info_dict["kit_no_hole"][-1][:, 1].min()) // 2)

        # random rotation & translation on image
        angle_s = np.random.uniform(0, 360)
        tu_s, tv_s = self._sample_translation(sources, angle_s, kit_uc,kit_vc)
        aff_1 = np.eye(3)
        aff_1[:2, 2] = [-kit_vc, -kit_uc]
        aff_2 = gen_rot_mtx_anticlockwise(angle_s,isdegree=True) # anticlockwise in uv coord, but clockwise in opencv xy coord
        aff_2[:2, 2] = [tv_s, tu_s]
        aff_3 = np.eye(3, 3)
        aff_3[:2, 2] = [kit_vc, kit_uc]
        affine_s = aff_3 @ aff_2 @ aff_1
        affine_s = affine_s[:2, :]
        shape = (self._W, self._H)
        info_dict["c_height_s"] = cv2.warpAffine(info_dict["c_height_s"], affine_s, shape, flags=cv2.INTER_NEAREST,borderMode= cv2.BORDER_REPLICATE)
        info_dict["d_height_s"] = cv2.warpAffine(info_dict["d_height_s"], affine_s, shape, flags=cv2.INTER_NEAREST,borderMode= cv2.BORDER_REPLICATE)
        
        aff_1[:2, 2] = [-kit_uc, -kit_vc]
        aff_2 = gen_rot_mtx_anticlockwise(-angle_s,isdegree=True)
        aff_2[:2, 2] = [tu_s, tv_s]
        aff_3[:2, 2] = [kit_uc, kit_vc]
        affine_s = aff_3 @ aff_2 @ aff_1
        affine_s = affine_s[:2, :]
        # update corres: do same roationt and translation
        info_dict = self._corres_rot_translation(info_dict, affine_s)

        # apply affine transformation to masks in source
        self._mask_rot_translation(info_dict, affine_s)

        # update gd_truth_rot
        for i, rot in enumerate(info_dict["delta_angle"]):
            info_dict["delta_angle"][i] = rot - angle_s
        
        # merge image and update mask
        info_dict = self.merge_process(info_dict)

        # TODO : update init_point
        return info_dict

    def _sample_translation(self, corrz, angle,kit_uc, kit_vc):
        # calculate valid offset range of translation, then selected one from it.
        # vailid range [10:-10]
        aff_1 = np.eye(3)
        aff_1[:2, 2] = [-kit_uc, -kit_vc]
        aff_2 = gen_rot_mtx_anticlockwise(-angle,isdegree=True)
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
    def __init__(self, change_ratio, key_list, shape):
        """Change background in rgb image , remain objects and kit and
          anything of depth unchanged."""
        # bg_path = "" # TODO
        # self._bg_imgs = pickle.load(open(bg_path,"rb"))
        self._H = shape[0]
        self._W = shape[1]
        self._bg_imgs = [np.full((self._H,self._W,3), 128, np.uint8), np.full((self._H,self._W,3), 0, np.uint8), np.full((self._H,self._W,3), 256, np.uint8)]
        self._num_bg_imgs = len(self._bg_imgs)
        self._change_ratio = change_ratio
        self._key_list = _key_list_hook(key_list)

    def get_fg_mask(self,info_dict):
        fg_mask = np.vstack(info_dict["obj"])
        fg_mask = np.vstack((fg_mask, info_dict["kit_no_hole"][-1]))
        if fg_mask.ndim != 2 or fg_mask.shape[1] != 2:
            raise RuntimeError(f"fg_mask shape mismatch, expect (N,2), but get {fg_mask.shape}")
        fg_image = coord2mask(np.floor(fg_mask).astype(np.int64), self._H, self._W, False)
        return fg_image

    @staticmethod
    def apply_fg_to_bg(mask,fg_imgs,bg_img,visual:bool, mask_info:str):
        """
        用mask把img_list中的图像分割出来,其中mask=0的位置全涂黑,否则使用原图像素值
        :param mask: 二维的二值mask
        :param imgs: 所有图片,可以是单张图片或图片列表
        :param color2gray: 是否把彩色图像转为灰度图像
        :param visual: 是否可视结果
        :param mask_info:mask相关信息,用以生成不同的mask窗口
        :return: 分割后的图像或图像列表
        """
        if fg_imgs.ndim == 3 and fg_imgs.shape[2] == 3:
            img = np.where(np.repeat(mask[:, :, np.newaxis], 3, axis=-1), fg_imgs, bg_img)
        else:
            img = np.where(mask, fg_imgs, bg_img)
        if visual:
            cv2.imshow('apply {} mask to img'.format(mask_info), img)
        return img


    def __call__(self,info_dict):
        """"""
        if self._change_ratio < np.random.rand():
            return info_dict
        fg_mask = self.get_fg_mask(info_dict)
        # randomly select bg imgs
        idx = np.random.choice(self._num_bg_imgs,1)[0]
        bg_img = self._bg_imgs[idx]
        for key in self._key_list:
            info_dict[key] = self.apply_fg_to_bg(fg_mask,info_dict[key],bg_img, False,"")
        return info_dict


class ColorJit():
    """Change color for rgb image. Will convert HWC np.ndarray to CHW tensor."""
    def __init__(self,change_ratio,key_list):
        """"""
        self.color_jitter = transforms.ColorJitter(0.3,0.3,0.3,0.3)
        self._key_list = _key_list_hook(key_list)
        self._image2tensor = ImageToTensor(self._key_list)
        self._change_ratio = change_ratio
        
    def __call__(self,info_dict):
        """"""
        if self._change_ratio < np.random.rand():
            return info_dict
        info_dict = self._image2tensor(info_dict)
        for key in self._key_list:
            info_dict[key] = self.color_jitter(info_dict[key])
        return info_dict
        
            
class ImageToTensor():
    """Convert HWC np.ndarray to CHW tensor."""
    def __init__(self, key_list):
        
        self._key_list = _key_list_hook(key_list)
        
    def __call__(self,info_dict):
        for key in self._key_list: 
            if isinstance(info_dict[key],torch.Tensor) and (info_dict[key].shape[0] == 1 or  info_dict[key].shape[0] == 3):
                # staisfy CHW tensor
                continue
            if isinstance(info_dict[key], np.ndarray):
                image_tensor = torch.from_numpy(info_dict[key])
            elif not isinstance(info_dict[key],torch.Tensor):
                raise NotImplementedError
            if image_tensor.ndim == 2: # HW -> CHW, which C = 1
                image_tensor = image_tensor.unsqueeze(0)
            elif image_tensor.ndim == 3 and image_tensor.shape[-1] == 3:
                image_tensor = image_tensor.permute(2,0,1) # HWC -> CHW 
            else:
                raise RuntimeError(f"Expect HW or WH3 tensor but get {image_tensor.shape}")
            info_dict[key] = image_tensor
        return info_dict
            


class TensorNorm():
    """Convert image to tensor if needed and normalize."""
    def __init__(self, kit_dir_root, use_color:bool, num_channels:int, key_list):
        """init"""
        # norm_info = pickle.load(open(os.path.join(Path(kit_dir_root).parent, "mean_std.pkl"), "rb"))
        # color
        # if use_color:
        #     color_key = "color"
        # else:
        #     color_key = "gray"
        # _c_mean = norm_info[color_key]["mean"]
        # _c_std = norm_info[color_key]["std"]
        # if not use_color and num_channels == 4:
        #     _c_mean = _c_mean* 3
        #     _c_std = _c_std * 3
        # # depth
        # _d_mean = norm_info["depth"]["mean"]
        # _d_std = norm_info["depth"]["std"]

        # init func
        # self._c_norm = transforms.Normalize(mean=_c_mean, std=_c_std) # need CHW
        # self._d_norm = transforms.Normalize(mean=_d_mean, std=_d_std)
        self._key_list = _key_list_hook(key_list)
        self._image2tensor = ImageToTensor(self._key_list)

        
    def __call__(self,info_dict):
        """ndarray -> tensor -> norm"""
        info_dict = self._image2tensor(info_dict)
        for key in self._key_list:
            info_dict[key] = info_dict[key].to(torch.float32)
            # if "c" in  key:
            #     info_dict[key] = self._c_norm(info_dict[key])
            # elif "d" in key:
            #     info_dict[key] = self._d_norm(info_dict[key])
            info_dict[key] /= 255.0
        return info_dict
 