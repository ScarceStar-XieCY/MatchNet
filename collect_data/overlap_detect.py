"""Detect overlap."""
# coding=UTF-8
import cv2
import os
import numpy as np
import sys
sys.path.append(os.getcwd())
from tools.image_mask.mask_process import open_morph,get_half_centroid_mask
from tools.matrix import gen_rot_mtx_anticlockwise, rot_around_point
from tools.image_mask.mask_process import mask2coord, coord2mask, get_mask_center
import logging

logger = logging.getLogger(__file__)


def _diff_mask(in_kit_img,out_kit_img,visual):
    diff_img = cv2.subtract(out_kit_img,in_kit_img)
    diff_gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("diff", diff_gray*10)
    dthresh, diff_mask = cv2.threshold(diff_gray, 0,255,cv2.THRESH_BINARY,cv2.THRESH_OTSU)
    # cv2.imshow("diffthre", diff_mask)
    diff_mask = open_morph(diff_mask,3,2)
    # cv2.imshow("diff_open", diff_mask)
    diff_mask = get_half_centroid_mask(diff_mask,False,0,visual)
    return diff_mask


def rigid_trans_mask_around_point(mask:np.ndarray, theta:float, init_point, final_point, clip_outer:bool = True,need_mask:bool=False, visual:bool=False):
    """Rotate theta around init_center, and translate to final center."""
    h,w  = mask.shape[:2]
    mask_coord = mask2coord(mask,need_xy=False)
    rot_mtx = gen_rot_mtx_anticlockwise(theta,isdegree=True)
    obj_after_rot = rot_around_point(rot_mtx,mask_coord,init_point)
    if isinstance(init_point,tuple):
        init_point = np.array(init_point)
    if isinstance(final_point,tuple):
        final_point = np.array(final_point)
    translation = final_point - init_point
    obj_after_rigid = obj_after_rot + translation
    obj_after_rigid = np.around(obj_after_rigid).astype("int")
    if clip_outer:
        valid_mask = (obj_after_rigid[:,0] >= 0 ) & (obj_after_rigid[:,1] >= 0) & (obj_after_rigid[:,0] < h) & (obj_after_rigid[:,1] < w)
        obj_after_rigid = obj_after_rigid[valid_mask]
    if need_mask:
        mask_after_rigid = coord2mask(obj_after_rigid,h,w,visual=visual)
        return mask_after_rigid
    return obj_after_rigid


class OverlapDetector:
    def __init__(self,shape):
        self._obj_mask = np.zeros(shape,dtype = np.int)
    
    def detect_overlap(self, mask, theta, init_point, final_point):
        rigid_mask = rigid_trans_mask_around_point(mask, theta, init_point,final_point, need_mask=True,visual=True)
        intersection_mask = (self._obj_mask & rigid_mask)
        is_overlap = intersection_mask.any()
        if is_overlap:
            return True
        else:
            self._obj_mask = self._obj_mask | rigid_mask
            return False
    
    def reset(self):
        self._obj_mask = np.zeros_like(self._obj_mask,dtype = np.int)


if __name__ == "__main__":
    data_root = '20221029test'
    data_type = "train"
    color_image_in = cv2.imread(os.path.join(data_root,data_type, f"color1.png"))
    color_image_out = cv2.imread(os.path.join(data_root,data_type, f"color2.png"))
    diff_mask = _diff_mask(color_image_in,color_image_out, visual=True)
    center = get_mask_center(diff_mask)
    final_point = (200, 100) # uv 
    detector = OverlapDetector(diff_mask.shape) 
    obj_num = 6
    for _ in range(obj_num):
        overlap_time = 0
        # new_mask = rigid_trans_mask_around_point(diff_mask,30,center,final_point,need_mask=True,visual=True)
        if detector.detect_overlap(diff_mask,30,center,final_point,need_mask=True,visual=True):
            if overlap_time < 10:
                overlap_time += 1
                continue
            else:
                logger.warning("Can't find a place.")
                break
        else:
            # do move
            pass
        cv2.waitKey()

    
