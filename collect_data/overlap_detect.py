"""Detect overlap."""
# coding=UTF-8
import cv2
import os
import numpy as np
import sys
sys.path.append(os.getcwd())
from tools.image_mask.mask_process import open_morph,get_half_centroid_mask,remove_inner_black,dilate
from tools.image_mask.mask_process import mask2coord, coord2mask, get_mask_center
from tools.matrix import rigid_trans_mask_around_point
from collect_data.suction_mask import get_obj_coord_with_mask_2d
import logging

logger = logging.getLogger(__file__)
DILATE_ITER = 3 

def _diff_mask(in_kit_img,out_kit_img,visual):
    """Get diff for ovrelap detect"""
    diff_img = cv2.subtract(out_kit_img,in_kit_img)
    diff_gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("diff", diff_gray*10)
    dthresh, diff_mask = cv2.threshold(diff_gray, 0,255,cv2.THRESH_BINARY,cv2.THRESH_OTSU)
    # cv2.imshow("diffthre", diff_mask)
    diff_mask = open_morph(diff_mask,3,2)
    # cv2.imshow("diff_open", diff_mask)
    diff_mask = get_half_centroid_mask(diff_mask,False,0,visual)
    return diff_mask

def random_coord():
    margin = 20
    u  = np.random.uniform(margin,240 - margin)
    v = np.random.uniform(margin,424//2 - margin)
    return u,v

def find_coord_to_place(robot,mask, coord):
    count = 0
    while True:
        # random_place_3d = gen_coords(method = "random",epoch = 1) #TODO:robot rndom
        random_place_2d = random_coord()
        random_theta = np.random.uniform(0,90)
        # random_place_2d = robot.b2c(random_place_3d)
        logger.warning("suc coord = %s, place_coord = %s, rot:%s",coord, random_place_2d, random_theta)
        if  not detector.detect_overlap(mask, random_theta, coord, random_place_2d):
            logger.warning("find random place")
            return None,random_place_2d,random_theta
        count += 1
        if count >= 10:
            logger.warning("can't find place after %s's try.",count)

    

class OverlapDetector:
    def __init__(self,shape):
        self._h = shape[0]
        self._w = shape[1]
        self._obj_mask = np.zeros(shape,dtype = np.uint8)
        self._cur_obj_mask = None
    
    def detect_overlap(self, mask, theta, init_point, final_point):
        if self._cur_obj_mask is None:
            mask = dilate(mask, 3, DILATE_ITER) 
            # pass
        else:
            mask = self._cur_obj_mask

        mask_coord = mask2coord(mask,need_xy=False)
        rigid_coord = rigid_trans_mask_around_point(mask, theta, init_point,final_point)
        rigid_mask = coord2mask(rigid_coord,self._h, self._w, visual=True)
        if len(mask_coord) != len(rigid_coord):
            # filetr part of obj outer img
            return True
        intersection_mask = (self._obj_mask & rigid_mask)
        is_overlap = intersection_mask.any()
        if is_overlap:
            self._cur_obj_mask = mask
            return True
        else:
            self._obj_mask = self._obj_mask | rigid_mask
            self._cur_obj_mask = None

            return False
    
    def reset(self):
        logger.warning("reset")
        self._obj_mask = np.zeros_like(self._obj_mask,dtype = np.uint8)
        self._cur_obj_mask = None


if __name__ == "__main__":

    data_root = os.path.join('20221029test')
    data_type = "train"
    kit_name = "bear"
    file_name_list = os.listdir(os.path.join(data_root,data_type))
    step_num = len(file_name_list) // 3
    compare_depth  =  cv2.imread(os.path.join("20221029test_compare",data_type,f"depth0.png"), cv2.IMREAD_GRAYSCALE)
    obj_num= 5 # kit obj num +2
    kit_count = 0
    detector = OverlapDetector(compare_depth.shape) 
    while True:
        # for one kit
        # suction stage
        # obj_num= obj_num_init - (i % obj_num_init)
        obj_coord = []
        # get image 
        # i = (obj_num_init - obj_num) + obj_num_init * kit_count
        i = 26
        print(i)
        depth_image = cv2.imread(os.path.join(data_root,data_type,  f"depth{i}.png"), cv2.IMREAD_GRAYSCALE)
        color_image = cv2.imread(os.path.join(data_root,data_type,  f"color{i}.png"))
        obj_coord,mask_list = get_obj_coord_with_mask_2d(compare_depth,depth_image, color_image, obj_num+2,obj_num)
        assert len(obj_coord) == obj_num
        for (uv_coord,radius),mask in zip(obj_coord,mask_list):
            find_coord_to_place(None,mask, uv_coord)
        detector.reset()
            # cv2.waitKey()

    
