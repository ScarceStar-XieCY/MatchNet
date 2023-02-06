"""Matrix calculate."""
# coding=UTF-8

import numpy as np
import math
import os
import sys
sys.path.append(os.getcwd())
from tools.image_mask.mask_process import mask2coord

def rot_around_point(rot_mtx,value,center_point):
    '''
    批量点的旋转，value的shape是(N,2),其中N表示N个点
    :param rot_mtx: 变换矩阵的内容
    :param value: 要旋转的点
    :param center_point: 旋转的中心点
    :return:
    '''
    if isinstance(center_point,tuple):
        center_point = np.array(center_point)
    trans_mtx_3 = np.eye(3)
    trans_mtx_1 = np.eye(3)
    rot_mtx33 = rot_mtx[:3, :3]
    trans_mtx_3[:2, 2] = center_point
    trans_mtx_1[:2, 2] = -center_point
    trans = trans_mtx_3 @ rot_mtx33 @ trans_mtx_1
    value_ones = np.ones((len(value), 1))
    value = np.hstack((value, value_ones))
    value_after_rot = (trans @ value.T).T
    value_after_rot = value_after_rot[:, :2]
    return value_after_rot

def gen_rot_mtx_anticlockwise(angle,isdegree:bool=False):
    '''
    生成逆时针旋转指定角的旋转矩阵
    :param angle:要逆时针旋转的角
    :param isdegree:旋转角的表示方式,如果是用角度表示,此项为True,若是弧度表示,此项为False
    :return:3*3的旋转矩阵
    '''
    if isdegree:
        angle = math.radians(angle)
    cos_rad = math.cos(angle)
    sin_rad = math.sin(angle)
    trans = np.eye(3)
    trans[:2, :2] = [[cos_rad, -sin_rad], [sin_rad, cos_rad]]
    return trans


def rigid_trans_mask_around_point(mask:np.ndarray, theta:float, init_point, final_point, is_degree:bool = True,clip_outer:bool = True):
    """Rotate theta around init_center, and translate to final center."""
    h,w  = mask.shape[:2]
    mask_coord = mask2coord(mask,need_xy=False)
    rot_mtx = gen_rot_mtx_anticlockwise(theta,isdegree=is_degree)
    obj_after_rot = rot_around_point(rot_mtx,mask_coord,init_point)
    if isinstance(init_point,tuple):
        init_point = np.array(init_point)
    if isinstance(final_point,tuple):
        final_point = np.array(final_point)
    translation = final_point - init_point
    obj_after_rigid = obj_after_rot + translation
    obj_after_rigid = np.ceil(obj_after_rigid).astype("int")
    if clip_outer:
        valid_mask = (obj_after_rigid[:,0] >= 0 ) & (obj_after_rigid[:,1] >= 0) & (obj_after_rigid[:,0] < h) & (obj_after_rigid[:,1] < w)
        obj_after_rigid = obj_after_rigid[valid_mask]
    return obj_after_rigid



def reverse_get_corres(coord_after_rigid,theta, init_point, final_point,is_degree, ):
    """Get reverse point thaT before rigid transform. https://www.zhihu.com/question/430095481"""
    if len(coord_after_rigid) == 0:
        return np.zeros((0,2), dtype = "int")
    rot_mtx = gen_rot_mtx_anticlockwise(-theta,isdegree=is_degree)
    obj_after_rot = rot_around_point(rot_mtx,coord_after_rigid,final_point)
    if isinstance(init_point,tuple):
        init_point = np.array(init_point)
    if isinstance(final_point,tuple):
        final_point = np.array(final_point)
    translation = init_point - final_point
    obj_reverse_coord = obj_after_rot + translation
    obj_reverse_coord = np.ceil(obj_reverse_coord).astype("int")
    return obj_reverse_coord
