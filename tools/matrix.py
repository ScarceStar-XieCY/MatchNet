"""Matrix calculate."""
# coding=UTF-8

import numpy as np
import math

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

def gen_rot_mtx_anticlockwise(angle,isdegree):
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
