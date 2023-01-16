"""Tools for interctive operation."""
# coding=UTF-8

import numpy as np
import cv2
from tools.image_mask.mask_process import get_centroid

def get_range_from_list(range_list):
    range_array = np.array(range_list, dtype=int)
    assert range_array.shape[1] == 3
    upper_value = []
    lower_value = []
    for i in range(3):
        upper_value.append(np.max(range_array[:, i]))
        lower_value.append(np.min(range_array[:, i]))
    upper_value = np.array(upper_value, dtype=int)
    lower_value = np.array(lower_value, dtype=int)
    return [lower_value, upper_value]

def add_point_color(point, image, color_range_dict):
    '''
    point是(y,x)
    :param point:
    :param image:
    :param color_name:
    :return:
    '''
    point = image[point[0],point[1]][np.newaxis,np.newaxis,:] #
    if color_range_dict.has_key('hsv'):
        point = cv2.cvtColor(point, cv2.COLOR_BGR2HSV)
        color_range_dict['hsv'].append(point.ravel())
    if color_range_dict.has_key('xyz'):
        point = cv2.cvtColor(point, cv2.COLOR_BGR2XYZ)
        color_range_dict['xyz'].append(point.ravel())
    if color_range_dict.has_key('ycrcb'):
        point = cv2.cvtColor(point, cv2.COLOR_BGR2YCrCb)
        color_range_dict['ycrcb'].append(point.ravel())
    if color_range_dict.has_key('hls'):
        point = cv2.cvtColor(point, cv2.COLOR_BGR2HLS)
        color_range_dict['hls'].append(point.ravel())
    if color_range_dict.has_key('lab'):
        point = cv2.cvtColor(point, cv2.COLOR_BGR2Lab)
        color_range_dict['lab'].append(point.ravel())
    if color_range_dict.has_key('luv'):
        point = cv2.cvtColor(point, cv2.COLOR_BGR2Luv)
        color_range_dict['luv'].append(point.ravel())
    if color_range_dict.has_key('bgr'):
        color_range_dict['bgr'].append(point.ravel())



def get_pointed_range(event, x, y, flags, param):
    '''
    鼠标事件响应程序：根据鼠标按键更新param[1]，然后把param[0]中像素值在range = [min(param[1]),max(param[1])]的点显示为白色
    鼠标按键更新方式：①左键选择的像素值被加入param[1]
    ②右键选择的像素值从param[1]中被删除，而且为range边界值实现近似删除，如果该像素值与range两端之一的差值在2之间，则会删除端点值
    ③每按下一次中键，去掉当前param[1]中最新加入的值
    :param event: 鼠标事件
    :param x: 鼠标选择的点的x坐标
    :param y: 鼠标选择的点的y坐标
    :param flags:
    :param param: param =[img,pixel]，分别是图像和列表
    :return: 无返回值，根据鼠标事件响应并更新窗口’update_mask‘中的图像后就结束运行
    '''
    # hsv_img = cv2.cvtColor(param[0], cv2.COLOR_BGR2HSV)
    hsv_img = convert_image(param[0],param[2])
    if event == cv2.EVENT_LBUTTONDOWN:
        print('本次按下左键')
        # print('param.type',type(param[0]))
        add_val = hsv_img[y, x]
        # print('add_val', add_val)
        # if add_val not in param[1]:
        param[1].append(add_val)
        # print('目前像素值列表为：', param[1])
        print('成功添加像素值 {}'.format(add_val))
        # else:
        #     print('像素值{}已在，不再添加'.format(add_val))
        range = get_range_from_list(param[1])
        print('目前像素范围为', range, '\n')
        img = mask_range(param[0].copy(), range, param[2])
        cv2.imshow('update_mask', img)
    if event == cv2.EVENT_RBUTTONDOWN:#TODO：是否要改成建立一个排除列表
        print('本次按下右键')
        remove_val = np.array(hsv_img[y, x])
        if remove_val in param[1]:
            param[1].remove(remove_val)
            print('成功删除像素值{}'.format(remove_val))
        else:
            min_val = min(param[1])
            max_val = max(param[1])
            print('像素值{}不在列表内，无法删除'.format(remove_val))
            distance = remove_val - min_val if remove_val > min_val else min_val - remove_val
            if distance <= 2:
                param[1].remove(min_val)
                print('与最小值距离近，删去最小值')
            distance = remove_val - max_val if remove_val > max_val else max_val - remove_val
            if distance <= 2 and len(param[1]) != 0:
                param[1].remove(max_val)
                print('与最大值距离近，删去最大值')
        if len(param[1]) == 0:
            range = []
        else:
            range = get_range_from_list(param[1])
        # print('目前像素值列表为：', param[1])
        print('目前像素范围为', range[0], '\n', range[1])
        img, = mask_range(param[0].copy(), range,param[2])
        cv2.imshow('update_mask', img)

    if event == cv2.EVENT_MBUTTONDOWN:
        if len(param[1]) == 0:
            range = []
        else:
            param[1].pop()
        range = get_range_from_list(param[1])
        # print('目前像素值列表为：', param[1])
        print('目前像素范围为', range[0], '\n', range[1])
        img = mask_range(param[0].copy(), range,param[2])
        cv2.imshow('update_mask', img)


def update_mask(img, pixel,color_space_name):
    '''
    获得mask所需的像素值范围
    :param img:需要选择范围的图像
    :param pixel:选择的像素值列表
    :return:range:选择的像素值范围
    '''
    cv2.namedWindow('update_mask')
    cv2.imshow('update_mask', img)
    param_event = [img, pixel,color_space_name]
    cv2.setMouseCallback('update_mask', get_pointed_range, param_event)
    key = cv2.waitKey()
    if key == ord('q'):
        cv2.destroyWindow('update_mask')
    range = get_range_from_list(pixel)
    return range


def mask_range(img, range,color_space_name):
    '''
    对img中的像素值在range中的点进行mask，mask经过一定的开运算以去除噪声点
    :param img: 需要被mask的图像
    :param range: mask的像素范围
    :param color_space_name:分割方式的颜色空间
    :return: img：已经被mask的图像；
    '''
    if range != []:
        space_name_img = convert_image(img, color_space_name)
        mask_layer = cv2.inRange(space_name_img, range[0], range[1])
        img_mask = put_mask_on_img(mask_layer, img, False, '')
    return img_mask


def visual_shape(image, cnt, shape):
    '''
    在图像上画出边框和标注结果,会改变传入的参数image
    :param image:
    :param cnt:
    :param shape:
    :return:
    '''
    cx, cy = get_centroid(cnt)
    cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
    cv2.putText(image, shape, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow('shape',image)


# def update_tool(image):
# pixel_range = update_mask(image, pixel_range)
# print('======new_range====', pixel_range)