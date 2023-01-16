"""Process for 3-channel image."""
# coding=UTF-8
import cv2
import numpy as np
from collect_data.mask_process import *
import os

def get_edge_sobel(image, color_name, channel, k = 3, visual = False):
    color_name_image = convert_image(image, color_name)
    onechannel_image = color_name_image[..., channel]
    onechannel_image = cv2.GaussianBlur(onechannel_image, ksize=(3, 3), sigmaX=0, dst=None, sigmaY=None,
                                        borderType=None)
    image_x = cv2.Sobel(onechannel_image, cv2.CV_64F, 1, 0, ksize=k)  # X方向Sobel
    '''
    参数2 depth：必选参数。表示输出图像的深度，-1表示采用的是与原图像相同的深度。目标图像的深度必须大于等于原图像的深度
    参数3和参数4 dx和dy表示的是求导的阶数，0表示这个方向上没有求导，一般为0、1、2
    参数6 ksize：可选参数。用于设置内核大小，即Sobel算子的矩阵大小，值必须是1、3、5、7，默认为3。
    '''
    absX = cv2.convertScaleAbs(image_x)  # 转回uint8
    image_y = cv2.Sobel(onechannel_image, cv2.CV_64F, 0, 1, ksize=k)  # Y方向Sobel
    absY = cv2.convertScaleAbs(image_y)
    # 进行权重融合
    line_mask = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    if visual:
        cv2.imshow('sobel_edge_{}'.format(color_name),line_mask)
    return line_mask

def get_edge_canny(image, color_name, channel, th1, th2, visual):
    color_name_image = convert_image(image, color_name)
    onechannel_image = color_name_image[..., channel]
    onechannel_image = cv2.GaussianBlur(onechannel_image, ksize=(3,3), sigmaX=0, dst=None, sigmaY=None, borderType=None)
    onechannel_image = cv2.medianBlur(onechannel_image, 3)
    color_name_edges = cv2.Canny(onechannel_image, th1, th2)
    if visual:
        cv2.imshow('edges_mask_{}'.format(color_name, channel),color_name_edges)
        # put_mask_on_img(color_name_edges, image, True, 'edges_on_image_{}_()'.format(color_name, channel))
    return color_name_edges


def adap_get_mask_in_color_space(image, color_name, visual):
    lab_image = convert_image(image, 'lab')
    # cv2.imwrite(os.path.join(cfg.data_root, 'show', 'lab_image.png'), lab_image)
    lab_mask = cv2.adaptiveThreshold(lab_image[..., 1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 2)
    lab_mask = remove_small_area(lab_mask, 1000, False, '')
    # lav_mask_iamge= put_mask_on_img(lab_mask, lab_image, False,'lab_mask_on_image')
    # cv2.imwrite(os.path.join(cfg.data_root, 'show', 'lav_mask_iamge.png'), lav_mask_iamge)
    if color_name == 'hsv':
        color_name_image = convert_image(image, color_name)
        mask = cv2.adaptiveThreshold(color_name_image[...,1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 201, 2)
        lab_mask = get_avaliable_part(mask, lab_mask, False)
        # lav_mask_iamge = put_mask_on_img(lab_mask, color_name_image, False, 'hsv_mask_on_image')
        # cv2.imwrite(os.path.join(cfg.data_root, 'show', 'hsv_mask_iamge.png'), lav_mask_iamge)
    if visual:
        cv2.imshow('{}_mask'.format(color_name), lab_mask)
    return lab_mask.astype('uint8')



def convert_image(image, color_space_name):
    '''根据名字返回不同的颜色空间的图片,不会改变原图
    '''
    if color_space_name == 'bgr':
        return image.copy()
    if color_space_name == 'hsv':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color_space_name == 'xyz':
        return cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    if color_space_name == 'ycrcb':
        return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    if color_space_name == 'hls':
        return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    if color_space_name == 'lab':
        return cv2.cvtColor(image, cv2.COLOR_BGR2Lab) 
    if color_space_name == 'luv':
        return cv2.cvtColor(image, cv2.COLOR_BGR2Luv)

def get_all_edge(image, restrict_mask, visual):
    rgb_edge = get_edge_canny(image, 'bgr', 1, 25, 45, True)
    hsv_edge = get_edge_canny(image, 'hsv', 2, 25, 45, True)
    lab_edge = get_edge_canny(image, 'lab', 0, 25, 45, True)
    all_edge = get_union(hsv_edge, lab_edge, False, 'two_edge')
    cv2.imshow('all_mask', all_edge)
    cv2.waitKey()
    all_edge = get_intersection(all_edge, restrict_mask, False, 'all')
    # all_edge = remove_scattered_pix(all_edge, 5, False)
    all_edge = get_half_mask(all_edge,True, 70)
    # all_edge = close_morph(all_edge, 3, 1)
    if visual:
        cv2.imshow('all_edge', all_edge)
    return all_edge.astype('uint8')


def draw_label_on_image(image, result_dict, visual):
    for geometry, info in result_dict.items():
        if geometry == 'circle':
            cv2.circle(image, info[0], info[2] , (0,255,0), 2)
        else:
            cv2.drawContours(image, [np.array(info[2])], -1, (0,255,0), 2)
        cv2.putText(image, geometry, info[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if visual:
        cv2.imshow('result',image)
    return image
# retval, labels_cv, stats, centroids = cv2.connectedComponentsWithStats(img, ltype=cv2.CV_32S)


def get_all_contour_points(contours):
    all_cnt = contours[0]
    for i in range(1, len(contours)):
        all_cnt = np.concatenate((all_cnt, contours[i]))
    return all_cnt


def adap_mask_one_channel_tool(image, need_save, i, img_dir, visual):
    # hsv的1通道经过adaptive之后可以很好的分割出整个物体，不要inv，但是还需要各种后处理
    # hsv的2通道在canny下可以得到不错的边缘，但是自适应分割时完全看不出

    # lab的2通道在adaptive分割之后可以分割出整个物体，要inv，保留左半边并去除小区域即可
    # lab 0 通道可用于canny得到两个表面的边缘
    
    def get_edge(onechannel_image, th1, th2):
        onechannel_image = cv2.GaussianBlur(onechannel_image, ksize=(3,3), sigmaX=0, dst=None, sigmaY=None, borderType=None)
        onechannel_image = cv2.medianBlur(onechannel_image, 3)
        color_name_edges = cv2.Canny(onechannel_image, th1, th2)
        return color_name_edges

    # #rgb三个通道分别处理,以及变成灰度
    # rgb0_mask = cv2.adaptiveThreshold(image[..., 0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)
    # rgb1_mask = cv2.adaptiveThreshold(image[..., 1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)
    # rgb2_mask = cv2.adaptiveThreshold(image[..., 2], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)

    #HSV三个
    hsv_image = convert_image(image, 'hsv')
    # hsv0_mask = cv2.adaptiveThreshold(hsv_image[..., 0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 2)
    # hsv1_mask = cv2.adaptiveThreshold(hsv_image[..., 1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 2)
    # hsv2_mask = cv2.adaptiveThreshold(hsv_image[..., 2], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)

    #Lab三个
    lab_image = convert_image(image, 'lab')
    # lab0_mask = cv2.adaptiveThreshold(lab_image[..., 0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)
    # lab1_mask = cv2.adaptiveThreshold(lab_image[..., 1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 2)
    # lab2_mask = cv2.adaptiveThreshold(lab_image[..., 2], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201, 2)

    # 对边缘的检测
    rgb0_edge = get_edge(image[..., 0], 5, 10)
    rgb1_edge = get_edge(image[..., 1], 5, 10)
    rgb2_edge = get_edge(image[..., 2], 5, 10)
    gray_edge = get_edge(gray, 5, 10)

    hsv0_edge = get_edge(hsv_image[..., 0], 5, 10)
    hsv1_edge = get_edge(hsv_image[..., 1], 5, 10)
    hsv2_edge = get_edge(hsv_image[..., 2], 5, 10)

    lab0_edge = get_edge(lab_image[..., 0], 5, 10)
    lab1_edge = get_edge(lab_image[..., 1], 5, 10)
    lab2_edge = get_edge(lab_image[..., 2], 5, 10)

    if need_save:
        # cv2.imwrite(os.path.join(img_dir, 'color{}_rgb0_mask.png'.format(i)), rgb0_mask)
        # cv2.imwrite(os.path.join(img_dir, 'color{}_rgb1_mask.png'.format(i)), rgb1_mask)
        # cv2.imwrite(os.path.join(img_dir, 'color{}_rgb2_mask.png'.format(i)), rgb2_mask)
        # cv2.imwrite(os.path.join(img_dir, 'color{}_rgb3_mask.png'.format(i)), gray_mask)
        #
        # cv2.imwrite(os.path.join(img_dir, 'color{}_hsv0_mask.png'.format(i)), hsv0_mask)
        # cv2.imwrite(os.path.join(img_dir, 'color{}_hsv1_mask.png'.format(i)), hsv1_mask)
        # cv2.imwrite(os.path.join(img_dir, 'color{}_hsv2_mask.png'.format(i)), hsv2_mask)
        #
        # cv2.imwrite(os.path.join(img_dir, 'color{}_lab0_mask.png'.format(i)), lab0_mask)
        # cv2.imwrite(os.path.join(img_dir, 'color{}_lab1_mask.png'.format(i)), lab1_mask)
        # cv2.imwrite(os.path.join(img_dir, 'color{}_lab2_mask.png'.format(i)), lab2_mask)


        cv2.imwrite(os.path.join(img_dir, 'color{}_rgb0_edge.png'.format(i)), 255- rgb0_edge)
        cv2.imwrite(os.path.join(img_dir, 'color{}_rgb1_edge.png'.format(i)), 255- rgb1_edge)
        cv2.imwrite(os.path.join(img_dir, 'color{}_rgb2_edge.png'.format(i)), 255- rgb2_edge)
        cv2.imwrite(os.path.join(img_dir, 'color{}_rgb3_edge.png'.format(i)), 255- gray_edge)

        cv2.imwrite(os.path.join(img_dir, 'color{}_hsv0_edge.png'.format(i)), 255- hsv0_edge)
        cv2.imwrite(os.path.join(img_dir, 'color{}_hsv1_edge.png'.format(i)), 255- hsv1_edge)
        cv2.imwrite(os.path.join(img_dir, 'color{}_hsv2_edge.png'.format(i)), 255- hsv2_edge)

        cv2.imwrite(os.path.join(img_dir, 'color{}_lab0_edge.png'.format(i)), 255- lab0_edge)
        cv2.imwrite(os.path.join(img_dir, 'color{}_lab1_edge.png'.format(i)), 255- lab1_edge)
        cv2.imwrite(os.path.join(img_dir, 'color{}_lab2_edge.png'.format(i)), 255- lab2_edge)

    # if visual:
    #     cv2.imshow('rgb0_mask', rgb0_mask)
    #     cv2.imshow('rgb1_mask', rgb1_mask)
    #     cv2.imshow('rgb2_mask', rgb2_mask)
    #     cv2.imshow('gray_mask', gray_mask)
    #
    #     cv2.imshow('hsv0_mask', hsv0_mask)
    #     cv2.imshow('hsv1_mask', hsv1_mask)
    #     cv2.imshow('hsv2_mask', hsv2_mask)
    #
    #     cv2.imshow('lab0_mask', lab0_mask)
    #     cv2.imshow('lab1_mask', lab1_mask)
    #     cv2.imshow('lab2_mask', lab2_mask)
    #
    #     cv2.waitKey()


def adap_mask_by_saturability(image, visual):
    #实验发现hsv1最能有效地分割出物体，hsv通道对应颜色的饱和度
    hsv_image = convert_image(image, 'hsv')
    hsv1_mask = cv2.adaptiveThreshold(hsv_image[..., 1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 201, 2)
    # 去噪,会损失顶点上的尖端
    hsv1_mask = remove_surrounding_white(hsv1_mask, False) #去掉左右与边缘连接的白色部分，这要求物体一定不能接触画面边缘，不然会被当做噪声去掉
    hsv1_mask = erode(hsv1_mask, 3, 2)
    hsv1_mask = remove_big_area(hsv1_mask, 10000, False, 'hsv1')
    hsv1_mask = get_half_mask(hsv1_mask, True, 100)
    hsv1_mask = remove_small_area(hsv1_mask, 2000, False, 'hsv1')
    hsv1_mask = dilate(hsv1_mask, 3, 2)
    if visual:
        cv2.imshow( "hsv1_mask after remove noise", hsv1_mask)
    return hsv1_mask

def get_color_acc2_coord(image, coord):
    # 获得xy坐标选中的部分的图形颜色（假设每个mask中只有一种颜色）
    one_pixel_color_bgr = image[coord[1], coord[0]]
    one_pixel_color_bgr =one_pixel_color_bgr[np.newaxis, np.newaxis,:]
    one_pixel_color_hsv = cv2.cvtColor(one_pixel_color_bgr, cv2.COLOR_BGR2HSV)
    print('当前mask的像素点，其bgr颜色为 {}， hsv颜色为{}'.format(one_pixel_color_bgr, one_pixel_color_hsv))


def grabcut_get_mask(image, fg_mask, color_space_name, visual):
    # 用grabcut分割图像中的物体，其中fg_mask是输入给grabcut的、确定为前景的mask，
    # 也就是说，这个mask如果覆盖到背景，就会造成分割失败，但同样，mask圈出来的内容不能太少，不然分割出来的区域不完整
    bgdModel = np.zeros((1, 65), np.float64)  # 背景模型
    fgdModel = np.zeros((1, 65), np.float64)  # 前景模型
    mask = np.full(image.shape[:2], 2, dtype=np.uint8) # 设定全部画面都为“可能的背景”s
    # fg_mask对应的位置设置为3,对应“可能的前景”
    mask = np.where((fg_mask == 255), 3, mask)
    h, w, c = image.shape
    color_space_image = convert_image(image, color_space_name)
    # 分别得到xmin，xmax，ymin，ymax 作为ROI，grabcut只处理ROI以降低运算量
    rmin, rmax, cmin, cmax = mask2bbox(fg_mask)
    r_tole = 20
    c_tole = 20
    rmin -= r_tole if rmin > r_tole else 0
    rmax += r_tole if rmax < h - r_tole else h
    cmin -= c_tole  if cmin > c_tole else 0
    cmax += c_tole if cmax < w - c_tole else w
    cv2.grabCut(color_space_image[rmin: rmax, cmin:cmax,:], mask[rmin: rmax, cmin:cmax], [0,0,0,0], bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)#使用mask初始化
    # cv2.grabCut(color_space_image, mask, [0,0,0,0], bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)#使用mask初始化
    grabcut_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8') #把2对应的“可能的背景”和0对应的“背景”部分设为0,其他部分为255
    # img_show = image * mask2[:, :, np.newaxis]
    # # 显示图片分割后结果--显示原图
    # cv2.imshow('grabcut', img_show)
    if visual:
        cv2.imshow('grabcut_mask',grabcut_mask)
    return grabcut_mask
