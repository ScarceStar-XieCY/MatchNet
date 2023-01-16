"""Various mask processing."""
# coding=UTF-8
import cv2
import numpy as np
from skimage.measure import label
import logging

logger = logging.getLogger(__file__)



def remove_small_area(mask, area_th,visual, message):
    '''
        去除小面积连通域
        :param mask: 待处理的mask
        :param area_th: 小于area_th的面积会被消除
        :return: 去除了小面积连通域的mask
    '''
    contours = get_exter_contours(mask, 'none')
    # contours = get_tree_contours(mask, 'none',-2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < area_th:
            cv2.drawContours(mask, [cnt], 0, 0, -1)
    if visual:
        cv2.imshow('after remove small area_{}'.format(message), mask)
    return mask


def remove_big_area(mask, area_th,visual, message):
    '''
        去除小面积连通域
        :param mask: 待处理的mask
        :param area_th: 小于area_th的面积会被消除
        :return: 去除了小面积连通域的mask
    '''
    contours = get_exter_contours(mask, 'none')
    # contours = get_tree_contours(mask, 'none',-2)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_th:
            cv2.drawContours(mask, [cnt], 0, 0, -1)
    if visual:
        cv2.imshow('after remove big area_{}'.format(message), mask)
    return mask


def remove_inner_white(mask, visual, message):
    contours = get_tree_contours(mask, 'none', -2)
    for cnt in contours:
        cv2.drawContours(mask, [cnt], 0, 0, -1)
    if visual:
        cv2.imshow('after remove inner white_{}'.format(message), mask)
    return mask

def remove_slim(mask, ratio = 10):
    '''
    去掉细长的非0部分,原理:越细长,连通域的面积/周长比就越小
    :param mask:待处理的mask
    :param ratio: ratio=面积/周长
    :return:处理后的mask
    '''
    contours = get_exter_contours(mask, 'none')
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        # print(area)
        # print(perimeter)
        if area < perimeter * ratio:
            cv2.drawContours(mask, [cnt], 0, 0, -1)
    return mask

def get_exter_contours(mask, method = 'none'):
    if method == 'simple' or method == 'SIMPLE':
        contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:# if method == 'none' or method == 'NONE'
        contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def largest_cc(mask,bol2img):
    '''
    选出除0像素之外,最大的连通域
    :param mask:一张图
    :param bol2mask
    :return: bool类型的矩阵,true部分对应的就是最大连通域
    '''
    labels = label(mask)
    largest = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    if bol2img:
        largest = np.array(largest).astype('uint8')*255
    return largest


def remove_scattered_pix(mask,th,visual):
    #去除只有th个像素的连通域而不影响其他内容
    labels = label(mask,connectivity=1)
    remove_index = np.where(np.bincount(labels.flat)[1:] <= th)
    for item in remove_index[0]:
        mask = np.where(labels == item +1, 0, mask)
    if visual:
        cv2.imshow('after remove_single_pix', mask)
    return mask


def mask2bbox(mask):
    #寻找二值化图像的mask=1处的方框
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def remove_surrounding_white(mask,visual):
    '''
    在mask中,去掉贴着图像边缘的白色部分（通常是背景）
    :param mask:处理前的mask
    :param visual: 是否可视化
    :return: mask:处理后的mask
    '''
    h,w = mask.shape
    labels = label(mask)
    if visual:
        cv2.imshow('labels going to remove_surrounding_white',(labels*40).astype('uint8'))
    num = np.max(labels)
    if num > 1:#如果只有一个连通域,不需要处理
        for i in range(num):
            domain = np.where(labels==i+1,1,0)
            if visual:
                cv2.imshow('domain in remove_surrounding_white',(domain.astype('uint8'))*255)
            rmin,rmax,cmin,cmax = mask2bbox(domain)
            if rmin ==0 or rmax == h-1 or cmin == 0 or cmax == w-1:
                labels = np.where(labels == i+1 , 0 , labels)
        mask = np.where(labels != 0,mask,0)
        if visual:
            cv2.imshow('mask in remove_surrounding_white',mask)
    return mask


def remove_inner_black(mask,visual):
    '''
    在mask中去掉白色部分中间的黑色
    :param mask:
    :param visual: 可视化
    :return: mask:处理后的mask
    '''
    h, w = mask.shape
    mask = 255 - mask
    labels = label(mask)
    if visual:
        cv2.imshow('labels going to remove_inner_black', (labels * 40).astype('uint8'))
    num = np.max(labels)
    for i in range(num):
        domain = np.where(labels == i + 1, 1, 0)
        if visual:
            cv2.imshow('domain in remove_inner_black', (domain.astype('uint8')) * 255)
        rmin, rmax, cmin, cmax = mask2bbox(domain)
        if not (rmin == 0 or rmax == h - 1 or cmin == 0 or cmax == w - 1):
            labels = np.where(labels == i + 1, 0, labels)
    mask = np.where(labels != 0, mask, 0)
    mask = 255 - mask
    if visual:
        cv2.imshow('mask in remove_inner_black', mask)
    return mask


def erode(mask,kernel_size,iterations):
    #腐蚀运算
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size,kernel_size))
    erode_mask = cv2.erode(mask, kernel, iterations=iterations)
    return erode_mask


def dilate(mask,kernel_size,iterations):
    #膨胀运算
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size,kernel_size))
    dilate_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return dilate_mask


def open_morph(mask,kernel_size,iterations):
    #开运算:先腐蚀后膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    open_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return open_mask


def close_morph(mask,kernel_size,iterations):
    #闭运算:先膨胀后腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    close_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return close_mask

def black_hat(mask,kernel_size,iterations):
    # 底帽运算 取出原图中较黑的部分
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    blackhat_mask = cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)
    return blackhat_mask

def get_half_centroid_mask(mask:np.ndarray, left_half:bool, tole:int, visual:bool = False):
    '''
    根绝left_half的真假情况,去掉中心点在右半边或左半边的连通域,
    :param mask:二值化mask
    :param left_half:是否要保留左半边的连通域
    :param tole:对额外偏移的容忍
    :return:
    '''
    if (mask == 0).all():
        return mask
    new_mask = mask.copy()
    w = new_mask.shape[1] #图片的宽
    w_half = int(w // 2)
    contours = get_exter_contours(new_mask, 'none')
    for cnt in contours:
        cx,cy = get_centroid(cnt)
        if (left_half and cx > w_half + tole) or (not left_half and cx < w_half - tole):
            cv2.drawContours(new_mask, [cnt], 0, 0, -1)
    new_mask = remove_small_area(new_mask, 50, False,"")
    if visual:
        cv2.imshow("half after remove small", np.uint8(new_mask))
    return np.uint8(new_mask)

def get_half_mask(mask,left_half, tole):
    h = mask.shape[0]
    w = mask.shape[1]
    w_half = int(w // 2)
    if left_half:
        mask = np.hstack((mask[:, :w_half+tole], np.zeros((h, w_half-tole))))
    else:
        mask = np.hstack((np.zeros((h, w_half - tole)), mask[:, w_half - tole:]))
    return mask


def mask2coord(mask,need_xy:bool):
    coord = np.column_stack(np.where(mask))
    if need_xy:
        coord = coord[:,::-1]
    return coord


def coord2mask(coord,h,w,visual):
    mask_layer = np.zeros((h,w))
    mask_layer[coord[:, 0],coord[:, 1]] = 255
    if visual:
        cv2.imshow('mask from coord', mask_layer)
    return mask_layer

def get_mask_center(mask):
    """Get center of one mask,(u,v). Mask shoud have only one connected-domain."""
    contours = get_exter_contours(mask, 'simple')
    if len(contours) > 0:
        logger.warning("more than one contous exsist.")
    cnt = contours[0]
    mask_center = get_centroid(cnt)
    mask_center = mask_center[::-1]
    return mask_center


def is_grayscale(img):
    #判断是否为灰度图
    if img.ndim == 2 or img.shape[2] == 1:
        return True
    else:
        return False


def get_centroid(cnt):
    '''
    获得某个连通域的质心x,y,若有问题返回-1,-1,xy坐标系是opencv坐标系,x水平向右,y竖直向下
    :param cnt:某个连通域的轮廓
    :return:
    '''
    #得到质心
    M = cv2.moments(cnt) # 获取中心点
    if M["m00"] == 0:
        return -1,-1
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx,cy


def get_exter_contours(mask, method = 'none'):
    if method == 'simple' or method == 'SIMPLE':
        contours, hierarch = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:# if method == 'none' or method == 'NONE'
        contours, hierarch = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def get_all_contours(mask, method = 'none'):
    if method == 'simple' or method == 'SIMPLE':
        contours, hierarch = cv2.findContours(mask.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:# if method == 'none' or method == 'NONE'
        contours, hierarch = cv2.findContours(mask.astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours


def get_ccomp_contours(mask, method = 'none', need_parent = True):
    if method == 'simple' or method == 'SIMPLE':
        contours, hierarch = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    else:# if method == 'none' or method == 'NONE'
        contours, hierarch = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if need_parent:
        # 要找到第一个[3]==-1（没有父亲轮廓 = 自己就是父级）的索引,然后根据[0]得到所有相同层级的轮廓
        index = np.array(np.where(hierarch[0,:,3] == -1)[0])
        contours = np.array(contours,dtype=object)[index]
    return contours


def get_tree_contours(mask, method = 'none', need_hierach_level = -1):
    if method == 'simple' or method == 'SIMPLE':
        contours, hierarch = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:# if method == 'none' or method == 'NONE'
        contours, hierarch = cv2.findContours(mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if need_hierach_level != -1:
        # 要找到第一个[3]==n（没有父亲轮廓 = 自己就是父级）的索引,然后根据[0]得到所有相同层级的轮廓
        if need_hierach_level == 0: #要得到最外层的轮廓
            index = np.array(np.where(hierarch[0, :, 3] == -1)[0])
            contours = np.array(contours)[index]
        elif need_hierach_level == -2: #要得到最内层,没有孩子,同时有父亲的轮廓
            condition = np.logical_and(hierarch[0, :, 2] == -1, hierarch[0, :, 3] != -1)
            index = np.array(np.where(condition)).squeeze()
            contours = np.array(contours)[index]
        else: #要得到内层的某层轮廓,一定都是有父轮廓的
            for i in range(len(contours)):
                while hierarch[0, i, 3] == -1:
                    continue
    return contours


def mean_filter(img, k = 10):
    kernel = np.ones((k, k)) / 100
    img = cv2.filter2D(img, -1, kernel)
    return img

def get_intersection(mask1, mask2,visual, message):
    mask = mask1 & mask2
    if visual:
        cv2.imshow('{}_intersection'.format(message), mask)
    return mask

def get_union(mask1, mask2,visual, message):
    mask = mask1 | mask2
    if visual:
        cv2.imshow('{}_union'.format(message), mask)
    return mask


# def detect_corner(image_1channel,visual,channel_name):
#     '''
#     从单通道图像中检测角点
#     :param image_1channel:
#     :param visual:
#     :param channel_name:
#     :return:
#     '''
#     corners = cv2.goodFeaturesToTrack(image_1channel, 25, 0.01, 10).astype('int64')
#     for i in corners:
#         x, y = i.ravel()
#         cv2.circle(image_1channel, (x, y), 3, (0,255,0), -1)
#     if visual:
#         cv2.imshow('detect_corner_in_{}_channel'.format(channel_name),image_1channel)


def get_avaliable_part(mask, ref_mask, visual):
    #mask中只留下与ref_mask几乎重合的完整连通域
    contours = get_exter_contours(mask,'simple')
    for cnt in contours:
        cx, cy = get_centroid(cnt)
        if ref_mask[cy, cx] == 0:
            cv2.drawContours(mask, [cnt], 0, 0, -1)
    if visual:
        cv2.imshow('inter',mask)
    return mask


def draw_a_line(line, mask, color, width):
    #会改变原图像
    h, w = mask.shape
    for rho, theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + w * (-b))
        y1 = int(y0 + w * a)
        x2 = int(x0 - w * (-b))
        y2 = int(y0 - w * a)
        mask = cv2.line(mask, (x1, y1), (x2, y2), color, width)
    return mask


def draw_lines(lines, mask, color, width, visual, message):
    #在mask上画多个直线,会改变mask
    for line in lines:
        draw_a_line(line, mask, color, width)
    if visual:
        cv2.imshow('lines_{}'.format(message), mask)
    return mask


def get_each_mask(mask):
    labels = label(mask,connectivity=1)
    num = np.max(labels) #背景+白色连通域个数
    each_mask_list = []
    for i in range(1, num+1):
        one_mask = np.where(labels == i, mask, 0)
        each_mask_list.append(one_mask)
    return each_mask_list


def get_max_inner_circle(mask, visual):
    # assume mask can cover total obj, not only contour
    # get max inner circle of external contour
    
    mask = mask.astype('uint8')
    mask_coord = mask2coord(mask, True)
    valid_point_count = 100
    if len(mask_coord) < valid_point_count:
        logger.warning(f"Less than {valid_point_count} points, skip.")
        return None,None
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    # Calcul ate the distances to the external contour  
    raw_dist = np.zeros(mask.shape, dtype=np.float32)

    
    if len(contours) == 1:
        contour = contours[0]
    elif len(contours) > 1:
        # choose the longest counter
        logger.warning("More than 1 contours ,choose th longest one.")
        max_len = 0
        max_idx = -1
        for idx,contour in enumerate(contours):
            if len(contour) > max_len:
                max_idx = idx
                max_len = len(contour)
        contour = contours[max_idx]
    # no hole in mask:
    for i in range(len(mask_coord)):
        x = mask_coord[i][0]
        y = mask_coord[i][1]
        raw_dist[y,x] = cv2.pointPolygonTest(contour, (int(x),int(y)), True) # 一定要保证是int
    # else: # len(contours) == 2:
    #     # have one hole in mask
    #     for i in range(len(mask_coord)):
    #         x = mask_coord[i][0]
    #         y = mask_coord[i][1]
    #         distance_to_internal = cv2.pointPolygonTest(contours[1], (int(x),int(y)), True)
    #         distance_to_external = cv2.pointPolygonTest(contours[0], (int(x),int(y)), True)
    #         if  abs(distance_to_external - abs(distance_to_internal)) < 0.3:
    #             raw_dist[y,x] = distance_to_external

    _, max_val, _, max_idx = cv2.minMaxLoc(raw_dist)
    if max_val < 0:
        result = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(result, [contour], 0, (255,0,0), 2)
        cv2.imshow('result_max_val < 0', result)
        cv2.waitKey()
        return None,None

    if visual:
        result = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        cv2.circle(result,max_idx, np.int(max_val),(0,255,0), 2, cv2.LINE_8, 0)
        cv2.imshow('result', result)  
    # return circle xy coord and fp32 radius 
    if isinstance(max_idx, tuple):
        max_idx = np.array(max_idx)[::-1]
    return max_idx, max_val


def apply_mask_to_img(mask,imgs,color2gray,visual, mask_info):
    '''
    用mask把img_list中的图像分割出来,其中mask=0的位置全涂黑,否则使用原图像素值
    :param mask: 二维的二值mask
    :param imgs: 所有图片,可以是单张图片或图片列表
    :param color2gray: 是否把彩色图像转为灰度图像
    :param visual: 是否可视结果
    :param mask_info:mask相关信息,用以生成不同的mask窗口
    :return: 分割后的图像或图像列表
    '''
    if isinstance(imgs, list):
        apply_list = []
        for i,img in enumerate(imgs):#enumerate中对list的处理是隔离的,不会影响原来的list
            # print(i)
            if is_grayscale(img):
                img = np.where(mask,img,0)
            else:
                img = np.where(np.repeat(mask[:, :, np.newaxis], 3, axis=-1), img, 0)
                if color2gray:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            apply_list.append(img)
            if visual:
                cv2.imshow('apply {} mask to img: img {} in img list'.format(mask_info, i), img)
        return apply_list
    else:
        if is_grayscale(imgs):
            img = np.where(mask,imgs,0)
        else:
            img = np.where(np.repeat(mask[:, :, np.newaxis], 3, axis=-1), imgs, 0)
            if color2gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if visual:
            cv2.imshow('apply {} mask to img'.format(mask_info), img)
        return img


def put_mask_on_img(mask, imgs, visual, mask_info):
    '''
    把半透明的红色mask覆盖在img_list中的所有图像上并（在visual=True时）显示
    :param mask: 二维的二值mask
    :param imgs: 所有图片,可以是单张图片或图片列表
    :param visual: 是否可视化
    :param mask_info:mask相关信息,用以生成不同的mask窗口
    :return: mask覆盖在图像上的三通道图像或图像列表
    '''
    mask_vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    if isinstance(imgs, list):
        put_list = []
        for i, img in enumerate(imgs):
            if is_grayscale(img):
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            mask_vis[:, :, 2] = np.where(mask, 255, 0)#只对bgr的r通道赋值,也就是给vis涂上红色
            mask_on_img = cv2.addWeighted(img, 0.5, mask_vis, 0.5, 0)
            put_list.append(mask_on_img)
            if visual:
                cv2.imshow('put {} mask on img: img {} in img list'.format(mask_info, i), mask_on_img)
        return put_list
    else:
        if is_grayscale(imgs):
            imgs = cv2.cvtColor(imgs, cv2.COLOR_GRAY2BGR)
        mask_vis[:, :, 2] = np.where(mask, 255, 0)#只对bgr的r通道赋值,也就是给vis涂上红色
        mask_on_img = cv2.addWeighted(imgs, 1, mask_vis, 0.5, 0)
        if visual:
            cv2.imshow('put {} mask on img'.format(mask_info), mask_on_img)
        return mask_on_img