"""Get mask when collect."""
# coding=UTF-8
import cv2
import numpy as np
import logging
import sys
import os
import random
# sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.getcwd())
from tools.image_mask.mask_process import remove_big_area,remove_small_area,erode,get_avaliable_part,get_half_centroid_mask,remove_inner_black,apply_mask_to_img,remove_scattered_pix,get_each_mask,get_max_inner_circle,put_mask_on_img,remove_slim
from tools.image_mask.image_process import grabcut_get_mask,convert_image,get_exter_contours
logger = logging.getLogger(__name__)
VALID_RADIUS = 5

def kmeans_image(image,k, need_center_value:bool=False):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) 
    height, width  = image.shape[:2]
    # channel = 1 if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1) else 3
    # kmeans needs fp32 format image
    image_f32 = image.astype("float32").reshape((height * width,-1))
    ret, label, label_center_value=cv2.kmeans(image_f32, k, None, criteria,10, cv2.KMEANS_RANDOM_CENTERS)
    label = label.reshape(image.shape[:2])
    if need_center_value:
        return label,label_center_value
    return label


def seg_depth(image, visual):
    # preprocess for depth
    height, width = image.shape

    
    # top_d = image[0, width//2]
    # bottom_d = image[-1, width // 2]
    # assert top_d >= bottom_d
    # diff = top_d - bottom_d
    # scale = diff / height
    # for i in range(height):
    #     image_f32[i] += (i * scale)
    # image_f32 -= (diff // 2)

    # center_d = image[-1, width // 2]
    # right_d = image[-1, -1]
    # left_d = image[-1, 0]
 
    # # add by half width
    # # assert center_d < right_d
    # # diff = right_d - center_d
    # # scale_right = diff/ (width // 2)
    # # for j in range (width//2):
    # #     image_f32[:,j] += (j * scale_right)
    # #     image_f32[:, width - j - 1] += (j * scale_right)
    # # image_f32 -= (right_d - center_d) // 2

    # # add by whole width
    # assert right_d > left_d
    # diff = right_d - left_d
    # scale = diff / width
    # for j in range(width):
    #     image_f32[:, width - 1 - j] += (j * scale)
    # image_f32 -= (diff // 2)
    # cv2.imshow("image after adding",np.uint8(image_f32))

    # use kmeans to seg, need to get obj in kit

    # for woden puzzle, 1. bg 2.kit out of kit and kit 3. obj in the kit
    # inner obj label is different from bg and kit ,and obj out of kit
    # if there is obj out of kit, obj's label is same
    # if there is no obj out of kit, obj's label is different from background(bg) and kit's label
    k = 3
    label_list = [i for i in range(k)]

    classified_image = kmeans_image(image, k)
    bg_label = classified_image[-1,-1]
    label_list.remove(bg_label)

    _, _, _, max_idx = cv2.minMaxLoc(image)

    obj_label = classified_image[max_idx[1], max_idx[0]]
    obj_mask = np.where(classified_image == obj_label, 255, 0)
    inner_obj_mask = get_half_centroid_mask(obj_mask, False, 0)

    if image[:,width//2:].max() < image[:,:width//2].max(): 
        # consider as kit witout any obj in it
        if visual:
            cv2.imshow("classified_image * 30", classified_image.astype("uint8") * 30)
            cv2.imshow("seg_mask ", np.zeros((height,width),dtype = np.uint8))
            cv2.waitKey() 
       
        return np.zeros((height,width),dtype = np.uint8)
    
    seg_mask = inner_obj_mask.astype('uint8')    
    seg_mask = remove_small_area(seg_mask,500, False,"depth mask")
    if visual:
        cv2.imshow("classified_image * 30", classified_image.astype("uint8") * 30)
        cv2.imshow("seg_mask ", seg_mask)
        cv2.waitKey() 
    return seg_mask
        
def seg_color_by_grabcut(color_image, pre_mask, color_space:str = "hsv",visual:bool=False):
    
    grabcut_mask = grabcut_get_mask(color_image, pre_mask, color_space, False)
    grabcut_mask = remove_small_area(grabcut_mask,500,False,"")
    grabcut_mask = remove_inner_black(grabcut_mask, False)
    # if all kit is treat as obj, remove it
    grabcut_mask = remove_big_area(grabcut_mask, 10000, False, "")
    if visual:
        cv2.imshow("grabcut_after_process", grabcut_mask)
    return grabcut_mask

def seg_depth_by_time(init_image, final_image):
    # diff_image = cv2.subtract(init_image, final_image)
    diff_image = np.uint8(abs(np.int8(final_image) - np.int8(init_image)))
    # diff_image = cv2.GaussianBlur(diff_image, ksize=(3, 3), sigmaX=0, dst=None, sigmaY=None,borderType=None)
    diff_image *= int(255 / diff_image.max())
    diff_mask = cv2.adaptiveThreshold(np.uint8(diff_image), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61,0)
    diff_mask = erode(diff_mask,3,1)
    diff_mask = remove_small_area(diff_mask, 1000, False, "")
    cv2.imshow("diff_image",diff_mask)
    cv2.waitKey()
    return diff_mask


def seg_color_by_kmeans(color_image,k, color_space = "bgr", channel = [0,1,2], visual = False):
    height, width = color_image.shape[:2]
    converted_image = convert_image(color_image, color_space)
    channel = np.array(channel)
    image = converted_image[:,:,channel]
    # if len(channel) ==1:
    #     image = converted_image[:,:,channel[0]]
    # elif len(channel) ==3:
    #     image = converted_image
    # else:
    #     image = converted_image[:,:,channel[0]]
    #     for c in channel[1:]:
    #         image = np.concatenate([image, converted_image[:,:,c]], axis = -1) #TODP axis
    classified_color_image = kmeans_image(image, k)
    if visual:
        # cv2.imshow("image", np.uint8(image))
        cv2.imshow("classified_color_image", classified_color_image.astype("uint8")*20)
        # cv2.imshow("color",color_image)
        cv2.waitKey()
    return classified_color_image

# def compose_depth_color(depth_image, color_image, visual):
#     depth_seg_mask = seg_depth(depth_image)
#     # get inner obj mask
#     seg_result = seg_color_by_grabcut(color_image, depth_seg_mask,depth_image)
#     # get color classfy label
#     k = 5
#     color_label = seg_color_by_kmeans(color_image, k)
#     for i in range(k):
#         one_color_mask = np.where(color_label==i, 255, 0).astype("uint8")
#         one_color_mask = get_avaliable_part(one_color_mask, seg_result, False)
#         one_color_mask = remove_inner_black(one_color_mask, False)
#         one_color_mask = remove_small_area(one_color_mask, 500, False, "")
        
#         if (one_color_mask == 0).all():
#             continue
#         if visual:
#             cv2.imshow(f"one_color_classify_{i}",one_color_mask)
#             cv2.waitKey()

def get_obj_mask_in_kit(compare_depth,depth_image,color_image, color_space):
    diff_depth = cv2.subtract(compare_depth, depth_image)
    scale = 255 / depth_image.max()
    # cv2.imshow("color",color_image)
    # cv2.imshow("diff", np.int8(diff_depth * scale))
    # cv2.waitKey()
    # # use kmeans to seg depth
    depth_mask = seg_depth(diff_depth,False)
    if (depth_mask == 0).all():
        # no obj in kit
        return np.zeros_like(depth_image, dtype=np.uint8)
    seg_result = seg_color_by_grabcut(color_image, depth_mask, color_space, visual = False)
    if (seg_result == 0).all(): 
        # if grabcut is invalid
        seg_result = depth_mask
    return seg_result


def get_each_color_mask(color_label):
    one_color_mask_list = []
    for i in range(color_label.max()):
        one_color_mask = np.where(color_label==i, 255, 0).astype("uint8")
        one_color_mask = remove_scattered_pix(one_color_mask,3,False)
        one_color_mask = remove_inner_black(one_color_mask, False)
        one_color_mask = remove_small_area(one_color_mask, 500, False, "")
        if (one_color_mask == 0).all():
            continue
        one_color_mask_list.append(one_color_mask)
    return one_color_mask_list


def get_each_suction_coord(mask_list,visual:bool,visual_image):
    obj_coord = []
    valid_list = []
    for idx,each_domain_mask in enumerate(mask_list):
        # cv2.imshow("each mask",np.uint8(each_domain_mask))
        # cv2.waitKey()
        uv_coord, radius = get_max_inner_circle(each_domain_mask, True)
        obj_coord.append((uv_coord, radius))
        if uv_coord is not None and radius >= VALID_RADIUS:
            valid_list.append(idx)
        # cv2.imshow("one_color_classify",one_color_mask)
    assert len(obj_coord) == len(mask_list)
    if visual:
        visual_ing =visual_image.copy()
        for idx, ((uv_coord, radius),mask) in enumerate(zip(obj_coord, mask_list)):
            if idx not in valid_list:
                visual_ing = put_mask_on_img(mask, visual_ing, False,"",(128,0,128))
            else:
                visual_ing = put_mask_on_img(mask, visual_ing, False,"",(0,128,0))
                cv2.circle(visual_ing,uv_coord[::-1], int(radius),(0,255,0), 1, cv2.LINE_8, 0)
            # cv2.imshow('result', visual_ing)
            pass    
    return obj_coord


def get_each_domain(color_label,visual:bool):
    """Get each domain."""
    each_domain_mask_list = []
    for i in range(color_label.max()+1):
        one_color_mask = np.where(color_label==i, 255, 0).astype("uint8")
        each_mask_list = get_each_mask(one_color_mask)
        for each_mask in each_mask_list:
            each_mask = remove_scattered_pix(each_mask,3,False)
            each_mask = remove_inner_black(each_mask, False)
            each_mask = remove_small_area(each_mask, 200, False, "")
            each_mask = remove_big_area(each_mask, 2000,False,"")
            if (each_mask == 0).all():
                continue
            each_domain_mask_list.append(each_mask)
            if visual:
                cv2.imshow("one domain", each_mask)
                cv2.waitKey()
    return each_domain_mask_list


def adap_get_obj_domain(color_img,k,obj_num,valid_mask):
    color_space_list = ["bgr","ycrcb","luv"]
    mask_num_list =[]
    mak_list_record = []
    for color_space in color_space_list:
        color_label = seg_color_by_kmeans(color_img, k, color_space, [0,1,2], visual=False)
        color_label= np.where(valid_mask, color_label, -1) 
        each_domain_mask_list = get_each_domain(color_label,False)
        if len(each_domain_mask_list) == obj_num:
            logger.info("In func: adap_get_obj_domain, get proper mask list.")
            return each_domain_mask_list
        mask_num_list.append(len(each_domain_mask_list))
        mak_list_record.append(each_domain_mask_list)
    mask_num_array = np.array(mask_num_list)
    diff_abs = np.abs(mask_num_array - obj_num)
    min_diff_idx = np.argmin(diff_abs)
    logger.warning("can't get proper mask, find least diff masks, have %s mask", mask_num_array[min_diff_idx])
    return mak_list_record[min_diff_idx]


def get_obj_coord_with_mask_2d(compare_depth,depth_image, color_image, kmeans_k, obj_num):
    seg_result = get_obj_mask_in_kit(compare_depth,depth_image, color_image,"bgr")
    # get all obj in kit
    color_obj = apply_mask_to_img(seg_result, color_image, False, False,"seg_result")
    # segment each color, 2 means background and eye
    each_domain_mask_list = adap_get_obj_domain(color_obj,kmeans_k,obj_num,seg_result)
    obj_coord = get_each_suction_coord(each_domain_mask_list, visual=False,visual_image=color_image)

    return obj_coord,each_domain_mask_list


def test_inner_circle():
    # data_root = os.path.join('20230108',"datasets_mixbr")
    data_root = os.path.join('20221029test')
    data_type = "train"
    kit_name = "bear"
    file_name_list = os.listdir(os.path.join(data_root,data_type))
    step_num = len(file_name_list) // 3
    compare_depth  =  cv2.imread(os.path.join("20221029test_compare",data_type,f"depth0.png"), cv2.IMREAD_GRAYSCALE)
    obj_num_init= 6 # kit obj num +2
    kit_count = 0
    while True:
        # for one kit
        # suction stage
        # obj_num= obj_num_init - (i % obj_num_init)
        obj_num = obj_num_init
        obj_coord = []
        while obj_num > 0:
            # get image 
            # i = (obj_num_init - obj_num) + obj_num_init * kit_count
            i = 26
            print(i)
            depth_image = cv2.imread(os.path.join(data_root,data_type,  f"depth{i}.png"), cv2.IMREAD_GRAYSCALE)
            color_image = cv2.imread(os.path.join(data_root,data_type,  f"color{i}.png"))
            obj_coord,mask_list = get_obj_coord_with_mask_2d(compare_depth,depth_image, color_image, obj_num_init + 2, obj_num_init)
            cv2.waitKey()
         
        # placement stage

        kit_count += 1




# if __name__ == "__main__": 
#     test_inner_circle()