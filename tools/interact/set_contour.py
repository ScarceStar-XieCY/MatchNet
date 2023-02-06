"""Interact tool for calculate angle."""
# coding=UTF-8

import numpy as np
import cv2
import logging
import pickle
import os
import sys
sys.path.append(os.getcwd())
from tools.image_mask.mask_process import get_mask_center,get_half_centroid_mask,remove_small_area, erode,get_avaliable_part,get_union,mask2coord,coord2mask,remain_largest_area, find_cavity
from tools.image_mask.image_process import grabcut_get_mask,put_mask_on_img
from tools.manager.log_manager import LOGGING_CONFIG
from tools.matrix import rigid_trans_mask_around_point, reverse_get_corres

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("console_logger")

DICT_NAME_LIST = ["kit_mask",]
DICT_SUFFIX = "_dict.pkl"



def draw_poly(image, pt_list,visual):
    pt_array = np.array(pt_list)
    poly_mask = np.zeros((image.shape[:2]), dtype="uint8")
    cv2.fillConvexPoly(poly_mask, pt_array, 255)
    put_mask_on_img(poly_mask, image, visual,"")
    return mask2coord(poly_mask, need_xy=False)

def window_react(event,x,y,flags,param):
    """Window reaction."""
    image,window_name, pt_list, _, _,_,_ = param
    if event == cv2.EVENT_LBUTTONDOWN:
        pt_list.append((x,y))
        logger.info("Add point (%s, %s)",x,y)
    if event == cv2.EVENT_RBUTTONDOWN:
        x,y = pt_list.pop(-1)
        logger.info("Delete point (%s,%s)",x,y)
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        img = image.copy()
        kit_coord = draw_poly(img, pt_list, True)
        param[-1] = kit_coord
        cv2.imshow(window_name, img)

def draw_exsist_mask(image, mask_coord,h,w, message):
    poly_mask = coord2mask(mask_coord,h,w,False)
    put_mask_on_img(poly_mask, image.copy(),True,message)

def label_one_image(on_mouse_param,):
    """Label:first click on obj out of kit, then the obj in the kit."""
    image,window_name, pt_list, image_path, kit_mask, last_kit_mask = on_mouse_param
    on_mouse_param.append(None)
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, window_react, on_mouse_param)
    # control with keyboard
    # q
    h,w = image.shape[:2]
    if kit_mask.get(image_path,None) is not None:
        draw_exsist_mask(image,kit_mask[image_path],h,w,"exsist")
    if last_kit_mask is not None:
        draw_exsist_mask(image,last_kit_mask,h,w,"last")


    while True:
        key = cv2.waitKey()
        pt_list = on_mouse_param[2]
        if key not in [ord("z"),ord("s"),ord("q"),ord("a")]:
            logger.info("Press z to skip this image; press s to save if possible; press q to quit and save current, press a to use last kit mask")
            continue
        # skip this image when press "z"
        if key == ord("z"):
            logger.info("skip this image")
            return 0
        if key == ord("a"):
            kit_mask[image_path]= last_kit_mask
            return 0
        # confirm save
        if key == ord("s") and len(pt_list) >= 4:
            kit_mask[image_path] = param[-1]
            return 0
        # quit
        if key == ord("q"):
            cv2.destroyWindow(window_name)
            return 1
        else: #  len(pt_list) <4:
            print("len of pt",len(pt_list))
            logger.warning("No enough pt. %s", pt_list)
        

def _diff_mask(in_kit_img,out_kit_img,visual,kit_name):
    """Get diff mask list (left, right) for label."""
    diff_mask_list = []
    choose_left_half = [True, False]
    for left_half in choose_left_half:
        if left_half:
            img_intre = out_kit_img
            img_ref = in_kit_img
        else:
            img_intre = in_kit_img
            img_ref = out_kit_img
        diff_img = cv2.subtract(img_ref, img_intre)
        diff_gray = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
        # diff_gray = cv2.blur(diff_gray,(5,5)) 
        if kit_name not in ["bee"]:
            diff_gray[diff_gray < 10] = 0
        if kit_name in ["bee"]:
            diff_gray[diff_gray != 0] = 255
        # if visual:
        #     cv2.imshow("diff_gray", diff_gray.astype("uint8"))
        #     cv2.waitKey()
        dthresh, diff_mask = cv2.threshold(diff_gray, 0,255,cv2.THRESH_BINARY,cv2.THRESH_OTSU)
        # diff_mask = open_morph(diff_mask,3,2)
        diff_mask = erode(diff_mask,3,2)
        if visual:
            cv2.imshow("diff_open", diff_mask)
        diff_mask = get_half_centroid_mask(diff_mask, left_half,0,visual)
        # grabcut to confim color domain
        center_coord = get_mask_center(diff_mask,False,False)
        # diff_mask = np.zeros_like(diff_mask, dtype= "uint8")
        # draw_point_on_img(diff_mask,center_coord, 255)
        
        # if visual:
            # cv2.imshow("diff_erode", diff_mask)
        grabcut_mask = grabcut_get_mask(img_intre,diff_mask,"hsv",visual,sure_point = center_coord, use_roi = False)
        # diff_mask = remove_surrounding_white(diff_mask,False)
        # diff_mask = remove_scattered_pix(diff_mask,5,True)
        diff_mask = remove_small_area(diff_mask, 40,False,"remove small")
        diff_mask = get_avaliable_part(grabcut_mask, diff_mask,False)
        diff_mask = remain_largest_area(diff_mask, visual,"")
        # diff_mask = remove_small_area(diff_mask, 100, False, "")
        
        # cv2.waitKey()
        diff_mask_list.append(diff_mask)

    return diff_mask_list

def load_info_dict(dict_name_list, dir_path):
    dict_list = []
    for dict_name in dict_name_list:
        dict_dump_path = os.path.join(dir_path, dict_name + DICT_SUFFIX)
        if os.path.exists(dict_dump_path):
            dict_con = pickle.load(open(dict_dump_path,"rb"))
        else:
            dict_con = {}
        dict_list.append(dict_con)
    return dict_list

def dump_info_dict(dict_name_list, dict_list, dir_path):
    for dict_name, info_dict in zip(dict_name_list, dict_list): 
        dict_dump_path = os.path.join(dir_path, dict_name + DICT_SUFFIX)
        with open(dict_dump_path,"wb") as fin:
            pickle.dump(info_dict, fin)
        logger.info("save %s dict into %s", dict_name, dict_dump_path)


def filter_sort_image(file_list):
    """Filter and sort as image number in file_list."""
    if len(file_list) != 0: 
        # filter image only
        file_list = list(filter(lambda file_name:file_name[-4:] == ".png", file_list))
        # sort list as number
        file_list.sort(key = lambda file_name:float(file_name[5:].split(".")[0].replace("_",".")))
    return file_list


if __name__ == "__main__":
    # read image loop
    dir_path = os.path.join("20230108","16_kit_color")

    # create info dicts
    
    dict_list = load_info_dict(DICT_NAME_LIST, dir_path)
    kit_mask = dict_list[0]

    window_name = "angle_label"
    skip_kit = ["bear","bee","bee_rev","butterfly","bug","bug","butterfly","car","circle_square","column","math","paint"]
    for root_dir_name, dir_list, file_list in os.walk(dir_path):
        ret_status = None
        last_kit_mask = None
        if os.path.basename(root_dir_name) in skip_kit:
            continue
        file_list = filter_sort_image(file_list)
        for img_idx in range(0,len(file_list)):
            # read cur and pre image
            cur_image_path = os.path.join(root_dir_name, file_list[img_idx])
            pre_image_path = os.path.join(root_dir_name,file_list[img_idx - 1])
            cur_image = cv2.imread(cur_image_path)
            pre_image = cv2.imread(pre_image_path)
            # cv2.imshow("cur_image",cur_image)
            # cv2.imshow("pre_image", pre_image)
            pt_list = []
            # get diff and calculate mask 
            show_image = cv2.addWeighted(pre_image,0.5, cur_image, 0.5, 0)
            param = [cur_image,window_name, pt_list, cur_image_path, kit_mask, last_kit_mask]
            logger.warning("label %s",cur_image_path)
            ret_status = label_one_image(param)
            if kit_mask.get(cur_image_path,None) is not None:
                last_kit_mask = kit_mask[cur_image_path]
            if ret_status == 1:
                break
        if ret_status == 1:
            break
    
    # dump info dicts
    dump_info_dict(DICT_NAME_LIST,dict_list, dir_path)



