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

DICT_NAME_LIST = ["angle","obj_mask","corres_mask","center",]
DICT_SUFFIX = "_dict.pkl"


def calculate_angle(pt1:tuple, pt2:tuple,):
    """Under opencv xy coord. X ---> clockwise is positive. pt1 is origin coord."""
    delta_y = pt2[1] - pt1[1]
    delta_x = pt2[0] - pt1[0]
    angle = - np.arctan2(delta_y,delta_x)
    return angle

def draw_point_on_img(image:np.ndarray,pt:tuple,color):
    """Draw point on image."""
    cv2.circle(image,pt,3,color,2)

def draw_points_on_img(image, pt_list,color:tuple=(0,255,0)):
    for pt in pt_list:
        draw_point_on_img(image, pt,color)

def draw_lines_on_image(image, pt_list):
    for i in range(0,len(pt_list)-1,2):
        draw_line_between_pts(image, pt_list[i],pt_list[i+1])

                
def calculate_angles(pt_list):
    angle_list = []
    for i in range(1,len(pt_list) ,2):
        angle = calculate_angle(pt_list[i],pt_list[i-1])
        angle_list.append(angle)
    return angle_list

def calculate_diff_angle(angle_list):
    delta_angle = angle_list[1] - angle_list[0]
    return delta_angle

def update_delta_angle(angle_list, angle_dict, image_path):
    # update dict when enough angle
    if len(angle_list) != 2:
        return None
    logger.info("angle list to put text %s", angle_list)
    delta_angle = calculate_diff_angle(angle_list)
    delta_angle = 180 * delta_angle / np.pi
    angle_dict[image_path] = delta_angle
    logger.info("angle = %s, for  %s", delta_angle,image_path)
    return delta_angle

def text_delta_angle(image_path, angle_dict, image):
    """Show text info of delta angle."""
    if angle_dict.get(image_path,None) is None:
        return
    delta_angle = angle_dict[image_path]
    cv2.putText(image,f"delta_angle = {delta_angle}",(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))


def draw_line_between_pts(image:np.ndarray,pt1:tuple,pt2:tuple):
    """Drawn line on image."""
    cv2.line(image, pt1,pt2,(0,255,0),1)


def update_corres_mask(image_path, obj_mask_dict, corres_dict,center_dict, delta_angle,is_degree):
    """"""
    if delta_angle is None:
        return
    obj_out_kit_mask, obj_in_kit_mask = obj_mask_dict[image_path]
    center_out_kit, center_in_kit = center_dict[image_path]
    # correspond mask
    obj_corres_in_kit_coord = rigid_trans_mask_around_point(obj_out_kit_mask, delta_angle, center_out_kit[::-1], center_in_kit[::-1],visual=True,is_degree=is_degree) 
    # fill cavity and get correspond point
    h , w = obj_out_kit_mask.shape[:2]
    cavity_coord = find_cavity(obj_corres_in_kit_coord, h,w) # cavity coord in left obj mask
    
    # the correspond coord to cavity in the left obj mask
    cavity_reverse_coord= reverse_get_corres(cavity_coord, delta_angle, center_out_kit[::-1], center_in_kit[::-1], is_degree=is_degree) 
    # cancatenace cavity corresponding coord in obj in& out of kit respectly.
    obj_complete_coord_in_kit = np.concatenate([obj_corres_in_kit_coord, cavity_coord], axis = 0)
    obj_corrs_coord_out_kit = np.concatenate([mask2coord(obj_out_kit_mask,need_xy=False),cavity_reverse_coord], axis = 0)
    assert len(obj_complete_coord_in_kit) == len(obj_corrs_coord_out_kit)
    # coord2mask(np.concatenate([obj_complete_coord_in_kit, obj_corrs_coord_out_kit],axis =0),h,w,False)
    corres_dict[image_path] = [obj_corrs_coord_out_kit, obj_complete_coord_in_kit] # left right
    

def draw_corres_mask(image_path, corres_dict, h,w, image):
    if corres_dict.get(image_path,None) is None:
        return
    mask1_corrd, mask2_coord = corres_dict[image_path]
    mask1 = coord2mask(mask1_corrd,h,w,False)
    mask2 = coord2mask(mask2_coord,h,w,False)
    corres_mask = get_union(mask1, mask2, False,"")
    put_mask_on_img(corres_mask,image.copy(),True,"corres mask")


def window_react(event,x,y,flags,param):
    """Window reaction."""
    image,window_name, pt_list, angle_list, image_path = param[:5]
    angle_dict, obj_mask_dict, corres_dict, center_dict = param[5:]
    if event == cv2.EVENT_LBUTTONDOWN:
        pt_list.append((x,y))
        logger.info("Add point (%s, %s)",x,y)
    if event == cv2.EVENT_RBUTTONDOWN:
        x,y = pt_list.pop(-1)
        logger.info("Delete point (%s,%s)",x,y)
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        img = image.copy()
        h,w = image.shape[:2]
        draw_points_on_img(img, pt_list)
        draw_lines_on_image(img, pt_list)
        angle_list = calculate_angles(pt_list)
        param[3] = angle_list
        delta_angle = update_delta_angle(angle_list, angle_dict, image_path)
        text_delta_angle(image_path, angle_dict, img)
        update_corres_mask(image_path, obj_mask_dict, corres_dict,center_dict, delta_angle,is_degree=True)
        draw_corres_mask(image_path, corres_dict, h,w, image) # will show corres mask window
        cv2.imshow(window_name, img)


def label_one_image(dict_list, on_mouse_param):
    """Label:first click on obj out of kit, then the obj in the kit."""
    image,window_name, _, _,_, = on_mouse_param
    on_mouse_param.extend(dict_list)
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, window_react, on_mouse_param)
    # control with keyboard
    # q
    while True:
        key = cv2.waitKey()
        angle_list = on_mouse_param[3]

        if key not in [ord("z"),ord("s"),ord("q")]:
            logger.info("Press z to skip this image; press s to save if possible; press q to quit and save current")
            continue
        # skip this image when press "z"
        if key == ord("z"):
            logger.info("skip this image")
            return 0

        # confirm save
        if key == ord("s") and len(angle_list) == 2:
            return 0
        # quit
        if key == ord("q"):
            cv2.destroyWindow(window_name)
            return 1
        if len(angle_list) != 2:
            print("len of pt",len(pt_list))
            logger.warning("No enough angle. %s", angle_list)
        

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


if __name__ == "__main__":
    # read image loop
    dir_path = os.path.join("20230108","16_kit_color")

    # create info dicts
    
    dict_list = load_info_dict(DICT_NAME_LIST, dir_path)
    angle_dict, obj_mask_dict, corres_dict, center_dict = dict_list

    window_name = "angle_label"
    skip_kit = ["bear","bee","bee_rev"]
    for root_dir_name, dir_list, file_list in os.walk(dir_path):
        ret_status = None
        if os.path.basename(root_dir_name) in skip_kit:
            continue
        if len(file_list) != 0: 
            # filter image only
            file_list = list(filter(lambda file_name:file_name[-4:] == ".png", file_list))
            # sort list as number
            file_list.sort(key = lambda file_name:int(file_name[5:].split(".")[0]))
        for img_idx in range(1,len(file_list)):
            # read cur and pre image
            cur_image_path = os.path.join(root_dir_name, file_list[img_idx])
            pre_image_path = os.path.join(root_dir_name,file_list[img_idx - 1])
            cur_image = cv2.imread(cur_image_path)
            pre_image = cv2.imread(pre_image_path)
            # cv2.imshow("cur_image",cur_image)
            # cv2.imshow("pre_image", pre_image)
            pt_list = []
            angle_list = []
            # get diff and calculate mask 
            try:
                diff_mask_list = _diff_mask(pre_image, cur_image,visual = False, kit_name = os.path.basename(root_dir_name))
                center_list = []
                for diff_mask in diff_mask_list:
                    center_coord = get_mask_center(diff_mask,multi_center=False,uv_coord = False)
                    center_list.append(center_coord)
                draw_points_on_img(cur_image,center_list,(255,0,255)) # purple
            except Exception:
                logger.warning("Can't get valid mask info. Skip. %s",cur_image_path)
                continue
            
            obj_mask_dict[cur_image_path] = diff_mask_list # left and right obj mask
            center_dict[cur_image_path] = center_list

            show_image = cv2.addWeighted(pre_image,0.5, cur_image, 0.5, 0)
            param = [show_image, window_name, pt_list, angle_list, cur_image_path]
            put_mask_on_img(get_union(diff_mask_list[0],diff_mask_list[1],False,"diff mask"), cur_image.copy(),True,"")
            h, w = diff_mask_list[0].shape[:2]
            draw_corres_mask(cur_image_path, corres_dict, h,w, show_image)
            ret_status = label_one_image(dict_list, param)

            if ret_status == 1:
                break
        if ret_status == 1:
            break
    
    # dump info dicts
    dump_info_dict(DICT_NAME_LIST,dict_list, dir_path)



