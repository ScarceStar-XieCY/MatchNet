import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from tools.interact.calculate_angle import load_info_dict, dump_info_dict, filter_sort_image
from tools.image_mask.mask_process import remove_inner_black, mask2coord,coord2mask
from collect_data.suction_mask import kmeans_image
import cv2
import logging
import pickle
logger = logging.getLogger(__name__)

KIT_PIECE = {"bear":6,"bug":4,"bug_rev":4,"butterfly":5, "butterfly_rev":5,"rabbit":4,"whale":5,"car":7,"paint":1,}
DICT_NAME_LIST = ["angle","obj_mask","corres_mask","center","kit_mask",]

def dump_images(load_dir_name,file_list, img_idx, dump_dir):
    """"""
    cur_image_path = os.path.join(load_dir_name, file_list[img_idx])
    pre_image_path = os.path.join(load_dir_name, file_list[img_idx - 1])

    cur_image = cv2.imread(cur_image_path)
    pre_image = cv2.imread(pre_image_path)

    cur_gray = cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)
    pre_gray = cv2.cvtColor(pre_image, cv2.COLOR_BGR2GRAY)

    cur_depth_path = os.path.join(load_dir_name, file_list[img_idx].replace("color","depth"))
    pre_depth_path = os.path.join(load_dir_name, file_list[img_idx - 1].replace("color","depth"))
    cur_depth = cv2.imread(cur_depth_path,cv2.IMREAD_UNCHANGED)
    pre_depth = cv2.imread(pre_depth_path,cv2.IMREAD_UNCHANGED)
    
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    # save 6 image, inclue bgr, gray and depth image of init and final
    save_name_list = ["init_color.png","final_color.png", "init_gray.png","final_gray.png","init_depth.png","final_depth.png"]
    save_image_list = [pre_image, cur_image, pre_gray, cur_gray, pre_depth, cur_depth]
    for file_name, file_image in zip(save_name_list, save_image_list):
        cv2.imwrite(os.path.join(dump_dir,file_name),file_image)
    return cur_image


def calculate_kit_no_hole_mask(depth_image, visual, compare_image=None):
    """Calculate kit no hole mask."""
    k = 3
    cv2.imshow("depth", depth_image)
    cv2.imshow("compare", compare_image)

    if compare_image is not None:
        depth_image = cv2.subtract(compare_image,depth_image)
    cv2.imshow("sub", depth_image)

    diff_gray = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    diff_gray[diff_gray != 0] = 255
    cv2.imshow("diff_gray", diff_gray)
    
    cv2.waitKey()
    # kmeans_label, label_center_value = kmeans_image(depth_image,k, True)
    # # TODO: sort as num and select the middle one
    # cv2.imshow("kmeans", (kmeans_label * 20).astype("uint8"))
    # sorted_index = np.argsort(label_center_value) # increasing index
    # kit_label = sorted_index[1]
    # kit_mask = (np.where(kmeans_label == kit_label, 255, 0)).astype("uint8")

    kit_no_hole_mask = remove_inner_black(kit_mask, False) 
    kit_no_hole = mask2coord(kit_no_hole_mask, need_xy=False)
    if visual:
        cv2.imshow("kit_no_hole",kit_no_hole_mask)
    return kit_no_hole


def update_dump_info_dict(dict_list, load_key, info_dict, dump_dir):
    angle_dict, obj_mask_dict, corres_dict, center_dict, kit_mask = dict_list
    key_list = ["delta_angle","obj","hole","kit_with_hole","kit_no_hole","corres_obj","corres_hole","corres","init_point","final_point"]
    for key in key_list:
        if info_dict.get(key, None) is None:
            info_dict[key] = []
    info_dict["delta_angle"].append(angle_dict[load_key])
    info_dict["obj"].append(mask2coord(obj_mask_dict[load_key][0], need_xy=False))
    info_dict["hole"].append(mask2coord(obj_mask_dict[load_key][1], need_xy=False))
    info_dict["corres_obj"].append(corres_dict[load_key][0])
    info_dict["corres_hole"].append(corres_dict[load_key][1])
    info_dict["corres"].append(np.concatenate([corres_dict[load_key][1], corres_dict[load_key][0]], axis=1)) # source, target
    info_dict["final_point"].append(center_dict[load_key][0][::-1])
    info_dict["init_point"].append(center_dict[load_key][1][::-1])
    # calculate kit mask

    info_dict["kit_no_hole"].append(kit_mask[load_key])
    h,w = obj_mask_dict[load_key][0].shape[:2]
    kit_no_hole_mask = coord2mask(kit_mask[load_key], h,w,visual=False)
    hole_mask = coord2mask(info_dict["hole"][-1], h,w,visual=False)
    kit_with_hole = cv2.subtract(kit_no_hole_mask, hole_mask)
    cv2.imshow("kit_with_hole", kit_with_hole)
    kit = mask2coord(kit_with_hole, need_xy=False)
    info_dict["kit_with_hole"].append(kit)

    # dump
    pickle.dump(info_dict,open(os.path.join(dump_dir, "info_dict.pkl"),"wb"))

    



if __name__ == "__main__":
    skip_kit = ["bee","bee_rev","butterfly","column","circle_square","bug","math","snail","snail_rev"]
    dir_path = os.path.join("20230108","16_kit_color")
    dump_path = os.path.join("20230108","datasets_mix0128")
    dict_list = load_info_dict(DICT_NAME_LIST, dir_path)
    angle_dict, obj_mask_dict, corres_dict, center_dict, kit_mask = dict_list
    info_dict = {} # to save labels
    time_step_start = 0
    compare_depth = cv2.imread("20230108_compare\\train\color0.png", cv2.IMREAD_UNCHANGED)
    for root_dir_name, dir_list, file_list in os.walk(dir_path):
        ret_status = None
        kit_name = os.path.basename(root_dir_name)
        # dump_dir = os.path.join(dump_path, kit_name)
        if kit_name in skip_kit:
            continue
        
        file_list = filter_sort_image(file_list)
        time_step_index = time_step_start
        for img_idx in range(1,len(file_list)):
            if (img_idx)% (KIT_PIECE[kit_name] + 1) == 0:
                # is the start of one kit
                info_dict.clear() # clear info for last kit
                continue

            cur_image_path = os.path.join(root_dir_name, file_list[img_idx])
            if corres_dict.get(cur_image_path,None) is None:
                logger.warning("No label for %s", cur_image_path)
                continue
            # has labeled
            load_dir_name = root_dir_name.replace("16_kit_color", "16_kit")

            # random chose to train, valid and test
            split_type = ["train","valid","test"]
            weights = np.array([0.8,0.1,0.1])
            random_index = np.random.choice(3,p=weights)
            random_type = split_type[random_index]
            dump_dir = os.path.join(dump_path, random_type, "__".join([kit_name,str(time_step_index)]))
            if not os.path.exists(dump_dir):
                os.makedirs(dump_dir)
            cur_image = dump_images(load_dir_name,file_list, img_idx, dump_dir)
            update_dump_info_dict(dict_list, cur_image_path, info_dict, dump_dir)
            time_step_index += 1
            # cv2.waitKey()


