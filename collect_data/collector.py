import os
import cv2
from suction_mask import get_obj_coord_with_mask_2d
from overlap_detect import OverlapDetector
from tools.hardware.camera.cam_d435i import initial_camera, get_curr_image
from tools.hardware.arm.calibration import gen_coords
from tools.matrix import rigid_trans_mask_around_point
from tools.image_mask.mask_process import mask2coord, coord2mask
from tools.hardware.arm.operation import MyRobot
import random
import logging
import pickle
from tqdm import tqdm

logger = logging.getLogger(__file__)
KIT_NUM = {"rabbit":4,"whale":5,"bear":6}
# KIT_NAME = 0
DUMP_PATH = "dataset0108_raw"

INIT_POINT = "init_point"
FINAL_POINT = "final_point"
FINAL_POINT3D = "final_point3d"
ROTATION = "rotation"
HOLE_MASK = "hole_mask"
OBJ_MASK = "obj_mask"

def find_coord_to_place(robot,mask, coord):
    count = 0
    while True:
        random_place_3d, random_theta = gen_coords(method = "random")
        random_place_2d = robot.b2c(random_place_3d)
        try:
            robot.arm_try_move(random_place_3d, random_theta, 70, do_move=False)
        except:
            logger.warning(f"Robot can't reach coord: {random_place_3d}, rotation {random_theta}")
            continue
        logger.warning("find a reacheable coord, detecting overlap...")
        if not detector.detect_overlap(mask, random_theta, coord, random_place_2d):
            logger.warning("find a place")
            return random_place_3d,random_place_2d,random_theta
        count += 1
        if count >= 10:
            logger.warning("can't find place after %s's try.",count)


if __name__ == "__main__":
    debug_mode = False
    kit_name = "bear"
    robot_id = 18
    robot = MyRobot(robot_id,mode="wooden",need_cali=False, debug_mode=debug_mode)
    if not debug_mode:
        dump_path = os.path.join(os.getcwd(),DUMP_PATH,kit_name)
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        # camera
        pipeline,align = initial_camera()
        compare_color, compare_depth = get_curr_image(pipeline,align)
        cv2.imwrite(os.path.join(dump_path,"compare_color.png"),compare_color)
        cv2.imwrite(os.path.join(dump_path,"compare_depth.png"),compare_depth)
        cv2.imshow("compare",compare_color)
        cv2.waitKey()
    else:
        dump_path = "collection_debug"
        assert os.path.exists(dump_path)
        compare_depth = cv2.imread(os.path.join(dump_path,"compare_depth.png"), cv2.IMREAD_GRAYSCALE)
    h,w = compare_depth.shape[:2]
    
    
    # compare_depth  =  cv2.imread(os.path.join("20230108_compare","train", f"depth0.png"), cv2.IMREAD_GRAYSCALE)
    detector = OverlapDetector(compare_depth.shape)
    arm_info_dict = {}
    i = 0
    for kit_index in tqdm(range(0,200)):
        # init status
        kit_start = i
        if not debug_mode:
            color_image, depth_image = get_curr_image(pipeline,align)
            cv2.imwrite(os.path.join(dump_path,f"color{i}.png"),color_image)
            cv2.imwrite(os.path.join(dump_path,f"depth{i}.png"),depth_image)
        else:
            depth_image = cv2.imread(os.path.join(dump_path,f"depth{kit_index}.png"), cv2.IMREAD_GRAYSCALE)
            color_image = cv2.imread(os.path.join(dump_path,f"color{kit_index}.png"))
        arm_info_dict[i] = None
        pickle.dump(arm_info_dict, open(os.path.join(dump_path,f"arm_info_dict.pkl"), "wb"))
        obj_coord, mask_list = get_obj_coord_with_mask_2d(compare_depth,depth_image, color_image, KIT_NUM[kit_name]+2, KIT_NUM[kit_name])
        # if cv2.waitKey() == ord("q"):
        #     break
        print("len(obj_coord) = ",len(obj_coord))
        assert len(obj_coord) == KIT_NUM[kit_name]

        # dissamble stage
        obj_index  = list(range(len(obj_coord)))
        random.shuffle(obj_index)
        for one_random_idx in obj_index:
            i += 1
            uv_coord, radius = obj_coord[one_random_idx]
            selected_mask = mask_list[one_random_idx]
            robot.arm_to_coord2d(uv_coord, 0,place=False,to_kit=True) #suc from kit
            # find coord to place
            random_place_3d,random_place_2d,random_theta = find_coord_to_place(robot, selected_mask, uv_coord)
            obj_mask_coord = rigid_trans_mask_around_point(selected_mask, random_theta, uv_coord,random_place_2d)
            obj_mask = coord2mask(obj_coord,h,w,False)
            arm_info_dict[i] = {INIT_POINT:uv_coord, FINAL_POINT:random_place_2d,ROTATION:random_theta,FINAL_POINT3D:random_place_3d,HOLE_MASK:selected_mask,OBJ_MASK:obj_mask_coord}
            robot.arm_to_coord_updown(random_place_3d,random_theta, place=True) # place to outer
            robot.arm_reset_pos()
            color_image, depth_image = get_curr_image(pipeline,align)
            cv2.imwrite(os.path.join(dump_path,f"color{i}.png"),color_image)
            cv2.imwrite(os.path.join(dump_path,f"depth{i}.png"),depth_image)
            pickle.dump(arm_info_dict, open(os.path.join(dump_path,f"arm_info_dict.pkl"), "wb"))

        # assamble stage
        for time_step in range(kit_start,kit_start +KIT_NUM[kit_name]+1):
            cur_step_info = arm_info_dict[time_step]
            if cur_step_info is None: # skip the timestep that all obj is in kit
                continue
            suc_point_coord = cur_step_info[FINAL_POINT3D]
            suc_point_coord = (suc_point_coord[0],suc_point_coord[1],suc_point_coord[2]-5)
            robot.arm_to_coord_updown(suc_point_coord, cur_step_info[ROTATION], place = False) # suc outer obj
            robot.arm_to_coord2d(cur_step_info[INIT_POINT], 0, place=True, to_kit=True) # place to kit
        robot.arm_reset_pos()