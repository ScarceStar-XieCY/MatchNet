import sys
import os
sys.path.append(os.getcwd())
from tools.hardware.arm.operation import MyRobot,CALI_ROOT,HEIGHT_OF_OBJ_WOODEN,HEIGHT_OF_OBJ_CIRCLE,RESET_COORD
import random
import logging
import cv2
import numpy as np
import pickle
from tools.hardware.camera.cam_d435i import initial_camera, get_curr_image
from tools.image_mask.mask_process import remove_surrounding_white,smallest_cc_with_zero,remove_scattered_pix,get_max_inner_circle
from tools.image_mask.image_process import convert_image
from collect_data.suction_mask import kmeans_image
from tqdm import tqdm


OBJ_HEIGHT = HEIGHT_OF_OBJ_WOODEN
logger = logging.getLogger(__name__)


def gen_coords(method:str= "grid"):
    """Generate coordinates."""
    coords = []
    if method == "grid":
        x = np.arange(50, 301, 50)
        y = np.arange(-250, 250, 50)
        x_grid, y_grid = np.meshgrid(x,y)
        x_grid = x_grid.flatten()
        y_grid = y_grid.flatten()
        for i in range(len(x_grid)):
            coords.append(((int(x_grid[i]), int(y_grid[i]), OBJ_HEIGHT), 0))
        return coords

    x1 = random.uniform(40, 250)
    y1 = random.uniform(-270, -40)
    randomrz = int(random.uniform(0, 360))                                   # 随机生成坐标,以及旋转角度，输出[x,y,z], rz
    logger.info(f"Generate coord: {((int(x1), int(y1), OBJ_HEIGHT), randomrz)}.")
    return ((int(x1), int(y1), OBJ_HEIGHT), randomrz)                   
    
 
def _get_obj_mask(image):
    hsv_image = convert_image(image,"hsv") 
    label = kmeans_image(hsv_image[:,:,1],2,False)
    obj_mask = smallest_cc_with_zero(label, True)
    obj_mask = remove_scattered_pix(obj_mask,5,False)
    if (obj_mask == 0).all():
        return None
    no_sur_obj = remove_surrounding_white(obj_mask,False, min_domain_num = 0)
    if (obj_mask != no_sur_obj).any():
        return None
    return obj_mask


def autocali2():                                    # 用于机械臂的自动标定。每一次让机械臂选择一个位置，放置物块，保存当前的机械臂坐标。
                                                    # 然后机械臂回归原点，相机截取当前图片，二值化取蓝色区域，求蓝色区域质心作为当前的相机坐标系坐标。
    # init position                                 # ***** 常用标定方法 *******
    objpoints = []
    imgpoints = []
    coords = gen_coords()                          # 随机生成放置的四个坐标list
    print("---------------init camera---------------")
    pipeline, align = initial_camera()

    print("-------------init the arm----------------")
    robot_id = 18
    robot = MyRobot(robot_id,need_cali=True)
    
    ################  开始标定   #################
    for (random_coord, random_rot) in tqdm(coords):         
        if not robot.arm_to_coord_updown(random_coord, random_rot, place=True): # place to the grid coord
            # can't reach, just skip
            logger.warning("can't reach coord: %s",random_coord)
            continue
        print("objpoint [{}, {}]".format(random_coord[0], random_coord[1]))
        robot.arm_to_coord(RESET_COORD, 0) # return to reset point
        color_image,_ = get_curr_image(pipeline, align)
        # cv2.imwrite("test.jpg", color_image)
        # get mask
        mask = _get_obj_mask(color_image)
        # get center
        if mask is None:
            # no obj in img ,skip
            logger.warning("no obj in img ,skip, coord: %s",random_coord)
            # while skip, the obj should be grasped
            robot.arm_to_coord_updown(random_coord, random_rot, place=False)
            continue
        uv_coord, _ = get_max_inner_circle(mask, False)
        objpoints.append([random_coord[0], random_coord[1],OBJ_HEIGHT])
        imgpoints.append(uv_coord)
        print(f"obj_point {random_coord} with img_point {uv_coord}")
        
        robot.arm_to_coord_updown(random_coord, random_rot, place=False)  # 机械臂把物体吸起，准备下一回合。
        pickle.dump(objpoints,open(os.path.join(CALI_ROOT,"obj_points.pkl"),"wb"))
        pickle.dump(imgpoints,open(os.path.join(CALI_ROOT,"img_points.pkl"),"wb"))

    pickle.dump(objpoints,open(os.path.join(CALI_ROOT,"obj_points.pkl"),"wb"))
    pickle.dump(imgpoints,open(os.path.join(CALI_ROOT,"img_points.pkl"),"wb"))
    print("points saved.")


def debug_cali():
    """Debug for calibration."""
    root = "calib"
    if not os.path.exists(root):
        os.makedirs(root)
    i = 0
    while True:
        color_iamg = cv2.imread(os.path.join(root,f"color{i}.png"))
        # cv2.imread(os.path.join(root,f"depth{i}.png"))
        obj_mask = _get_obj_mask(color_iamg)
        if obj_mask is not None:
            # valid mask
            i +=1

if __name__ == "__main__":
    autocali2()
