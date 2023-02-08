import sys
import os
sys.path.append(os.getcwd())
from tools.hardware.arm.operation import MyRobot
import time
import random
import logging
import cv2
import numpy as np
# from tools.hardware.arm.HitbotInterface import HitbotInterface
import pickle
from tools.hardware.camera.cam_d435i import initial_camera, get_curr_image
from tools.image_mask.mask_process import get_mask_center, dilate,remove_surrounding_white
from tools.image_mask.image_process import adap_get_mask_in_color_space
from tqdm import tqdm

OBJ_HEIGHT = -112 # -117 -110
logger = logging.getLogger(__file__)



def gen_coords(method:str= "grid"):
    """Generate coordinates."""
    coords = []
    if method == "grid":
        x = np.arange(90, 231, 60)
        y = np.arange(-180, 181, 60)
        x_grid, y_grid = np.meshgrid(x,y)
        for i in range(len(x_grid)):
            coords.append(((int(x_grid[i]), int(y_grid[i]), OBJ_HEIGHT), 0))
    else:
        x1 = random.uniform(90, 230)
        y1 = random.uniform(-180, -40)
        randomrz = int(random.uniform(0, 90))                                   # 随机生成坐标,以及旋转角度，输出[x,y,z], rz
        coords.append(((int(x1), int(y1), OBJ_HEIGHT), randomrz))                                 
    logger.info(f"Generate coord: {coords}.")
    return coords


def _get_obj_mask(image):
    mask_lab = adap_get_mask_in_color_space(image, 'lab', False)
    # mask_hsv = dilate(mask_hsv, 3, 2)
    mask_lab = dilate(mask_lab, 3, 1)
    mask_lab = remove_surrounding_white(mask_lab,False)
    return mask_lab

# def get_data_txt(dir='data.txt'):                   # 按照 [imgpoints0, imgpoints1, objpoints0, objpoints1, objpoints2] 的原则读取txt.
#     f = open(dir, 'r')
#     imgpoints = []
#     objpoints = []
#     for lines in f:
#         ls = lines.strip('\n').replace(' ','').replace('、','/').replace('?','').split(',')
#         print("img/n", imgpoints)
#         print("obj/n", objpoints)
#         imgpoints.append([float(ls[0]), float(ls[1])])
#         objpoints.append([float(ls[2]), float(ls[3]), float(ls[4])])
#     return imgpoints, objpoints


def autocali2():                                    # 用于机械臂的自动标定。每一次让机械臂选择一个位置，放置物块，保存当前的机械臂坐标。
                                                    # 然后机械臂回归原点，相机截取当前图片，二值化取蓝色区域，求蓝色区域质心作为当前的相机坐标系坐标。
    # init position                                 # ***** 常用标定方法 *******
    # box_pos = [-64.6159, 269.41, -39]            # 不遮挡相机拍摄的机械臂位置，每次结束放置后移动至该位置。
    objpoints = []
    imgpoints = []
    allpoints = []                          # allpoints用于存储适合放入神经网络的坐标。格式：[obj0,obj1,obj2,img0,img1]
    coords = gen_coords()                          # 随机生成放置的四个坐标list
    print("---------------init camera---------------")
    pipeline, align = initial_camera()

    print("-------------init the arm----------------")
    robot_id = 18
    robot = MyRobot(robot_id)

    ################  开始标定   #################
    for (random_coord, random_rot) in tqdm(coords):         
        robot.arm_to_coord(random_coord, random_rot, place=True)                       # 机械臂放置，并返回原位置，准备拍摄

        objpoints.append([random_coord[0], random_coord[1],OBJ_HEIGHT])
        print("objpoint [{}, {}]".format(random_coord[0], random_coord[1]))
        color_image,_ = get_curr_image(pipeline, align)                                      # 记录图像
        cv2.imwrite("test.jpg", color_image)
        # get mask
        mask = _get_obj_mask(color_image)
        # get center
        center = get_mask_center(mask)                                                  # 求质心
        imgpoints.append(center)
        # allpoints.append([center[0], center[1], coord2[0], coord2[1], OBJ_HEIGHT])
        print(f"obj_point {random_coord} with img_point {imgpoints}")
        
        robot.arm_to_coord(random_coord, random_rot, place=False)  # 机械臂把物体吸起，准备下一回合。
        pickle.dump(objpoints,open("obj_points.pkl","wb"))
        pickle.dump(imgpoints,open("img_points.pkl","wb"))

    pickle.dump(objpoints,open("obj_points.pkl","wb"))
    pickle.dump(imgpoints,open("img_points.pkl","wb"))
    print("points saved.")



if __name__ == "__main__":
    autocali2()
