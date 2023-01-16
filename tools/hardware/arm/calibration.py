import sys
import os
sys.path.append(os.getcwd())
from tools.hardware.arm.operation import arm_placement,arm_suction,arm_init
import time
import random
import logging
import cv2
import numpy as np
# from tools.hardware.arm.HitbotInterface import HitbotInterface
import pickle
from tools.hardware.camera.cam_d435i import initial_camera, get_curr_image
from collect_data.mask_process import get_mask_center, dilate,remove_surrounding_white
from collect_data.image_process import adap_get_mask_in_color_space


OBJ_HEIGHT = -112 # -117 -110
logger = logging.getLogger(__file__)

def gen_coords(method:str= "grid",epoch:int = 100):
    """Generate coordinates."""
    coords = []
    if method == "grid":
        
        x1,y1 = 90, -100
        coords.append(((int(x1), int(y1), OBJ_HEIGHT), 0))
        i = 0
        while(i<epoch):
            if x1 < 230:
                x1 = x1 + 40
            else:
                y1 = y1 + 40
                x1 = 90
            coords.append(((int(x1), int(y1), OBJ_HEIGHT), 0))
            i += 1
    else:
        for _ in range(epoch):
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

def get_data_txt(dir='data.txt'):                   # 按照 [imgpoints0, imgpoints1, objpoints0, objpoints1, objpoints2] 的原则读取txt.
    f = open(dir, 'r')
    imgpoints = []
    objpoints = []
    for lines in f:
        ls = lines.strip('\n').replace(' ','').replace('、','/').replace('?','').split(',')
        print("img/n", imgpoints)
        print("obj/n", objpoints)
        imgpoints.append([float(ls[0]), float(ls[1])])
        objpoints.append([float(ls[2]), float(ls[3]), float(ls[4])])
    return imgpoints, objpoints


def getM():
    # imgpoints = [[260.0, 170.0], [261.0, 207.0], [263.0, 245.0], [264.0, 284.0], [266.0, 322.0], [268.0, 360.0], [268.0, 399.0], [270.0, 436.0], [300.0, 169.0], [301.0, 207.0], [303.0, 245.0], [303.0, 283.0], [304.0, 321.0], [306.0, 359.0], [308.0, 397.0], [310.0, 435.0], [196.0, 278.0], [339.0, 205.0], [340.0, 243.0], [340.0, 282.0], [342.0, 320.0], [345.0, 358.0], [345.0, 396.0], [346.0, 434.0]]
    # objpoints =  [[90.0, -100.0], [110.0, -100.0], [130.0, -100.0], [150.0, -100.0], [170.0, -100.0], [190.0, -100.0], [210.0, -100.0], [230.0, -100.0], [90.0, -80.0], [110.0, -80.0], [130.0, -80.0], [150.0, -80.0], [170.0, -80.0], [190.0, -80.0], [210.0, -80.0], [230.0, -80.0], [90.0, -60.0], [110.0, -60.0], [130.0, -60.0], [150.0, -60.0], [170.0, -60.0], [190.0, -60.0], [210.0, -60.0], [230.0, -60.0]]
    
    # imgpoints, objpoints = get_data_txt(dir)
    objpoints = pickle.load(open("obj_points.pkl","rb"))
    imgpoints = pickle.load(open("img_points.pkl","rb"))
    
    imgpoints = np.array(imgpoints,dtype='float32')
    objpoints = np.array(objpoints,dtype='float32')
    # convert shape to satisfy cv2.estimateAffine2D's shape check
    if objpoints.ndim == 2:
        objpoints = objpoints[None,:,:2]
    if imgpoints.ndim == 2:
        imgpoints = imgpoints[None,:,:2]
    matrix_img2arm, _ = cv2.estimateAffine2D(imgpoints, objpoints,True)
    matrix_arm2img, _ = cv2.estimateAffine2D(objpoints, imgpoints,True)
    return matrix_img2arm, matrix_arm2img


def c2b(M, cam_points):                             # camera  to  board
    if isinstance(cam_points,tuple):
        np.array(cam_points)
    cam_points = np.reshape((1,2))
    assert cam_points.shape == (1,2)
    assert M.shape == (2,3)
    # R = M[:,:2]
    # T = M[:,2]
    cam_points = np.float32(cam_points)
    board_points = (M @ np.hstack((cam_points, np.ones((len(cam_points), 1)))).T).T
    return board_points

def b2c(Mn, board_points):                          # board  to  camera
    assert Mn.shape == (2,3)
    board_points = np.array(board_points, dtype='float32')
    board_points = np.expand_dims(board_points, axis=0)
    # Mn = np.linalg.inv(M) # 求逆
    cam_points = (Mn @ np.hstack((board_points, np.ones((len(board_points), 1)))).T).T
    return cam_points.tolist()


def autocali2():                                    # 用于机械臂的自动标定。每一次让机械臂选择一个位置，放置物块，保存当前的机械臂坐标。
                                                    # 然后机械臂回归原点，相机截取当前图片，二值化取蓝色区域，求蓝色区域质心作为当前的相机坐标系坐标。
    # init position                                 # ***** 常用标定方法 *******
    # box_pos = [-64.6159, 269.41, -39]            # 不遮挡相机拍摄的机械臂位置，每次结束放置后移动至该位置。
    objpoints = []
    imgpoints = []
    allpoints = []                          # allpoints用于存储适合放入神经网络的坐标。格式：[obj0,obj1,obj2,img0,img1]
    epoch = 10
    coords = gen_coords(epoch)                          # 随机生成放置的四个坐标list
    print("---------------init camera---------------")
    pipeline, align = initial_camera()

    print("-------------init the arm----------------")
    robot = arm_init()

    ################  开始标定   #################
    for i in range(epoch): 
        coord2, rz1= coords[i]                         #随机生成坐标和角度 [x,y,z]coord2  rz1
        
        arm_placement(robot, coord2, rz1)                               # 机械臂放置，并返回原位置，准备拍摄

        objpoints.append([coord2[0], coord2[1],OBJ_HEIGHT])
        print("objpoint [{}, {}] no.{}".format(coord2[0], coord2[1], i))
        color_image,_ = get_curr_image(pipeline, align)                                      # 记录图像
        cv2.imwrite("test.jpg", color_image)
        # get mask
        mask = _get_obj_mask(color_image)
        # get center
        center = get_mask_center(mask)                                                  # 求质心
        imgpoints.append(center)
        # allpoints.append([center[0], center[1], coord2[0], coord2[1], OBJ_HEIGHT])
        print("imgpoint {} no.{}".format(center, i))
        
        arm_suction(robot, coord2)                               # 机械臂把物体吸起，准备下一回合。
        # np.savetxt('data.txt', allpoints, delimiter=',')

    # np.savetxt('data_test.txt', allpoints, delimiter=',')
    pickle.dump(objpoints,open("obj_points.pkl","wb"))
    pickle.dump(imgpoints,open("img_points.pkl","wb"))
    print("points saved.")


def arm_suction_img(robot, M,img_coord):
    world_coord = c2b(M, img_coord)
    coord1 = np.array([world_coord[0],world_coord[1],OBJ_HEIGHT])
    arm_suction(robot, coord1, rz2=0)  


if __name__ == "__main__":
    # autocali2()
    getM()