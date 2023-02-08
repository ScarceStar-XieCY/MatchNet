from tools.hardware.arm.HitbotInterface import HitbotInterface
import time
from pkgutil import get_data
import time
import random
from turtle import color
import cv2
import numpy as np
from torch import int32
from tools.hardware.arm.pump import pump_off
import logging
import pickle
# from walle.core import RotationMatrix
# from HitbotInterface import HitbotInterface
# from form2fit.code.utils import analyse_shape
# from form2fit.code.utils import get_center
# from form2fit.code.get_align_img import initial_camera,get_curr_image
logger = logging.getLogger(__file__)
box_pos = [-64.6159, 269.41, -39]
HEIGHT_OF_KIT_OBJ_WOODEN = -100
HEIGHT_OF_OBJ_WOODEN = -112
HEIGHT_OF_KIT_OBJ_CIRCLE = -100 #TODO
HEIGHT_OF_OBJ_CIRCLE = -100


class MyRobot():
    def __init__(self, robot_id=18, mode="wooden"):
        if mode == "wooden":
            self.height_of_kit_obj = HEIGHT_OF_KIT_OBJ_WOODEN
            self.height_of_obj = HEIGHT_OF_OBJ_WOODEN
        else:
            self.height_of_kit_obj = HEIGHT_OF_KIT_OBJ_CIRCLE
            self.height_of_obj = HEIGHT_OF_OBJ_CIRCLE
        self.mode = mode
        self._robot = self.arm_init(robot_id)
        self.arm_reset_pos()
        self._hand = 0
        self.get_matrix()
    
    def arm_init(self, robot_id):
        # robot_id = 18
        # box_pos = [-58.66, 309.92, -39]            # 不遮挡相机拍摄的机械臂位置，每次结束放置后移动至该位置。
        robot = HitbotInterface(robot_id)
        while robot.net_port_initial() != 1:
            logger.warning("Check port 40000.")
            time.sleep(0.5) 
        print("net and port initial successed")
        while not robot.is_connect():
            logger.warning("robot is not connected yet.")
            time.sleep(0.1)
            # ret = robot.is_connect()
            # print(ret)
        while robot.initial(3, 180) != 1:
            logger.warning("robot is not initialed yet.")
        while robot.unlock_position() != 1:
            logger.warning("robot is not unlocked yet.")
        logger.warning("Robot init successfully.")

        
        # a = robot.new_movej_xyz_lr(box_pos[0], box_pos[1], -34, 0, speed=120, roughly=0, lr=1)
        # robot.wait_stop()
        # print("robot statics is {}".format(a))
        # if a == 1: print("the robot is ready for the collection.")
        # time.sleep(0.5)
        return robot

    def arm_reset_pos(self):
        self.arm_try_move_update_hand(box_pos, rotation=0)
        self._robot.wait_stop()

    def arm_try_move_update_hand(self, coord3d, rotation):
        if self._hand != 0:
            hand = self._hand
        else:
            hand = -1 # left hand
        while True:
            a = self._robot.new_movej_xyz_lr(coord3d[0], coord3d[1], coord3d[2], rotation,140,0,hand)
            if a == 1:
                self._hand = hand
                break
            elif a == 4 or a == 7:
                hand = - hand
            elif a in [0,102,103]:
                continue
            else:
                raise RuntimeError(f"return status {a} when placement, stop.")

    def arm_to_coord(self, coord3d, rotation, place:bool=False):
        coord3d_over = (coord3d[0], coord3d[1], coord3d[2] + 40)
        self.arm_try_move_update_hand(coord3d_over, rotation)
        self.arm_try_move_update_hand(coord3d, rotation)
        self._robot.wait_stop()
        if place:
            pump_off()                                                                          # 机械臂松手
            

    def get_matrix(self):
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
        self.matrix_img2arm, _ = cv2.estimateAffine2D(imgpoints, objpoints,True)
        self.matrix_arm2img, _ = cv2.estimateAffine2D(objpoints, imgpoints,True)
        assert self.matrix_img2arm.shape == (2,3)
        assert self.matrix_arm2img.shape == (2,3)

    def c2b(self, cam_points):                             # camera  to  board
        if isinstance(cam_points,tuple) or isinstance(cam_points,list):
            np.array(cam_points)
        cam_points = np.reshape((-1,2))
        # R = M[:,:2]
        # T = M[:,2]
        cam_points = np.float32(cam_points)
        board_points = (self.matrix_img2arm @ np.hstack((cam_points, np.ones((len(cam_points), 1)))).T).T
        return board_points

    def b2c(self, board_points):                          # board  to  camera
        board_points = np.array(board_points, dtype='float32')
        board_points = np.expand_dims(board_points, axis=0)
        # matrix_arm2img = np.linalg.inv(M) # 求逆
        cam_points = (self.matrix_arm2img @ np.hstack((board_points, np.ones((len(board_points), 1)))).T).T
        return cam_points.tolist()

    def arm_to_coord2d(self,coord2d,rotation, place:bool=False,to_kit:bool=False):
        world_coord = self.c2b(coord2d)
        if to_kit:
            coord3d = np.array([world_coord[0],world_coord[1],self.height_of_kit_obj])
        else:
            coord3d = np.array([world_coord[0],world_coord[1],self.height_of_obj])
        self.arm_to_coord(coord3d, rotation,place)
    
    

# def arm_placement_each_step(robot, coord2, rz1=0, hand=-1):   # 机械臂的放置指令。hand为-1，则为物体部分。1为盒子部分, 旋转角度为rz1
#     # box_pos = [-64.6159, 269.41, -39]            # 不遮挡相机拍摄的机械臂位置，每次结束放置后移动至该位置。
#     time.sleep(0.5) 
#     a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2] + 40, 0,140,0,hand)    # 机械臂悬停在指定坐标正上方
#     robot.wait_stop()
#     print("ready to place: coord value {}, speed 140\n".format(coord2))
#     if a == 1: print("moving") 
#     else: 
#         print("error, code is {}".format(a))
#         raise RuntimeError("arm cannot move to the location {}".format(coord2))
#     time.sleep(0.25)
#     a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2] + 40, rz1,140,0,hand)    # 机械臂旋转
#     robot.wait_stop()
#     time.sleep(0.25)
#     print("::placing, coord value {}, speed 100\n".format(coord2))
#     a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2]-5 , rz1,100,0,hand)       # 机械臂放置.z轴如果空中放置则-1，多线程放置则-5
#     robot.wait_stop()
#     if a == 1: print("moving") 
#     else: print("error, code is {}".format(a))
#     time.sleep(0.5)
#     pump_off()                                                                          # 机械臂松手
#     time.sleep(0.5)

#     print("::send_coords, coord value {}, speed 140\n".format(coord2))
#     a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2] + 40, rz1,140,0,hand)      # 机械臂抬起
#     robot.wait_stop()
#     if a == 1: print("moving") 
#     else: print("error, code is {}".format(a))
#     time.sleep(0.5)
    
#     a = robot.new_movej_xyz_lr(box_pos[0], box_pos[1], -34, 0,120,0,1)                   # 机械臂回原位置
#     robot.wait_stop()    

# def arm_suction(robot, coord2, rz2=0):              # 机械臂的吸取指令。
#     a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2]+40, rz2,120,0,-1)      # 机械臂准备吸取  (盒子这边是1，物块那一边是-1)
#     robot.wait_stop()
#     if a == 1: print("moving") 
#     else: print("error, code is {}".format(a))
#     time.sleep(0.5)  

#     a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2]-5, rz2, 90, 0,-1)         # 机械臂吸取
#     robot.wait_stop()
#     if a == 1: 
#         print("moving") 
#         print("::send_coords, coord value {}, speed 90\n".format(coord2))
#     else: print("error, code is {}".format(a)) 
#     time.sleep(0.5)  
#     # coord1up = coordup(coord1)
#     a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2] + 40, rz2, 110,0,-1)       # 机械臂抬起
#     robot.wait_stop()
#     if a == 1: print("moving") 
#     else: print("error, code is {}".format(a))







                
    