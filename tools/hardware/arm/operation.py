import sys
import os
sys.path.append(os.getcwd())

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
RESET_COORD = (-104,277,-90)
HEIGHT_OF_KIT_OBJ_WOODEN = -110
HEIGHT_OF_OBJ_WOODEN = -117
HEIGHT_OF_KIT_OBJ_CIRCLE = -75 #TODO
HEIGHT_OF_OBJ_CIRCLE = -98

CALI_ROOT = os.path.join("tools","hardware","arm")
if not os.path.exists(CALI_ROOT):
    os.makedirs(CALI_ROOT)

class MyRobot():
    def __init__(self, robot_id=18, mode="wooden",need_cali=True, debug_mode=False):
        if mode == "wooden":
            self.height_of_kit_obj = HEIGHT_OF_KIT_OBJ_WOODEN
            self.height_of_obj = HEIGHT_OF_OBJ_WOODEN
        else:
            self.height_of_kit_obj = HEIGHT_OF_KIT_OBJ_CIRCLE
            self.height_of_obj = HEIGHT_OF_OBJ_CIRCLE
        self.mode = mode
        if not debug_mode:
            self._robot = self.arm_init(robot_id)
            self.arm_reset_pos()
        if not need_cali:
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
        self.arm_try_move(RESET_COORD, rotation=0)
        self._robot.wait_stop()

    def arm_try_move(self, coord3d, rotation, velocity=100, do_move:bool=True):
        switch_hand = False
        hand = -1 # left hand
        while True:
            a = self._robot.new_movej_xyz_lr(coord3d[0], coord3d[1], coord3d[2], rotation,velocity,1,hand)
            if a == 1:
                if do_move:
                    break
                self._robot.new_stop_move()
                break
            elif a == 4 or a == 7 and not switch_hand:
                hand = - hand
                switch_hand = True
            elif a in [0,102,103]:
                continue
            else:
                raise RuntimeError(f"return status {a} when placement, stop.")

    def arm_to_coord_updown(self, coord3d, rotation, place:bool=False):
        coord3d_over = (coord3d[0], coord3d[1], coord3d[2] + 40)
        try:
            self.arm_try_move(coord3d_over, rotation, 70)
        except:
            return False
        self.arm_try_move(coord3d, rotation, 70)
        self._robot.wait_stop()
        if place:
            pump_off()                                                           # 机械臂松手
        self.arm_try_move(coord3d_over, rotation, 70)
        self._robot.wait_stop()
        return True
            
    def arm_to_coord(self, coord3d, rotation, place:bool=False):
        try:
            self.arm_try_move(coord3d, rotation, 120)
        except:
            return False
        self._robot.wait_stop()
        if place:
            pump_off()                                                            # 机械臂松手
        return True

    def get_matrix(self):
        # imgpoints, objpoints = get_data_txt(dir)
        objpoints = pickle.load(open(os.path.join(CALI_ROOT,"obj_points.pkl"),"rb"))
        imgpoints = pickle.load(open(os.path.join(CALI_ROOT,"img_points.pkl"),"rb"))
        
        imgpoints = np.array(imgpoints,dtype='float32')
        objpoints = np.array(objpoints,dtype='float32')
        assert len(imgpoints) == len(objpoints)
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
        cam_points = np.reshape(cam_points,(-1,2))
        # R = M[:,:2]
        # T = M[:,2]
        cam_points = np.float32(cam_points)
        board_points = (self.matrix_img2arm @ np.hstack((cam_points, np.ones((len(cam_points), 1)))).T).T
        return board_points

    def b2c(self, board_points):                          # board  to  camera
        board_points = np.array(board_points, dtype='float32')
        if board_points.ndim == 1:
            board_points = np.expand_dims(board_points, axis=0)
        assert board_points.ndim == 2
        board_points = board_points[:,:2]
        # matrix_arm2img = np.linalg.inv(M) # 求逆
        cam_points = (self.matrix_arm2img @ np.hstack((board_points, np.ones((len(board_points), 1)))).T).T
        return cam_points.tolist()

    def arm_to_coord2d(self,coord2d,rotation, place:bool=False,to_kit:bool=False):
        world_coord = self.c2b(coord2d)
        world_coord = world_coord.squeeze()
        if to_kit:
            coord3d = np.array([world_coord[0],world_coord[1],self.height_of_kit_obj])
        else:
            coord3d = np.array([world_coord[0],world_coord[1],self.height_of_obj])
        return self.arm_to_coord_updown(coord3d, rotation,place)
    


if __name__ == "__main__":
    robot = MyRobot(need_cali=False)

                
    