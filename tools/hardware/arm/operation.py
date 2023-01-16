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
# from walle.core import RotationMatrix
# from HitbotInterface import HitbotInterface
# from form2fit.code.utils import analyse_shape
# from form2fit.code.utils import get_center
# from form2fit.code.get_align_img import initial_camera,get_curr_image
box_pos = [-64.6159, 269.41, -39] 

def arm_placement(robot, coord2, rz1=0, hand=-1):   # 机械臂的放置指令。hand为-1，则为物体部分。1为盒子部分, 旋转角度为rz1
    # box_pos = [-64.6159, 269.41, -39]            # 不遮挡相机拍摄的机械臂位置，每次结束放置后移动至该位置。
    time.sleep(0.5) 
    a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2] + 40, 0,140,0,hand)    # 机械臂准备放置
    robot.wait_stop()
    print("ready to place: coord value {}, speed 140\n".format(coord2))
    if a == 1: print("moving") 
    else: 
        print("error, code is {}".format(a))
        raise RuntimeError("arm cannot move to the location {}".format(coord2))
    time.sleep(0.25)
    a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2] + 40, rz1,140,0,hand)    # 机械臂旋转
    robot.wait_stop()
    time.sleep(0.25)
    print("::placing, coord value {}, speed 100\n".format(coord2))
    a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2]-5 , rz1,100,0,hand)       # 机械臂放置.z轴如果空中放置则-1，多线程放置则-5
    robot.wait_stop()
    if a == 1: print("moving") 
    else: print("error, code is {}".format(a))
    time.sleep(0.5)
    pump_off()                                                                          # 机械臂松手
    time.sleep(0.5)

    print("::send_coords, coord value {}, speed 140\n".format(coord2))
    a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2] + 40, rz1,140,0,hand)      # 机械臂抬起
    robot.wait_stop()
    if a == 1: print("moving") 
    else: print("error, code is {}".format(a))
    time.sleep(0.5)
    
    a = robot.new_movej_xyz_lr(box_pos[0], box_pos[1], -34, 0,120,0,1)                   # 机械臂回原位置
    robot.wait_stop()    

def arm_suction(robot, coord2, rz2=0):              # 机械臂的吸取指令。
    a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2]+40, rz2,120,0,-1)      # 机械臂准备吸取  (盒子这边是1，物块那一边是-1)
    robot.wait_stop()
    if a == 1: print("moving") 
    else: print("error, code is {}".format(a))
    time.sleep(0.5)  

    a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2]-5, rz2, 90, 0,-1)         # 机械臂吸取
    robot.wait_stop()
    if a == 1: 
        print("moving") 
        print("::send_coords, coord value {}, speed 90\n".format(coord2))
    else: print("error, code is {}".format(a)) 
    time.sleep(0.5)  
    # coord1up = coordup(coord1)
    a = robot.new_movej_xyz_lr(coord2[0], coord2[1], coord2[2] + 40, rz2, 110,0,-1)       # 机械臂抬起
    robot.wait_stop()
    if a == 1: print("moving") 
    else: print("error, code is {}".format(a))



def arm_init():
    robot_id = 18
    # box_pos = [-58.66, 309.92, -39]            # 不遮挡相机拍摄的机械臂位置，每次结束放置后移动至该位置。
    robot = HitbotInterface(robot_id)
    robot.net_port_initial()
    time.sleep(0.5)
    print("initial successed")
    ret = robot.is_connect()
    while ret != 1:
        time.sleep(0.1)
        ret = robot.is_connect()
        print(ret)
    ret = robot.initial(3, 180)
    if ret == 1:
        print("robot initial successful")
        robot.unlock_position()
    else:
        print("robot initial failed")
    if robot.unlock_position():
        print("------unlock------")
    time.sleep(0.5)

    if robot.is_connect():
        print("robot online")
        a = robot.new_movej_xyz_lr(box_pos[0], box_pos[1], -34, 0, speed=120, roughly=0, lr=1)
        robot.wait_stop()
        print("robot statics is {}".format(a))
        if a == 1: print("the robot is ready for the collection.")
        time.sleep(0.5)
    return robot



                
    