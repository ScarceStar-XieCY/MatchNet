# coding=utf-8
#used to show the result of SuctionNet
import os
import sys 
import cv2
import glob
import imutils

import torch
import torch.nn 
import numpy as np

from matchnet import config
from matchnet.code.ml.dataloader import suction, suction_infer
from matchnet.code.ml.models import SuctionNet
from matchnet.code.ml.dataloader import get_corr_loader

from get_align_img import initial_camera
import argparse
import warnings
warnings.filterwarnings("ignore")
sys.path.append('/usr/local/lib/python3.6/pyrealsense2')


def get_curr_image(pipeline,align):

    decimation_scale = 2
    wait_frame_count = 30
    #decimation, spatial, temporal, hole_filling, depth_to_disparity, disparity_to_depth = filters_config(decimation_scale)
    # colorizer = rs.colorizer(color_scheme=3)
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 424x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        #guaratee that depth frames are useable
        aligned_depth_data = np.asanyarray(aligned_depth_frame.get_data()).astype('uint8')
        # if Counter(aligned_depth_data.ravel())[0] > 0.2 * 848 * 480: 
        #     continue

        #再等待30帧，图片亮度会有提升
        if wait_frame_count > 0:
            wait_frame_count = wait_frame_count - 1 
            continue
        
        #processed_frame = depth_processing(, decimation, spatial, temporal, hole_filling, depth_to_disparity, disparity_to_depth)
        processed_depth = aligned_depth_frame.as_depth_frame()
        depth_image = np.asanyarray(processed_depth.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        depth_image = depth_image.astype('uint8')

        # if depth_image.ndim == 3:
        #     if depth_image.shape[2]==3:
        #         depth_image = cv2.cvtColor(depth_image,cv2.COLOR_BGR2GRAY)
        #     #depth_image = np.squeeze(depth_image,axis=2)

        if color_image.ndim == 3:
            if color_image.shape[2]==3:
                color_image = color_image[:,:,0]
                #color_image = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
        #     #color_image = np.squeeze(color_image,axis=2)

        # assert depth_image.shape == (480,848)
        # assert color_image.shape == (480,848)
        # H,W = depth_image.shape 
        # cv2.imshow('depth',depth_image)
        # cv2.imshow('depth',color_image)
        print('color.shape',color_image.shape)
        print('depth.shape',depth_image.shape)
        cv2.imwrite('depth_image.png',depth_image)
        cv2.imwrite('color_image.png',color_image)
        # cv2.waitKey()
        break

    return color_image,depth_image

def findcoord(img, threshold=185):
    #_ , thresh1 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)             # 根据阈值调整范围
    thresh = cv2.GaussianBlur(img, (55, 51), 0)                                 # 高斯模糊，去除周边杂物
    _ , thresh2 = cv2.threshold(thresh, threshold, 255, cv2.THRESH_BINARY) 

    cnts = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] 

    for c in cnts:                                                                  
        M = cv2.moments(c)                                                          # 获取中心点
        if M["m00"] == 0:
            break
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
                                                                                     
        #cv2.drawContours(img, [c], -1, (0, 255, 0), 2)                             # 画出轮廓
        cv2.circle(img, (cX, cY), 7, (0, 0, 0), -1)                                 # 画出中点
        cv2.putText(img, "center", (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imwrite('out.jpg', img)

# python form2fit/code/infer_suction.py --weights form2fit/code/ml/savedmodel/epoch180.pth
if __name__ == "__main__":
    def str2bool(s):
        return s.lower() in ["1", "true"]
    parser = argparse.ArgumentParser(description="Descriptor Network Visualizer")
    parser.add_argument("--modelname", default="black-floss", type=str)
    parser.add_argument("--batchsize", type=int, default=4, help="The batchsize of the dataset.")
    parser.add_argument("--official", type=int, default=0, help="Whether to use the official dataloader to test the result.")
    parser.add_argument("--dtype", type=str, default="valid")
    parser.add_argument("--num_desc", type=int, default=64)
    parser.add_argument("--num_channels", type=int, default=2)
    parser.add_argument("--background_subtract", action='store_true', help="(bool) Whether to apply background subtract.")
    parser.add_argument("--augment", type=str2bool, default=False)
    parser.add_argument("--root", type=str, default="", help="the path of project")
    parser.add_argument("--weights", type=str, default="form2fit/code/ml/savedmodel/epoch80.pth", help="the path of dataset")
    opt = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    #inference
    print("--------------getting img----------------")
    
    
    pipeline, align = initial_camera()
    c_height,d_height = get_curr_image(pipeline, align)
    root =  os.path.join(config.ml_data_dir, opt.root, "infer","0")
    cv2.imwrite(os.path.join(root,"final_color_height.png"),c_height)
    cv2.imwrite(os.path.join(root,"final_depth_height.png"),d_height)
    findcoord(c_height, threshold=195)


        

