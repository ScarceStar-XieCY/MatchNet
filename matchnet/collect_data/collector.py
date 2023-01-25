import os
import cv2
from suction_mask import get_obj_coord_as_poss
from overlap_detect import OverlapDetector
from tools.hardware.camera.cam_d435i import initial_camera, get_curr_image
from tools.hardware.arm.calibration import arm_suction_img,getM,gen_coords,b2c
from tools.hardware.arm.operation import arm_placement,arm_init
import random


KIT_NAME = ("rabbit","whale","bear","car")
KIT_IDX = 0
obj_num_init = KIT_IDX + 4
DUMP_PATH = "dataset0108_raw"


if __name__ == "__main__":
    dump_path = os.path.join(os.getcwd(),DUMP_PATH,KIT_NAME[KIT_IDX])
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    pipeline,align = initial_camera()    
    robot = arm_init()    
    compare_color, compare_depth = get_curr_image(pipeline,align)
    cv2.imwrite(os.path.join(dump_path,"compare_color.png"),compare_color)
    cv2.imwrite(os.path.join(dump_path,"compare_depth.png"),compare_depth)
    # compare_depth  =  cv2.imread(os.path.join("20230108_compare","train", f"depth0.png"), cv2.IMREAD_GRAYSCALE)
    kit_count = 0
    detector = OverlapDetector(compare_depth.shape) 
    
    matrix_img2arm, matrix_arm2img = getM(dir='data.txt')
    while True:
        # for one kit
        # suction stage
        # obj_num= obj_num_init - (i % obj_num_init)
        obj_num = obj_num_init
        all_obj_list = []
        obj_coord = []
        while obj_num > 0:
            # get image 
            i = (obj_num_init - obj_num) + obj_num_init * kit_count + 80
            print(i)
            color_image, depth_image = get_curr_image(pipeline,align)
            obj_coord = get_obj_coord_as_poss(compare_depth,depth_image, color_image, obj_num)
            all_obj_list.extend(obj_coord)
            while len(obj_coord) != 0:
                random_idx = random.randrange(0,len(obj_coord))
                selected_point = obj_coord.pop(random_idx)
                print("len(obj_coord) = ",len(obj_coord))
                # fake move
                obj_num -= 1
                count = 0
                arm_suction_img(robot,matrix_img2arm,selected_point)
                place_coord = gen_coords(method = "random",epoch = 1)
                place_img_coord = b2c(matrix_arm2img, place_coord)
                while detector.detect_overlap and count < 10:
                    place_coord = gen_coords(method = "random",epoch = 1)
                    place_img_coord = b2c(matrix_arm2img, place_coord)
                    count += 1
                if count >= 10:
                    print("can't find place.")
                    break
                arm_placement(robot,place_coord)
         
        # placement stage


        kit_count += 1