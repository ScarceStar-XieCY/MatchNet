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


