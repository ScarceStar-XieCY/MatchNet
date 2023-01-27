"""A set of commonly used utilities.
"""

import datetime
import os

import numpy as np
import skimage.io as io
import cv2
import os
from PIL import Image


def makedir(dirname):
    """Safely creates a new directory.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def gen_timestamp():
    """Generates a timestamp in YYYY-MM-DD-hh-mm-ss format.
    """
    date = str(datetime.datetime.now()).split(".")[0]
    return date.split(" ")[0] + "-" + "-".join(date.split(" ")[1].split(":"))


def colorsave(filename, x):
    """Saves an rgb image as a 24 bit PNG.
    """
    io.imsave(filename, x, check_contrast=False)


def depthsave(filename, x):
    """Saves a depth image as a 16 bit PNG.
    """
    io.imsave(filename, (x * 1000).astype("uint16"), check_contrast=False)


def colorload(folder,init, use_color):
    """Loads an rgb image as a numpy array.
    """
    if init:
        time_prefix = "init_"
    else:
        time_prefix = "final_"
    if use_color:
                color_name = "color.png"
    else:
        color_name = "gray.png"

    filename = os.path.join(folder, time_prefix + color_name)
    return cv2.imread(filename,cv2.IMREAD_UNCHANGED)


def depthload(folder, init):
    """Loads a depth image as a numpy array.
    """
    if init:
        time_prefix = "init_"
    else:
        time_prefix = "final_"
    filename = os.path.join(folder, time_prefix + "depth.png")
    x = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
    # x = (x * 1e-3).astype("float32")
    return x
