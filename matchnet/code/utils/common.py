"""A set of commonly used utilities.
"""

import datetime
import os

import numpy as np
import skimage.io as io
import cv2
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


def colorload(filename):
    """Loads an rgb image as a numpy array.
    """
    return cv2.imread(filename,cv2.IMREAD_UNCHANGED)


def depthload(filename):
    """Loads a depth image as a numpy array.
    """
    x = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
    # x = (x * 1e-3).astype("float32")
    return x
