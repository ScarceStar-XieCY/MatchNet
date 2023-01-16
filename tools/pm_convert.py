"""Convert points and mask."""
# coding=UTF-8
import numpy as np
import cv2

def mask2coord(mask):
    """Get uv coord for mask."""
    coord = np.column_stack(np.where(mask))
    return coord


def coord2mask(coord,h,w,visual):
    mask_layer = np.zeros((h,w))
    mask_layer[coord[:, 0],coord[:, 1]] = 1
    if visual:
        cv2.imshow('mask from coord', mask_layer)
    return mask_layer