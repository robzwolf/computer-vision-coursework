############################################################################
#
# This module contains the necessary logic to compute disparity
# for the given left and right images.
#
# Computer Vision Coursework, Software Systems and Applications III
# Author: vzbf32
#
# This work borrows heavily from the examples provided by Toby Breckon:
# - https://github.com/tobybreckon/stereo-disparity/blob/master/stereo_disparity.py
# - https://github.com/tobybreckon/stereo-disparity/blob/master/stereo_to_3d.py
# - https://github.com/tobybreckon/python-examples-cv/blob/master/yolo.py
# - https://github.com/tobybreckon/python-examples-cv/blob/master/surf_detection.py
# All retrieved 11th Dec 2019.
#
############################################################################

import cv2
import numpy as np

import helpers
from helpers import crop_top, crop_bottom


def preprocess_images(greyL, greyR):
    """
    Perform relevant preprocessing steps to the greyscale images
    before calculating the disparity.
    """
    greyL = preprocess_image_power(greyL)
    greyR = preprocess_image_power(greyR)

    greyL = preprocess_equalise_histogram(greyL)
    greyR = preprocess_equalise_histogram(greyR)

    return greyL, greyR


def preprocess_equalise_histogram(img):
    destination = np.copy(img)
    cv2.equalizeHist(img, destination)
    return destination


def preprocess_image_power(img, power=0.75):
    """
    Performing preprocessing by raising to the power subjectively appears to
    improve subsequent disparity calculation.
    @param img: The image to preprocess
    @param power: The power to raise each value to
    @return: The image after preprocessing
    """
    return np.power(img, power).astype('uint8')


def preprocess_image_crop_irrelevant_regions(img):
    """
    Crop out regions of the image where we shouldn't perform processing.
    """
    return img[crop_top:crop_bottom]
