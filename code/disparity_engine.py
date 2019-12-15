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


def convert_to_greyscale(imgL, imgR):
    """
    Convert images from 3-channel RGB to greyscale.
    """
    greyL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    greyR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    return greyL, greyR


def preprocess_image_power(img, power=0.75):
    """
    Performing preprocessing by raising to the power subjectively appears to
    improve subsequent disparity calculation.
    @param img: The image to preprocess
    @param power: The power to raise each value to
    @return: The image after preprocessing
    """
    return np.power(img, power).astype('uint8')


def preprocess_images(greyL, greyR):
    """
    Perform relevant preprocessing steps to the greyscale images
    before calculating the disparity.
    """
    greyL = preprocess_image_power(greyL)
    greyR = preprocess_image_power(greyR)

    greyL = helpers.preprocess_image_crop_irrelevant_regions(greyL)
    greyR = helpers.preprocess_image_crop_irrelevant_regions(greyR)

    return greyL, greyR


def compute_disparity(greyL, greyR, max_disparity):
    """
    Set up the disparity stereo processor and compute the disparity image.
    """

    # Set up the disparity stereo processor to find a maximum of max_disparity values
    # This uses a modified H. Hisrchmuller algorithm [Hirschmuller, 2008] that differs
    # (see openCV manual). Parameters can be adjusted, these ones are from [Hamilton /
    # Breckon et al. 2013]
    block_size = 21
    stereo_processor = cv2.StereoSGBM_create(0, max_disparity, block_size)

    # Compute the disparity image from the undistorted, rectified stereo images
    # that we've loaded. For reasons best known to the OpenCV developers, the
    # disparity is returned scaled by 16.
    return stereo_processor.compute(greyL, greyR)


def filter_speckles(disparity, max_disparity):
    """
    Filter out noise and speckles.
    """
    disparity_noise_filter = 5
    max_speckle_size = 4000

    cv2.filterSpeckles(disparity, 0, max_speckle_size, max_disparity - disparity_noise_filter)


def scale_disparity_to_8_bit(disparity, max_disparity):
    """
    Scales the disparity to 8-bit for viewing.
    """

    # Divide by 16 and convert to 8-bit image. Then, the range of values should
    # be between 0 and max_disparity, but in fact it's -1 to max_disparity - 1).
    # To fix this, we use an initial threshold between 0 and max_disparity, because
    # disparity=-1 means no disparity is available.
    _, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)

    return (disparity / 16.0).astype(np.uint8)


def crop_disparity_map(disparity_scaled):
    """
    Crop disparity to crop out the left part where is no disparity, as this area
    is not seen by both cameras.
    """
    return disparity_scaled[:,135:]


def display_disparity_window(disparity_scaled):
    """
    Display the image. Scale it to the full 0 to 255 range based on the
    number of disparities in use for the stereo part.
    """
    disparity_to_display = np.copy(disparity_scaled)

    # Normalise the image so that we use the full range of 0 to 255
    cv2.normalize(disparity_scaled, disparity_to_display, 0, 255, cv2.NORM_MINMAX)

    cv2.imshow('Disparity', disparity_to_display)
