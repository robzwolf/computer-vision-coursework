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
import preprocessor

######################
# Important Constants
######################

# Maximum allowable disparity
# Must be a multiple of 16
max_disparity = 160


def compute_disparity(greyL, greyR):
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


def filter_speckles(disparity):
    """
    Filter out noise and speckles.
    """
    disparity_noise_filter = 5
    max_speckle_size = 4000

    cv2.filterSpeckles(disparity, 0, max_speckle_size, max_disparity - disparity_noise_filter)


def scale_disparity_to_8_bit(disparity):
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
    return disparity_scaled[:,160:]


def display_disparity_window(disparity_scaled):
    """
    Display the image. Scale it to the full 0 to 255 range based on the
    number of disparities in use for the stereo part.
    """
    disparity_to_display = np.copy(disparity_scaled)

    # Normalise the image so that we use the full range of 0 to 255
    cv2.normalize(disparity_scaled, disparity_to_display, 0, 255, cv2.NORM_MINMAX)

    cv2.imshow('Disparity', disparity_to_display)


def bilateral_filter(disparity_scaled):
    """
    Apply bilateral filtering to the scaled disparity map.
    It removes noise while keeping edges sharp.
    """
    sigma = 5
    return cv2.bilateralFilter(disparity_scaled, 5, sigma, sigma)


def get_disparity(imgL, imgR):
    """
    Produce a disparity map for the two images.
    Take a left image and a right image, and using the methods in disparity_engine,
    and the stereo processor in OpenCV (specifically, a modified version of the H.
    Hirschmuller algorithm [Hirschmuller, 2008]), calculate a disparity map and
    return it.
    @param imgL: The left image
    @param imgR: The right image
    @return: The disparity map
    """

    # Disparity matching works on greyscale, so start by converting to greyscale.
    # Do this for both images as they're both given as 3-channel RGB images.
    greyL, greyR = helpers.convert_to_greyscale(imgL, imgR)

    # Perform pre-processing
    greyL = preprocessor.preprocess_image_for_disparity(greyL)
    greyR = preprocessor.preprocess_image_for_disparity(greyR)

    # Set up the disparity stereo processor and compute the disparity.
    disparity = compute_disparity(greyL, greyR)

    # Filter out noise and speckles
    filter_speckles(disparity)

    # Scale the disparity to 8-bit for viewing as an image.
    disparity_scaled = scale_disparity_to_8_bit(disparity)

    # Apply bilateral filtereing
    disparity_scaled = bilateral_filter(disparity_scaled)

    # Crop out the left side where is no disparity
    disparity_scaled = crop_disparity_map(disparity_scaled)

    return disparity_scaled
