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


def preprocess_image_for_disparity(img):
    """
    Perform relevant pre-processing steps to the image to improve the disparity map.
    """

    # Raise each pixel to a power.
    img = preprocess_image_power(img)

    # Equalise histogram for image
    img = preprocess_equalise_histogram(img)

    return img


def preprocess_image_for_yolo(img):
    """
    Perform relevant pre-processing steps to the image to improve object detection and classification.
    """

    # Raise each pixel to a power.
    # img = preprocess_image_power(img)

    # Use CLAHE
    img = preprocess_clahe(img)

    return img


def preprocess_equalise_histogram(img):
    """
    Equalise the histogram of the image, to increase contrast and
    use the full range of available values.
    """
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


def preprocess_clahe(img):
    """
    Use CLAHE (contrast limited adaptive histogram equalisation) to equalise histogram in
    specific regions of the image.
    We only apply CLAHE to the Lightness of the image.
    Uses: https://stackoverflow.com/a/47370615/2176546
    """

    clahe = cv2.createCLAHE(2.0, tileGridSize=(8,8))
    
    # Convert the image to LAB colour space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    
    # Apply CLAHE to the Lightness channel of the LAB image
    lab_planes[0] = clahe.apply(lab_planes[0])
    
    # Convert from LAB back to RGB
    lab = cv2.merge(lab_planes)
    clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return clahe_img


def preprocess_image_crop_irrelevant_regions(img):
    """
    Crop out regions of the image where we shouldn't perform processing.
    """
    return img[helpers.crop_top:helpers.crop_bottom]
