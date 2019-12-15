############################################################################
#
# This module contains the YOLO logic for object detection and classification.
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

import math
import statistics
import random


######################
# Important Constants
######################

# Crop out the sky
crop_top = 80

# Crop out the bonnet of the car
crop_bottom = 390


def bounding_box_centre(a, b, pixels):
    """
    Extract horizontal/vertical centre co-ordinates.
    """
    return math.ceil( (a+b) // 2 - pixels ), math.ceil( (a+b) // 2 + pixels )


def get_mean_pixels(top, bottom, left, right, threshold=0.4):
    """
    Get {threshold}*100 % of the box in the middle
    """
    height = (top - bottom)
    width = (left - right)
    return (height * width * threshold) / 100


def get_formatted_median(Z):
    """
    Convert the distance Z into a human-readable string
    @param Z:
    @return:
    """
    # Only display the distance if it is less than max_distance, to avoid giving inaccurate measurements for
    # objects that are a long way from the camera.
    max_distance = 45
    if Z and statistics.median(Z) < max_distance:
        return f'{round(statistics.median(Z), 2)} m'
    else:
        return ''


def random_number_between(min, max, integer=True):
    """
    Randomly generate a number between min and max, inclusive.
    @param min: Lower bound (inclusive)
    @param max: Upper bound (inclusive)
    @param integer: Whether the returned number should be forced to be an integer or not
    """
    value = random.random()
    random_number = min + (value * (max - min + 1))
    if integer:
        return int(random_number)
    else:
        return random_number


def random_colour():
    """
    Generate a random colour.
    @return: A tuple of (blue, green, red)
    """
    darkest = 50
    lightest = 255
    red = random_number_between(darkest, lightest)
    green = random_number_between(darkest, lightest)
    blue = random_number_between(darkest, lightest)
    return blue, green, red


def preprocess_image_crop_irrelevant_regions(img):
    """
    Crop out regions of the image where we shouldn't perform processing.
    """
    return img[crop_top:crop_bottom]
