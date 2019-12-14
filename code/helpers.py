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


def bounding_box_centre(a, b, pixels):
    """
    Extract horizontal/vertical centre co-ordinates.
    """
    return math.ceil( (a+b) // 2 - pixels ), math.ceil( (a+b) // 2 + pixels )


def get_mean_pixels(top, bottom, left, right, threshold=0.3):
    """
    Get {threshold}*100 % of the box in the middle
    """
    height = (top - bottom)
    width = (left - right)
    return (height * width * threshold) / 100


def get_formatted_median(Z):
    if Z and statistics.median(Z) < 45:
        return f'{str(round(statistics.median(Z), 2))} m'
    else:
        return ''
