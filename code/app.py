############################################################################
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

import os
import cv2

############################################################################
# LOCATION OF DATA SET
############################################################################

# Change this to specify the location of the master dataset
master_path_to_dataset = '../TTBB-durham-02-10-17-sub10'

# Name of the left images within the master dataset directory
directory_to_cycle_left = 'left-images'

# Name of the right images within the master dataset directory
directory_to_cycle_right = 'right-images'

# Left images' filenames end in this suffix
filename_left_suffix = '_L.png'

# Right images' filenames end in this suffix
filename_right_suffix = '_R.png'

# Set a timestamp to start from, or leave it blank to start from the beginning.
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns
skip_forward_file_pattern = ''

##########################################
# Constants
##########################################

# Focal length in pixels
camera_focal_length_px = 399.9745178222656

# Focal length in metres (4.8 mm)
camera_focal_length_m = 4.8 / 1000

# Camera baseline in metres
stereo_camera_baseline_m = 0.2090607502

# Confidence threshold
confidence_threshold = 0.5

# Non-maximum supression threshold
nonmaximum_suspression_threshold = 0.4

# Width of network's input image
input_width = 416

# Height of network's input image
input_height = 416

# Maximum allowable disparity
max_disparity = 128

##########################################
# Playback Controls
##########################################

# Display full or cropped disparity image
crop_disparity = False

# Pause until key press after each image
pause_playback = False

##########################################
# Initialise variables
##########################################
classes_file =  'coco.names'
config_file = 'yolov3.cfg'
weights_file = 'yolov3.weights'

# Read classes into a list like ['person', 'bicycle', ...]
with open(classes_file, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Set up window
window_name = "Object Detection (YOLOv3) using '{}'".format(weights_file)
print(window_name)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Resolve full directory location of data set for left / right images
full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right)

# Get a list of the left image files and sort them (by timestamp in filename)
left_file_list = sorted(os.listdir(full_path_directory_left))

# Set up the disparity stereo processor to find a maximum of max_disparity values
# This uses a modified H. Hisrchmuller algorithm [Hirschmuller, 2008] that differs
# (see openCV manual). Parameters can be adjusted, these ones are from [Hamilton /
# Breckon et al. 2013]
stereo_processor = cv2.StereoSGBM_create(0, max_disparity, 21)

##########################################
# Loop through each image pair
##########################################
for filename_left in left_file_list:

    # Skip forward to start a file that was specified by skip_forward_file_pattern,
    # if this is set.
    if skip_forward_file_pattern and skip_forward_file_pattern not in filename_left:
        continue
    elif skip_forward_file_pattern and skip_forward_file_pattern in filename_left:
        skip_forward_file_pattern = ''

    # From the left image, get the corresponding right image
    filename_right = filename_left.replace(filename_left_suffix, filename_right_suffix)
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left)
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right)

    # For sanity, print out these filenames.
    print('#############################################################################')
    print()
    print('Processing:')
    print('    ' + full_path_filename_left)
    print('    ' + full_path_filename_right)
    print()

    # Check the left file is a PNG image
    # And check that a corresponding right image actually exists

    if '.png' in filename_left and os.path.isfile(full_path_filename_right):

        # Read left and right images and display them in windows
        # N.B. Both are stored as 3-channel (even though one is is greyscale)
        # RGB images, so we should load both as such.

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        cv2.imshow('Left Image', imgL)

        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
        cv2.imshow('Right image', imgR)

        print('-- Files loaded successfully!')
        print()
