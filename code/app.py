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

import disparity_engine
import os
import cv2
import numpy as np

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
# skip_forward_file_pattern = '1506943946.380279' # Start at final frame
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

# Non-maximum suppression threshold
nonmaximum_suppression_threshold = 0.4

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
classes_file = 'coco.names'
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


def show_disparity(imgL, imgR):

    # Disparity matching works on greyscale, so start by converting to greyscale.
    # Do this for both images as they're both given as 3-channel RGB images.
    greyL, greyR = disparity_engine.convert_to_greyscale(imgL, imgR)

    # Perform preprocessing
    greyL, greyR = disparity_engine.preprocess_images(greyL, greyR)

    # Set up the disparity stereo processor and compute the disparity.
    disparity = disparity_engine.compute_disparity(greyL, greyR, max_disparity)

    # Filter out noise and speckles
    disparity_engine.filter_speckles(disparity, max_disparity)

    # Scale the disparity to 8-bit for viewing as an image.
    disparity_scaled = disparity_engine.scale_disparity_to_8_bit(disparity, max_disparity)

    # If user wants to crop the disparity, then crop out the left side and the car bonnet.
    if crop_disparity:
        disparity_scaled = disparity_engine.crop_disparity_map(disparity_scaled)

    # Display the image
    disparity_engine.display_disparity_window(disparity_scaled, max_disparity)


def get_right_filename_from_left(filename_left):
    """
    @param filename_left: Filename for the left image, e.g. 'some_image_L.png'
    @return: Filename for the right image, based on the suffixes defined in the constants above
    """
    return filename_left.replace(filename_left_suffix, filename_right_suffix)


def get_full_path_filename_left(filename_left):
    """
    @param filename_left: Filename for the left image, e.g. 'some_image_L.png'
    @return: The fully qualified path for the left image
    """
    return os.path.join(full_path_directory_left, filename_left)


def get_full_path_filename_right(filename_right):
    """
    @param filename_right: Filename for the right image, e.g. 'some_image_R.png'
    @return: The fully qualified path for hte right image
    """
    return os.path.join(full_path_directory_right, filename_right)


def files_exist_and_are_png(full_path_filename_left, full_path_filename_right):
    """
    @param full_path_filename_left: Fully qualified path to the left image
    @param full_path_filename_right: Fully qualified path to the right image
    @return: Whether the files exist and are PNG files.
    """
    if not full_path_filename_left.endswith('.png'):
        return False

    if not full_path_filename_right.endswith('.png'):
        return False

    if not os.path.isfile(full_path_filename_left):
        return False

    if not os.path.isfile(full_path_filename_right):
        return False

    return True


def loop_through_files():
    global skip_forward_file_pattern, crop_disparity, pause_playback

    for filename_left in left_file_list:

        # Skip forward to start a file that was specified by skip_forward_file_pattern,
        # if this is set.
        if skip_forward_file_pattern and skip_forward_file_pattern not in filename_left:
            continue
        elif skip_forward_file_pattern and skip_forward_file_pattern in filename_left:
            skip_forward_file_pattern = ''

        # From the left image, get the corresponding right image
        filename_right = get_right_filename_from_left(filename_left)

        # Get the fully qualified paths to the left and right image files
        full_path_filename_left = get_full_path_filename_left(filename_left)
        full_path_filename_right = get_full_path_filename_right(filename_right)

        # For sanity, print out these filenames.
        print('#############################################################################')
        print()
        print('Processing:')
        print(f'    {full_path_filename_left}')
        print(f'    {full_path_filename_right}')
        print()

        # Check the left file is a PNG image
        # And check that a corresponding right image actually exists

        if files_exist_and_are_png(full_path_filename_left, full_path_filename_right):

            # Read left and right images and display them in windows
            # N.B. Both are stored as 3-channel (even though one is is greyscale)
            # RGB images, so we should load both as such.

            imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
            cv2.imshow('Left Image', imgL)

            imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)
            cv2.imshow('Right image', imgR)

            print('-- Files loaded successfully!')
            print()

            # Calculate and generate the disparity map
            show_disparity(imgL, imgR)

            # Wait 40ms (i.e. 1000 ms / 25 fps = 40 ms)
            key = cv2.waitKey(40 * (not pause_playback)) & 0xFF
            if key == ord('x'):
                # Exit
                print('Exiting...')
                break
            elif key == ord('c'):
                # Toggle cropping
                if crop_disparity == True:
                    print('Disabled crop.')
                    crop_disparity = False
                else:
                    print('Enabled crop.')
                    crop_disparity = True
            elif key == ord(' '):
                # Pause (on next frame)
                if pause_playback == True:
                    print('Resumed playback.')
                    pause_playback = False
                else:
                    print('Paused playback.')
                    pause_playback = True

        else:
            print('-- Files skipped. Perhaps one is missing, or not PNG.')
            print()


loop_through_files()
