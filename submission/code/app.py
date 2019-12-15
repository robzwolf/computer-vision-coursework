############################################################################
#
# Computer Vision Coursework, Software Systems and Applications III
# Author: vzbf32
#
# TO RUN THIS FILE (after installing / activating OpenCV): python3 app.py
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
import helpers
import preprocessor
import yolo_engine

print('Loading... This can take a few seconds on some machines.')

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
nms_threshold = 0.4

# Width of network's input image
input_width = 416

# Height of network's input image
input_height = 416

##########################################
# Playback Controls
##########################################

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

# Set up network using configuration and weight files for the model
(net, output_layer_names) = yolo_engine.setup_net(config_file, weights_file)

# Resolve full directory location of data set for left / right images
full_path_directory_left = os.path.join(master_path_to_dataset, directory_to_cycle_left)
full_path_directory_right = os.path.join(master_path_to_dataset, directory_to_cycle_right)

# Get a list of the left image files and sort them (by timestamp in filename)
left_file_list = sorted(os.listdir(full_path_directory_left))


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
    global skip_forward_file_pattern, pause_playback

    # Loop through every file the directory for the left images, and process it and display
    # the relevant windows.
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
        print('    ' + full_path_filename_left)
        print('    ' + full_path_filename_right)
        print()

        # Check the left file is a PNG image
        # And check that a corresponding right image actually exists
        if files_exist_and_are_png(full_path_filename_left, full_path_filename_right):

            # Read left and right images.
            # N.B. Both are stored as 3-channel (even though one is is greyscale)
            # RGB images, so we should load both as such.
            imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
            imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

            print('-- Files loaded successfully!')
            print()
            print('At any point, press the following keys to control playback:')
            print('    (x) or (q) - quit')
            print('    (space)    - play/pause playback')

            # Crop main image so we only detect objects in the uncropped area
            cropped_imgL = preprocessor.preprocess_image_crop_irrelevant_regions(imgL)
            cropped_imgR = preprocessor.preprocess_image_crop_irrelevant_regions(imgR)

            # Calculate the disparity map
            disparity_map = disparity_engine.get_disparity(cropped_imgL, cropped_imgR)

            # Display the disparity map as an image
            disparity_engine.display_disparity_window(disparity_map)

            preprocessed_cropped_imgL = preprocessor.preprocess_image_for_yolo(cropped_imgL)

            # Create a 4D tensor (OpenCV 'blob') from image frame (pixels scaled 0 to 1, image resized)
            tensor = cv2.dnn.blobFromImage(preprocessed_cropped_imgL, 1 / 255, (input_width, input_height))

            # Set the input to the CNN network
            net.setInput(tensor)

            # Run forward inference to get output of the final output layers
            results = net.forward(output_layer_names)

            # Remove the bounding boxes with lower confidence
            class_IDs, confidences, boxes = yolo_engine.postprocess(preprocessed_cropped_imgL, results, confidence_threshold, nms_threshold)

            # Closest object (start with something really high)
            closest_object_distance = 0

            # Get indices (objects) and draw info and distance label for each one
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
            for i in indices:
                i = i[0]
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]

                # Calculate coordinates of object (x,y) and select mean pixels in the middle
                size = helpers.get_mean_pixels(top, top + height, left, left + width) / 2
                x_start, x_end = helpers.bounding_box_centre(left, left + width, size)
                y_start, y_end = helpers.bounding_box_centre(top, top + height, size)

                # Let's calculate Z (depth of an object) for each pixel
                # From the lecture slides, Z = f * B / disparity_map(x,y)
                Z = []

                # Loop through all pixels and calculate Z, taking into account focal length and baseline
                for x in range(x_start, x_end):
                    for y in range(y_start, y_end):
                        try:
                            if disparity_map[y][x] > 0:
                                Z_single = (camera_focal_length_px * stereo_camera_baseline_m) / disparity_map[y][x]
                                Z.append(Z_single)
                        except IndexError:
                            # If we couldn't access disparity_map[y][x] for some reason, just continue to the next
                            # pixel. We'll get an IndexError if we're accessing a part of the disparity map that's
                            # cropped out.
                            continue

                # Work out which object is closest to the camera
                median = helpers.get_median(Z)
                if closest_object_distance == 0 or (median and median < closest_object_distance):
                    closest_object_distance = median

                # Convert Z to a formatted number calculating a median of the middle portion of box pixels
                formatted_depth = helpers.get_formatted_median(Z)

                # Colour of the outline box, in Blue, Green, Red format
                box_outline_colour = helpers.random_colour()

                # Draw the bounding box for the detected object.
                # Make sure to shift the boxes back down by the amount we cropped from the top of the original imgL.
                yolo_engine.draw_bounding_box(imgL, classes[class_IDs[i]], confidences[i], left, top + helpers.crop_top,
                                              left + width, top + height + helpers.crop_top, box_outline_colour,
                                              formatted_depth)

            print()
            print('Nearest detected scene object (' + helpers.format_median(closest_object_distance) + ')')
            print()

            cv2.imshow('YOLOv3 Object Detection using "' + weights_file + '"', imgL)

            # Wait 40ms (i.e. 1000 ms / 25 fps = 40 ms)
            key = cv2.waitKey(40 * (not pause_playback)) & 0xFF
            if key == ord('x') or key == ord('q'):
                # Exit application
                print('Exiting...')
                break
            elif key == ord(' '):
                # Pause (on next frame)
                if pause_playback:
                    print('Resumed playback.')
                    pause_playback = False
                else:
                    print('Paused playback.')
                    pause_playback = True

        else:
            print('-- Files skipped. Perhaps one is missing, or not PNG.')
            print()

    # Close all windows
    cv2.destroyAllWindows()


loop_through_files()
