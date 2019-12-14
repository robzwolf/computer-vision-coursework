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

import cv2


# Dummy on trackbar callback function
import numpy as np


def on_trackbar(val):
    pass


def draw_bounding_box(output_image, class_name, confidence, left, top, right, bottom, colour, depth):
    """
    Draw the predicted bounding box on the specified image.
    @param output_image: Image upon which detection is performed
    @param class_name: String name of detected object_detection
    @param left: Rectangle parameter for detection
    @param top: Rectangle parameter for detection
    @param right: Rectangle parameter for detection
    @param bottom: Rectangle parameter for detection
    @param colour: Colour in which to draw detection rectangle
    @param depth: Distance of the detected object from the camera
    """

    # Draw a bounding box
    box_thickness = 3
    cv2.rectangle(output_image, (left, top), (right, bottom), colour, box_thickness)

    # Construct label
    label_text = f'{class_name}: {depth}'

    font_scale = 0.5
    font_thickness = 1
    label_size, base_line = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, font_thickness)
    top = max(top, label_size[1])

    # Draw label background
    white = (255, 255, 255)
    cv2.rectangle(output_image, (left, top - round(1.5 * label_size[1])),
                  (left + round(1.5 * label_size[0]), top + base_line), white, cv2.FILLED)

    # Draw the actual label text
    black = (0, 0, 0)
    cv2.putText(output_image, label_text, (left, top), cv2.FONT_HERSHEY_DUPLEX, font_scale, black, font_thickness)


def postprocess(image, results, threshold_confidence, threshold_nms):
    """
    Remove the bounding boxes with low confidence using non-maxima suppression.
    @param image: Image upon which detection was performed
    @param results: Output from YOLO CNN network
    @param threshold_confidence: Threshold on keeping detection
    @param threshold_nms: Threshold used in non-maximum suppression
    @return:
    """
    frame_height = image.shape[0]
    frame_width = image.shape[1]

    # Scan through all the bounding boxes output from the network and:
    #     1) Keep only the ones with high confidence scores
    #     2) Assign the box class label as the class with the highest score
    #     3) Construct a list of bounding boxes, class labels and confidence scores

    class_IDs = []
    confidences = []
    boxes = []

    for result in results:
        for detection in result:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > threshold_confidence:
                centre_x = int(detection[0] * frame_width)
                centre_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(centre_x - width/2)
                top = int(centre_y - height/2)
                class_IDs.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non-maximum suppression to eliminate redundant overlapping boxes
    # with lower confidences.
    class_IDs_nms = []
    confidences_nms = []
    boxes_nms = []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold_confidence, threshold_nms)
    for i in indices:
        i = i[0]
        class_IDs_nms.append(class_IDs[i])
        confidences_nms.append(confidences[i])
        boxes_nms.append(boxes[i])

    # Return post-processed lists of class_IDs, confidences and bounding boxes
    return (class_IDs_nms, confidences_nms, boxes_nms)


def get_outputs_names(net):
    """
    Get the names of the output layers of the CNN network.
    @param net: An OpenCV DNN module network object
    """

    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def setup_net(config_file, weights_file):

    # Load configuration and weight files for the model and load the network using them
    net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
    output_layer_names = get_outputs_names(net)

    # Defaults to DNN_BACKEND_INFERENCE_ENGINE if Intel Inference Engine lib is available
    # or DNN_BACKEND_OPENCV otherwise
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)

    # Change this to cv2.dnn.DNN_TARGET_CUP (slower) if this causes issues.
    # (It should fail gracefully if OpenCL is not available.)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

    # # Set up display window name and trackbar
    # window_name = f"Object Detection (YOLOv3) using '{weights_file}'"
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # trackbar_name = 'Reporting confidence > (x 0.01)'
    # initial_trackbar_value = 0
    # trackbar_count = 100
    # cv2.createTrackbar(trackbar_name, window_name, initial_trackbar_value, trackbar_count, on_trackbar)

    return (net, output_layer_names)
