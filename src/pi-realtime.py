#! /usr/bin/env python
import argparse
import colorsys
import imghdr
import os
import random
import sys
import time

import cv2 # use opencv 3.4.0
import picamera
import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

from yad2k.models.keras_yolo import yolo_eval, yolo_head


MODEL_PATH = 'model_data/yolov2-tiny-voc.h5'
ANCHORS_PATH = 'model_data/yolov2-tiny-voc_anchors.txt'
CLASSES_PATH = 'model_data/pascal_classes.txt'

parser = argparse.ArgumentParser(
    description='Run a YOLO_v2 style detection model on test images..')
parser.add_argument(
    '-s',
    '--score_threshold',
    type=float,
    help='threshold for bounding box scores, default .3',
    default=.3)
parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .5',
    default=.5)

def detect_image(image, sess, boxes, scores, classes):
    image = Image.fromarray(image)

    resized_image = image.resize(
        tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
            yolo_model.input: image_data,
            input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })

    font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        # print detection result
        print(predicted_class, score)

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return image



def _main(args):
    sess = K.get_session()

    print("loading class file")
    global class_names
    with open(CLASSES_PATH) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    print("loading anchors")
    with open(ANCHORS_PATH) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    print("loading yolo model")
    global yolo_model
    yolo_model = load_model(MODEL_PATH)


    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    # TODO: Assumes dim ordering is channel last
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == len(anchors) * (num_classes + 5), \
        'Mismatch between model and given anchor and class sizes. ' \
        'Exit.'
    print('{} model, anchors, and classes loaded.'.format(MODEL_PATH))

    # Check if model is fully convolutional, assuming channel last order.
    global model_image_size
    model_image_size = yolo_model.layers[0].input_shape[1:3]

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / num_classes, 1., 1.)
                  for x in range(num_classes)]
    global colors
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(yolo_model.output, anchors, num_classes)
    global input_image_shape
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs,
        input_image_shape,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold)


    #########################################################################
    print("capture pi camera...")

    # set camera resolution
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 320)

    # use opencv to capture frames
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    cam.set(cv2.CAP_PROP_FPS, 30)

    rval, frame = cam.read()
    if rval == False:
        print("Error: Camera is not Found. Exit")
        sys.exit(1)
    cv2.imshow("preview", np.array(frame))

    while True:
        rval, frame = cam.read()
        before = time.time()
        detect_img = detect_image(frame, sess, boxes, scores, classes)
        after = time.time()

        cv2.imshow("preview", np.array(detect_img))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print("FPS: {}".format(1 / (after-before)))

    sess.close()
    cam.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    _main(parser.parse_args())
