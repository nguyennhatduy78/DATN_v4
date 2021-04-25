import time

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl.flags import FLAGS
import tools.utils as utils
from tools.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import save_model as sm
import cv2
import numpy as np

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession


# Variables:
file_out = False
new = False
dont_show = False
video_file_name = 'stop1.mp4'
weights_prototype_file = 'yolov4.weights'
input_size = 416
threshold = 0.2
class_names = 'coco.names'


def Run():
    if new:
        sm.save_tf(weights_prototype_file, input_size, threshold, class_names)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    model_loaded = tf.saved_model.load('./weights/{}'.format(weights_prototype_file.split('.')[0]),
                                       tags=[tag_constants.SERVING])
    infer = model_loaded.signatures['serving_default']
    if video_file_name != "":
        vid = cv2.VideoCapture('./data/' + video_file_name)
    else:
        vid = cv2.VideoCapture(0)
    if file_out:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    frame_num = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_num += 1
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break

        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.5,
            score_threshold=threshold
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names_list = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names_list.values())

        # custom allowed classes (uncomment line below to allow detections for only people)
        # allowed_classes = ['stop sign']

        image = utils.draw_bbox(frame, pred_bbox, True, allowed_classes=allowed_classes, read_plate=False)

        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        if not dont_show:
            cv2.imshow("result", result)

        if file_out:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()
