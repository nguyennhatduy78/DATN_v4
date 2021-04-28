import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import tools.utils as utils
from tools.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import save_model as sm
import cv2
import numpy as np

# Input setting
input_type = 1
webcam = False
image_file_path = './data/noentry/noentry.jpeg'
video_file_path = './data/noentry/noentry.mov' if not webcam and input_type == 2 else 0

# Opt init
file_out_video = True
file_out_image = True
new = False
dont_show = False

# Parameter init
weights_prototype_file = 'noentryV4_final.weights'
class_names = 'noentry.names'
input_size = 608
threshold = 0.2

# Variables
video_file_name = video_file_path.split('/')[-1]
image_file_name = image_file_path.split('/')[-1]


def Run():
    print("Current threshold: ", threshold)
    cfg.YOLO.CLASSES = './weights/' + class_names
    if new:
        sm.save_tf(weights_prototype_file, input_size, threshold, class_names)
    # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
    model_loaded = tf.saved_model.load('./weights/{}'.format(weights_prototype_file.split('.')[0]),
                                       tags=[tag_constants.SERVING])
    infer = model_loaded.signatures['serving_default']
    # read in all class names from config
    class_names_list = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    allowed_classes = list(class_names_list.values())

    if input_type == 1:
        image_ = cv2.imread(image_file_path)
        image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image_, (input_size, input_size))
        image = image / 255
        image = image[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image)
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
        original_h, original_w, _ = image_.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
        image = utils.draw_bbox(image_, pred_bbox, True, allowed_classes=allowed_classes, read_plate=False)
        image = Image.fromarray(image.astype(np.uint8))
        if not dont_show:
            image.show()
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        if file_out_image:
            cv2.imwrite('./data/image/{}.jpg'.format(image_file_name.split('.')[0] + '_result'), image)
    elif input_type == 2:
        vid = cv2.VideoCapture(video_file_path)
        if file_out_video:
            # by default VideoCapture returns float instead of int
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('./data/video/{}.mp4'.format(video_file_name.split('.')[0] + '_result'), codec, fps,
                                  (width, height))
        while True:
            ret, frame = vid.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                print("Video end")
                break
            # frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
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
            image = utils.draw_bbox(frame, pred_bbox, True, allowed_classes=allowed_classes, read_plate=False)
            # result = np.asarray(image)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if not dont_show:
                cv2.imshow("result", result)
            if file_out_video:
                out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Q pressed")
                break
        vid.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Run()
