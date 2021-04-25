from easydict import EasyDict as edict
import sys
sys.path.append("..")
__C = edict()
# Consumers can get config by: from config import cfg

cfg = edict()

# YOLO options
cfg.YOLO = edict()

cfg.YOLO.CLASSES = './weights/coco.names'
cfg.YOLO.ANCHORS = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
cfg.YOLO.ANCHORS_V3 = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
cfg.YOLO.ANCHORS_TINY = [23, 27, 37, 58, 81, 82, 81, 82, 135, 169, 344, 319]
cfg.YOLO.STRIDES = [8, 16, 32]
cfg.YOLO.STRIDES_TINY = [16, 32]
cfg.YOLO.XYSCALE = [1.2, 1.1, 1.05]
cfg.YOLO.XYSCALE_TINY = [1.05, 1.05]
cfg.YOLO.ANCHOR_PER_SCALE = 3
cfg.YOLO.IOU_LOSS_THRESH = 0.5

# Train options
__C.TRAIN = edict()

__C.TRAIN.ANNOT_PATH = "./data/dataset/val2017.txt"
__C.TRAIN.BATCH_SIZE = 2
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE = 416
__C.TRAIN.DATA_AUG = True
__C.TRAIN.LR_INIT = 1e-3
__C.TRAIN.LR_END = 1e-6
__C.TRAIN.WARMUP_EPOCHS = 2
__C.TRAIN.FISRT_STAGE_EPOCHS = 20
__C.TRAIN.SECOND_STAGE_EPOCHS = 30

# TEST options
__C.TEST = edict()

__C.TEST.ANNOT_PATH = "./data/dataset/val2017.txt"
__C.TEST.BATCH_SIZE = 2
__C.TEST.INPUT_SIZE = 416
__C.TEST.DATA_AUG = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD = 0.25
__C.TEST.IOU_THRESHOLD = 0.5
