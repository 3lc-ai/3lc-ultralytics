import tlc
from ultralytics.utils import colorstr

# Column names
TRAINING_PHASE = "Training Phase"
IMAGE_COLUMN_NAME = tlc.IMAGE
CLASSIFY_LABEL_COLUMN_NAME = tlc.LABEL
DETECTION_LABEL_COLUMN_NAME = f"{tlc.BOUNDING_BOXES}.{tlc.BOUNDING_BOX_LIST}.{tlc.LABEL}"
SEGMENTATION_LABEL_COLUMN_NAME = f"{tlc.SEGMENTATIONS}.{tlc.INSTANCE_PROPERTIES}.{tlc.LABEL}"
PRECISION = "precision"
PRECISION_SEG = "precision_seg"
RECALL = "recall"
RECALL_SEG = "recall_seg"
MAP = "mAP"
MAP_SEG = "mAP_seg"
MAP50_95 = "mAP50-95"
MAP50_95_SEG = "mAP50-95_seg"
NUM_IMAGES = "num_images"
NUM_INSTANCES = "num_instances"
PER_CLASS_METRICS_STREAM_NAME = "per_class_metrics"

# Other
DEFAULT_TRAIN_RUN_DESCRIPTION = ""
DEFAULT_COLLECT_RUN_DESCRIPTION = "Created with model.collect()"

TLC_PREFIX = "3LC://"
TLC_COLORSTR = colorstr("3lc: ")

REQUIREMENTS_TO_CHECK = [
    ("3lc", "tlc"),
    ("ultralytics", "ultralytics"),
]
