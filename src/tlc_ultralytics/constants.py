from ultralytics.utils import colorstr

# Column names
EXAMPLE_ID = "example_id"
FOREIGN_TABLE_ID = "input_table_id"
CONFIDENCE = "confidence"
PREDICTED_BOUNDING_BOXES = "bbs_predicted"
PREDICTED_SEGMENTATIONS = "segmentations_predicted"
PREDICTED_KEYPOINTS_2D = "keypoints_2d_predicted"

# Instance embeddings
PREDICTED_INSTANCE_EMBEDDING = "predicted_instance_embedding"
PREDICTED_INSTANCE_EMBEDDING_RAW = "predicted_instance_embedding_raw"
GROUND_TRUTH_INSTANCE_EMBEDDING = "ground_truth_instance_embedding"
GROUND_TRUTH_INSTANCE_EMBEDDING_RAW = "ground_truth_instance_embedding_raw"

LABEL = "label"
EPOCH = "epoch"
TRAINING_PHASE = "Training Phase"
IMAGE_COLUMN_NAME = "image"
CLASSIFY_LABEL_COLUMN_NAME = "label"
DETECTION_LABEL_COLUMN_NAME = "bbs.instances_additional_data.label"
SEGMENTATION_LABEL_COLUMN_NAME = "segmentations.instance_properties.label"
OBB_LABEL_COLUMN_NAME = "oriented_bbs_2d"
POSE_LABEL_COLUMN_NAME = "keypoints_2d"
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
