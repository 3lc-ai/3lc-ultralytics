"""Constants shared by the test modules: per-task datasets, models and trainers, and optional-dependency flags."""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import numpy as np
import pytest
import tlc
from packaging.version import Version
from tlc.helpers import KeypointHelper
from tmp_paths import PROJECT_ROOT
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.models.yolo.segment import SegmentationTrainer

from tlc_ultralytics.classify.trainer import TLCClassificationTrainer
from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.obb.trainer import TLCOBBTrainer
from tlc_ultralytics.pose.trainer import TLCPoseTrainer
from tlc_ultralytics.segment.trainer import TLCSegmentationTrainer

TMP_PROJECT_ROOT_URL = tlc.Url(PROJECT_ROOT)


# PaCMAP < 0.9 (annoy-based) is known not to work on macOS: the reducer collects
# zero embeddings and silently produces no reduced table. Embedding-specific
# checks are therefore skipped there (everything else still runs). PaCMAP 0.9 uses faiss and works.
PACMAP_BROKEN_ON_MACOS = sys.platform == "darwin" and Version(importlib.metadata.version("pacmap")) < Version("0.9")


skip_pacmap_on_macos = pytest.mark.skipif(
    PACMAP_BROKEN_ON_MACOS,
    reason="PaCMAP embedding reduction does not work on macOS",
)


DUMMY_IMAGE_FILE = Path(__file__).parent.parent / "src" / "tlc_ultralytics" / "_static" / "dashboard.png"


TASK2DATASET = {
    "detect": "coco8.yaml",
    "classify": "imagenet10",
    "segment": "coco8-seg.yaml",
    "pose": "coco8-pose.yaml",
    "obb": "dota8.yaml",
}


TASK2MODEL = {
    "detect": "yolo26n.pt",
    "classify": "yolo26n-cls.pt",
    "segment": "yolo26n-seg.pt",
    "pose": "yolo26n-pose.pt",
    "obb": "yolo26n-obb.pt",
}


TASK2LABEL_COLUMN_NAME = {
    "detect": "bbs.instances_additional_data.label",
    "classify": "label",
    "segment": "segmentations.instance_properties.label",
    "pose": "keypoints_2d",
    "obb": "oriented_bbs_2d",
}


TASK2PREDICTED_LABEL_COLUMN_NAME = {
    "detect": "bbs_predicted.instances_additional_data.label",
    "classify": "predicted",
    "segment": "segmentations_predicted.instance_properties.label",
    "pose": "keypoints_2d_predicted",
    "obb": "oriented_bbs_2d_predicted",
}


TASK2TRAINER = {
    "detect": TLCDetectionTrainer,
    "classify": TLCClassificationTrainer,
    "segment": TLCSegmentationTrainer,
    "obb": TLCOBBTrainer,
    "pose": TLCPoseTrainer,
}


TASK2ULTRALYTICS_TRAINER = {
    "classify": PoseTrainer,
    "obb": OBBTrainer,
    "pose": PoseTrainer,
    "segment": SegmentationTrainer,
    "detect": DetectionTrainer,
}


COCO_POSE_SETTINGS_OVERRIDES = {
    "points": KeypointHelper.COCO_KEYPOINT_DEFAULT_POSE,
    "lines": KeypointHelper.COCO_SKELETON,
    "point_attributes": [f"p{i}" for i in range(17)],
    "line_attributes": [f"l{i}" for i in range(16)],
}


OKS_SIGMAS = np.array([0.069] * 17, dtype=np.float64)


try:
    import umap  # noqa: F401

    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False


INSTANCE_EMB_OVERRIDES = {"batch": 4, "device": "cpu", "workers": 0}
