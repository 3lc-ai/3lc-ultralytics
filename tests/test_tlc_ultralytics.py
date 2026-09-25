from __future__ import annotations

import io
import json
import logging
import os
import pathlib
import pickle
import random
import sys
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pandas as pd
import pytest
import tlc
import yaml
from PIL import Image
from testing_helpers import (
    check_pose_table_and_metrics_tables,
    compare_dataset_values,
    plot_ultralytics,
    stub_model_with_stride,
)
from tlc._core.objects.tables.from_table.edited_table import EditedTable
from tlc._core.objects.tables.null_overlay import NullOverlay
from tlc.constants._run_status import RUN_STATUS_COMPLETED
from tlc.helpers import KeypointHelper
from tmp_paths import PROJECT_ROOT, TMP
from ultralytics.cfg import ASSETS
from ultralytics.models.yolo import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.models.yolo.semantic import SemanticSegmentationTrainer

from tlc_ultralytics import YOLO as TLCYOLO
from tlc_ultralytics import Settings
from tlc_ultralytics.classify.trainer import TLCClassificationTrainer
from tlc_ultralytics.constants import (
    DEFAULT_COLLECT_RUN_DESCRIPTION,
    DETECTION_LABEL_COLUMN_NAME,
    EPOCH,
    EXAMPLE_ID,
    FOREIGN_TABLE_ID,
    LABEL,
    MAP,
    MAP50_95,
    NUM_IMAGES,
    NUM_INSTANCES,
    OBB_LABEL_COLUMN_NAME,
    PER_CLASS_METRICS_STREAM_NAME,
    POSE_LABEL_COLUMN_NAME,
    PRECISION,
    PREDICTED_SEGMENTATIONS,
    PREDICTED_SEMANTIC_SEGMENTATION,
    RECALL,
    SEGMENTATION_LABEL_COLUMN_NAME,
    TRAINING_PHASE,
)
from tlc_ultralytics.detect.dataset import TLCYOLODataset
from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.engine.dataset import TLCDatasetMixin
from tlc_ultralytics.obb.trainer import TLCOBBTrainer
from tlc_ultralytics.pose.trainer import TLCPoseTrainer
from tlc_ultralytics.segment.trainer import TLCSegmentationTrainer
from tlc_ultralytics.segment.utils import check_seg_table
from tlc_ultralytics.semantic.trainer import TLCSemanticSegmentationTrainer
from tlc_ultralytics.utils import check_tlc_dataset
from tlc_ultralytics.utils.dataset import _complete_label_column_name

# PaCMAP embedding reduction is known not to work on macOS: the reducer collects
# zero embeddings and silently produces no reduced table. Embedding-specific
# checks are therefore skipped on macOS (everything else still runs there).
PACMAP_BROKEN_ON_MACOS = sys.platform == "darwin"
skip_pacmap_on_macos = pytest.mark.skipif(
    PACMAP_BROKEN_ON_MACOS,
    reason="PaCMAP embedding reduction does not work on macOS",
)

DUMMY_IMAGE_FILE = Path(__file__).parent.parent / "src" / "tlc_ultralytics" / "_static" / "dashboard.png"
TMP_PROJECT_ROOT_URL = tlc.Url(PROJECT_ROOT)
tlc.url.register_url_alias("<TEST_ALIAS>", "/test/alias")
tlc.configuration.Configuration.instance().project_root_url = TMP_PROJECT_ROOT_URL
tlc._core.objects.tables.system_tables.indexing_tables.table_indexing_table.TableIndexingTable.instance().add_scan_url(
    {
        "url": tlc.Url(TMP_PROJECT_ROOT_URL),
        "layout": "project",
        "object_type": "table",
        "static": True,
    }
)

TASK2DATASET = {
    "detect": "coco8.yaml",
    "classify": "imagenet10",
    "segment": "coco8-seg.yaml",
    "pose": "coco8-pose.yaml",
    "obb": "dota8.yaml",
    "semantic": "cityscapes8.yaml",
}
TASK2MODEL = {
    "detect": "yolo26n.pt",
    "classify": "yolo26n-cls.pt",
    "segment": "yolo26n-seg.pt",
    "pose": "yolo26n-pose.pt",
    "obb": "yolo26n-obb.pt",
    "semantic": "yolo26n-sem.pt",
}
TASK2LABEL_COLUMN_NAME = {
    "detect": "bbs.instances_additional_data.label",
    "classify": "label",
    "segment": "segmentations.instance_properties.label",
    "pose": "keypoints_2d",
    "obb": "oriented_bbs_2d",
    "semantic": "mask",
}
TASK2PREDICTED_LABEL_COLUMN_NAME = {
    "detect": "bbs_predicted.instances_additional_data.label",
    "classify": "predicted",
    "segment": "segmentations_predicted.instance_properties.label",
    "pose": "keypoints_2d_predicted",
    "obb": "oriented_bbs_2d_predicted",
    "semantic": PREDICTED_SEMANTIC_SEGMENTATION,
}
TASK2TRAINER = {
    "detect": TLCDetectionTrainer,
    "classify": TLCClassificationTrainer,
    "segment": TLCSegmentationTrainer,
    "obb": TLCOBBTrainer,
    "pose": TLCPoseTrainer,
    "semantic": TLCSemanticSegmentationTrainer,
}

TASK2ULTRALYTICS_TRAINER = {
    "classify": PoseTrainer,
    "obb": OBBTrainer,
    "pose": PoseTrainer,
    "segment": SegmentationTrainer,
    "detect": DetectionTrainer,
    "semantic": SemanticSegmentationTrainer,
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


def get_metrics_tables_from_run(run: tlc.Run) -> dict[str, list[tlc.Table]]:
    """Return metrics tables grouped by stream name"""
    metrics_infos = run.metrics
    metrics_tables = defaultdict(list)
    for metrics_info in metrics_infos:
        metrics_table = tlc.Table.from_url(tlc.Url(metrics_info["url"]).to_absolute(run.url))
        metrics_tables[metrics_info["stream_name"]].append(metrics_table)
    return metrics_tables


class CapturingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_records = []
        self.log_messages = []

    def emit(self, record):
        self.log_records.append(record)
        self.log_messages.append(self.format(record))


def override_oks_sigmas() -> None:
    """Return coco8-pose.yaml contents overridden with the COCO oks_sigmas"""
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.metrics import OKS_SIGMA

    data = check_det_dataset("coco8-pose.yaml")

    yaml_data = {
        "train": data["train"],
        "val": data["val"],
        "test": None,
        "kpt_shape": [17, 3],
        "flip_idx": [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
        "oks_sigmas": OKS_SIGMA.tolist(),
        "names": {0: "person"},
        "nc": 1,
    }

    (TMP / "coco8-pose.yaml").write_text(yaml.safe_dump(yaml_data))


@pytest.mark.parametrize("task", ["detect", "segment", "pose", "obb"])
def test_training(task: str) -> None:
    # End-to-end test of training for detection, segmentation, pose, and obb

    overrides = {
        "data": TASK2DATASET[task],
        "epochs": 2,
        "batch": 4,
        "device": "cpu",
        "save": True,
        "plots": False,
        "seed": 3 + ord("L") + ord("C"),
        "deterministic": True,
        "workers": 0,
    }

    if task == "pose":
        override_oks_sigmas()
        overrides["data"] = str(TMP / "coco8-pose.yaml")

    settings = Settings(
        collection_epoch_start=1,
        project_name=f"test_{task}_project",
        run_name=f"test_{task}",
        run_description=f"Test {task} training",
        collect_loss=True,
    )

    # Run ultralytics training and capture logs
    model_ultralytics = YOLO(TASK2MODEL[task])
    results_ultralytics = model_ultralytics.train(**overrides)

    # Run 3LC training and capture logs
    model_3lc = TLCYOLO(TASK2MODEL[task])
    with capture_logs() as tlc_messages:
        results_3lc = model_3lc.train(**overrides, settings=settings)

        assert results_3lc, "Detection training failed"

    msg = "Update with 'pip install -U ultralytics'"

    # Check that there are no messages prompting an update
    assert not any(msg in message for message in tlc_messages), (
        f"Found {msg} in 3LC logs, which should be patched to not happen"
    )

    # Compare 3LC integration with ultralytics results
    # Segmentation results will be slightly different due to the 3lc mask storage format and conversion
    # back to polygons. Detection may have minor numerical differences due to data loading variations.
    atol = 0.01
    if task == "segment":
        atol = 0.1
    elif task == "pose":
        atol = 0.1
    for k in results_ultralytics.results_dict.keys():
        assert np.isclose(results_ultralytics.results_dict[k], results_3lc.results_dict[k], atol=atol), (
            f"Results validation metrics 3LC different from Ultralytics for {k}"
        )

    assert results_ultralytics.names == results_3lc.names, "Results validation names"

    # Ensure the trainer can be serialized
    trainer_serialized = model_3lc.trainer._serialize_state()
    assert isinstance(trainer_serialized, str), "Trainer serialization failed"
    trainer_json_content = json.loads(trainer_serialized)
    assert trainer_json_content["run_url"] == model_3lc.trainer._run.url.to_str(), "Run URL mismatch"
    assert trainer_json_content["settings"] == model_3lc.trainer._settings.to_dict(), "Settings mismatch"
    assert trainer_json_content["data"] == model_3lc.trainer.args.data, "Data mismatch"
    assert trainer_json_content["tables"] == model_3lc.trainer._tables, "Tables mismatch"

    # Get 3LC run and inspect the results
    run = _get_run_from_settings(settings)

    assert run.status == RUN_STATUS_COMPLETED, "Run status not set to completed after training"

    assert run.project_name == settings.project_name, "Project name mismatch"
    assert run.description == settings.run_description, "Description mismatch"
    # Check that hyperparameters and overrides are saved
    for key, value in overrides.items():
        assert run.constants["parameters"][key] == value, (
            f"Parameter {key} mismatch, {run.constants['parameters'][key]} != {value}"
        )

    # Check that confidence-recall-precision-f1 data is written
    assert "3LC/Precision" in run.constants["parameters"]
    assert "3LC/Recall" in run.constants["parameters"]
    assert "3LC/F1_score" in run.constants["parameters"]

    # Check that there is a per-epoch value written
    assert len(run.constants["outputs"]) > 0, "No outputs written"

    # Each metrics table URL must be registered under exactly one stream. A previous bug had
    # the per-class writer auto-register under default_stream and then re-register under
    # per_class_metrics, which polluted default_stream with per-class rows lacking
    # example_id/predictions, breaking dashboard rendering.
    url_streams: dict[str, list[str]] = defaultdict(list)
    for info in run.metrics:
        url_streams[info["url"]].append(info["stream_name"])
    duplicates = {url: streams for url, streams in url_streams.items() if len(streams) > 1}
    assert not duplicates, f"Metrics URLs registered under multiple streams: {duplicates}"

    metrics_tables = get_metrics_tables_from_run(run)

    # Check that the desired metrics were written
    metrics_df = pd.concat(
        [m.to_pandas() for m in metrics_tables["default_stream"]],
        ignore_index=True,
    )
    # Per-sample loss columns are covered by the dedicated per-sample loss tests.
    assert 0 in metrics_df[TRAINING_PHASE], "Expected metrics from during training"
    assert 1 in metrics_df[TRAINING_PHASE], "Expected metrics from after training"

    if task == "segment":
        # Get row of metrics_df with epoch = 1 and Training Phase = 1
        row = metrics_df[(metrics_df["epoch"] == 1) & (metrics_df["example_id"] == 3)]["segmentations_predicted"][0]
        prediction_category = row["instance_properties"]["label"][0]

        category = model_3lc.trainer.data["names"][prediction_category]
        assert category == "zebra", "Expected zebra as first prediction when epoch = 1 and example_id = 3"

    # model.predict() should work and be the same as vanilla ultralytics
    # Pass explicit source for OBB to avoid downloading boats.jpg to the project root
    source = ASSETS if task != "obb" else ASSETS / "bus.jpg"
    if task == "obb":
        ultralytics_pred = model_ultralytics.predict(source, imgsz=320)[0]
        tlc_pred = model_3lc.predict(source, imgsz=320)[0]
        assert all(ultralytics_pred.obb.cls == tlc_pred.obb.cls), "Predictions mismatch"

    else:
        ultralytics_pred = model_ultralytics.predict(source, imgsz=320)[0]
        tlc_pred = model_3lc.predict(source, imgsz=320)[0]
        assert all(ultralytics_pred.boxes.cls == tlc_pred.boxes.cls), "Predictions mismatch"

    if task == "pose":
        # Pose does not collect per-class metrics (yet?!)
        return

    per_class_metrics_tables = metrics_tables[PER_CLASS_METRICS_STREAM_NAME]
    # 6 = 2 epochs * 2 splits + 2 splits after training
    assert len(per_class_metrics_tables) == 6, "Expected 6 per-class metrics tables to be written"
    per_class_metrics_df = pd.concat(
        [m.to_pandas() for m in per_class_metrics_tables],
        ignore_index=True,
    )

    foreign_table_url = per_class_metrics_tables[0].get_foreign_table_url()
    assert not foreign_table_url.is_absolute(), "Expected foreign table url to be relative"
    assert foreign_table_url.to_absolute(per_class_metrics_tables[0].url).exists()

    assert TRAINING_PHASE in per_class_metrics_df.columns, "Expected training phase column in per-class metrics"
    assert EPOCH in per_class_metrics_df.columns, "Expected epoch column in per-class metrics"
    assert FOREIGN_TABLE_ID in per_class_metrics_df.columns, "Expected foreign_table_id column in per-class metrics"
    assert LABEL in per_class_metrics_df.columns, "Expected label column in per-class metrics"
    assert PRECISION in per_class_metrics_df.columns, "Expected precision column in per-class metrics"
    assert RECALL in per_class_metrics_df.columns, "Expected recall column in per-class metrics"
    assert MAP in per_class_metrics_df.columns, "Expected mAP column in per-class metrics"
    assert MAP50_95 in per_class_metrics_df.columns, "Expected mAP50-95 column in per-class metrics"
    assert NUM_IMAGES in per_class_metrics_df.columns, "Expected num_images column in per-class metrics"
    assert NUM_INSTANCES in per_class_metrics_df.columns, "Expected num_instances column in per-class metrics"


def test_detect_training_with_yolo12() -> None:
    model = "yolo12n.pt"
    data = TASK2DATASET["detect"]
    overrides = {"data": data, "device": "cpu", "epochs": 1, "batch": 64, "imgsz": 32, "label_column_name": "bbs"}

    model_3lc = TLCYOLO(model)
    # Embeddings can't be collected for yolo12
    with pytest.raises(ValueError):
        model_3lc.train(**overrides, settings=Settings(image_embeddings_dim=2, run_name="test_yolo12_embeddings"))

    # But should run to completion without embeddings collection
    model_3lc.train(**overrides, settings=Settings(run_name="test_yolo12_no_embeddings"))

    # Also check that validation is skipped when training and validating on the same table
    overrides["tables"] = {
        "train": tlc.Table.from_url(model_3lc.trainer.data["train"].url),
        "val": tlc.Table.from_url(model_3lc.trainer.data["train"].url),
    }
    results_dupe = model_3lc.train(**overrides, settings=Settings(run_name="test_yolo12_dupe_validation"))

    run_dupe = tlc.Run.from_url(results_dupe.run_url)
    per_sample_metrics_tables = get_metrics_tables_from_run(run_dupe)["default_stream"]

    assert len(per_sample_metrics_tables) == 1, (
        "Expected 1 per-sample metrics table to be written when training and validating on the same table"
    )


def test_detect_training_with_yolo11_per_sample_loss() -> None:
    """Test that per-sample loss collection works for YOLO11 models (detection)."""
    model = "yolo11n.pt"
    data = TASK2DATASET["detect"]
    overrides = {
        "data": data,
        "device": "cpu",
        "epochs": 1,
        "batch": 4,
        "imgsz": 320,
        "workers": 0,
    }

    settings = Settings(
        project_name="test_yolo11_per_sample_loss",
        run_name="test_yolo11_per_sample_loss",
        collect_loss=True,
        collection_epoch_start=1,
    )

    model_3lc = TLCYOLO(model)
    results = model_3lc.train(**overrides, settings=settings)

    assert results, "YOLO11 detection training with per-sample loss failed"

    run = _get_run_from_settings(settings)
    metrics_tables = get_metrics_tables_from_run(run)

    # Check that per-sample loss metrics were collected
    metrics_df = pd.concat(
        [m.to_pandas() for m in metrics_tables["default_stream"]],
        ignore_index=True,
    )

    # Verify loss columns are present
    assert "loss" in metrics_df.columns, "Expected 'loss' column to be present"
    assert "box_loss" in metrics_df.columns, "Expected 'box_loss' column to be present"
    assert "cls_loss" in metrics_df.columns, "Expected 'cls_loss' column to be present"
    assert "dfl_loss" in metrics_df.columns, "Expected 'dfl_loss' column to be present"

    # Verify loss values are reasonable (not all zeros, not all NaN)
    assert not metrics_df["loss"].isna().all(), "All loss values are NaN"
    assert metrics_df["loss"].sum() > 0, "Total loss should be positive"


def test_detect_yolo26_per_sample_loss() -> None:
    """Test that per-sample loss collection works for YOLO26 (end2end) detection models."""
    model = "yolo26n.pt"
    data = TASK2DATASET["detect"]
    overrides = {
        "data": data,
        "device": "cpu",
        "epochs": 1,
        "batch": 4,
        "imgsz": 320,
        "workers": 0,
    }

    settings = Settings(
        project_name="test_yolo26_per_sample_loss",
        run_name="test_yolo26_per_sample_loss",
        collect_loss=True,
        collection_epoch_start=1,
    )

    model_3lc = TLCYOLO(model)
    with capture_logs() as log_messages:
        results = model_3lc.train(**overrides, settings=settings)

    assert results, "YOLO26 detection training failed"

    # Loss collection is supported for YOLO26 detection models, so no warning should be logged
    loss_warning_found = any("Per-sample loss collection is not supported" in msg for msg in log_messages)
    assert not loss_warning_found, "Unexpected warning about loss collection not being supported"

    run = _get_run_from_settings(settings)
    metrics_tables = get_metrics_tables_from_run(run)

    metrics_df = pd.concat(
        [m.to_pandas() for m in metrics_tables["default_stream"]],
        ignore_index=True,
    )

    # Verify loss columns are present; YOLO26 is DFL-free, so no dfl_loss column is written
    assert "loss" in metrics_df.columns, "Expected 'loss' column to be present"
    assert "box_loss" in metrics_df.columns, "Expected 'box_loss' column to be present"
    assert "cls_loss" in metrics_df.columns, "Expected 'cls_loss' column to be present"
    assert "dfl_loss" not in metrics_df.columns, "Expected no 'dfl_loss' column for DFL-free YOLO26"

    # Verify loss values are reasonable (not all zeros, not all NaN)
    assert not metrics_df["loss"].isna().all(), "All loss values are NaN"
    assert metrics_df["loss"].sum() > 0, "Total loss should be positive"


@pytest.mark.parametrize("task", ["segment", "obb"])
def test_collect_loss_unsupported_task_warns(task) -> None:
    """Test that collect_loss=True warns and is disabled for tasks without per-sample loss support."""
    settings = Settings(
        project_name=f"test_{task}_no_loss",
        run_name=f"test_{task}_no_loss",
        collect_loss=True,
    )

    model_3lc = TLCYOLO(TASK2MODEL[task])
    with capture_logs() as log_messages:
        model_3lc.val(
            data=TASK2DATASET[task],
            device="cpu",
            imgsz=320,
            batch=4,
            workers=0,
            settings=settings,
        )

    loss_warning_found = any(
        f"Per-sample loss collection is not supported for the '{task}' task" in msg for msg in log_messages
    )
    assert loss_warning_found, f"Expected warning about loss collection not being supported for {task}"

    run = _get_run_from_settings(settings)
    metrics_tables = get_metrics_tables_from_run(run)
    metrics_df = pd.concat(
        [m.to_pandas() for m in metrics_tables["default_stream"]],
        ignore_index=True,
    )
    assert "loss" not in metrics_df.columns, f"Expected no 'loss' column for {task}"


def test_pose_yolo26_disables_per_sample_loss() -> None:
    """Test that collect_loss=True warns and is disabled for YOLO26 (end2end) pose models."""
    task = "pose"
    settings = Settings(
        project_name=f"test_{task}_no_loss",
        run_name=f"test_{task}_no_loss",
        collect_loss=True,
    )

    model_3lc = TLCYOLO(TASK2MODEL[task])
    with capture_logs() as log_messages:
        model_3lc.val(
            data=TASK2DATASET[task],
            device="cpu",
            imgsz=320,
            batch=4,
            workers=0,
            settings=settings,
        )

    loss_warning_found = any(
        "Per-sample loss collection is not supported for YOLO26 (end2end) pose models" in msg for msg in log_messages
    )
    assert loss_warning_found, "Expected warning about YOLO26 pose loss collection not being supported"

    run = _get_run_from_settings(settings)
    metrics_tables = get_metrics_tables_from_run(run)
    metrics_df = pd.concat(
        [m.to_pandas() for m in metrics_tables["default_stream"]],
        ignore_index=True,
    )
    assert "loss" not in metrics_df.columns, "Expected no 'loss' column for YOLO26 pose"


def test_classify_instance_embeddings_unsupported_warns() -> None:
    """Test that instance_embeddings_dim > 0 warns and is disabled for the 'classify' task."""
    # yolo26n-cls.pt is pretrained on 1000 imagenet classes, incompatible with the ten-class imagenet10 dataset used
    # in tests, so a model matching the dataset's class count is trained first (mirrors test_classify_training).
    train_model = TLCYOLO(TASK2MODEL["classify"])
    train_results = train_model.train(
        data=TASK2DATASET["classify"],
        device="cpu",
        epochs=1,
        batch=4,
        imgsz=32,
        workers=0,
    )
    best = train_results.save_dir / "weights" / "best.pt"

    settings = Settings(
        project_name="test_classify_no_instance_embeddings",
        run_name="test_classify_no_instance_embeddings",
        instance_embeddings_dim=2,
        instance_embeddings_reducer="pca",
    )

    model_3lc = TLCYOLO(best)
    with capture_logs() as log_messages:
        model_3lc.val(
            data=TASK2DATASET["classify"],
            device="cpu",
            imgsz=320,
            batch=4,
            workers=0,
            settings=settings,
        )

    embeddings_warning_found = any(
        "Instance embeddings are not supported for the 'classify' task" in msg for msg in log_messages
    )
    assert embeddings_warning_found, "Expected warning about instance embeddings not being supported for classify"

    run = _get_run_from_settings(settings)
    metrics_tables = get_metrics_tables_from_run(run)
    metrics_df = pd.concat(
        [m.to_pandas() for m in metrics_tables["default_stream"]],
        ignore_index=True,
    )
    assert not any("embedding" in col for col in metrics_df.columns), "Expected no instance-embedding columns"


def test_classify_training() -> None:
    model = TASK2MODEL["classify"]
    data = TASK2DATASET["classify"]
    overrides = {"data": data, "device": "cpu", "epochs": 3, "batch": 64, "imgsz": 32}

    # Compare results from 3LC with ultralytics
    model_ultralytics = YOLO(model)
    results_ultralytics = model_ultralytics.train(**overrides)

    model_3lc = TLCYOLO(model)

    settings = Settings(
        image_embeddings_dim=3,
        collection_epoch_start=2,
        collection_epoch_interval=1,
        project_name="test_classify_project",
        run_name="test_classify",
    )
    results_3lc = model_3lc.train(**overrides, settings=settings)

    assert results_3lc, "Classification training failed"

    assert results_ultralytics.results_dict == results_3lc.results_dict, (
        "Results validation metrics 3LC different from Ultralytics"
    )

    run = _get_run_from_settings(settings)

    assert run.status == RUN_STATUS_COMPLETED, "Run status not set to completed after training"
    assert not run.description, "Description mismatch, default should be empty string"

    # Imagenet should get special treatment with label display name overrides
    input_table = tlc.Table.from_url(run.url / run.constants["inputs"][0]["input_table_url"])
    value_map = input_table.get_value_map("label")
    assert value_map is not None, "Expected value map to be not None"
    assert all(v.display_name for v in value_map.values()), "Expected display names for all classes"
    display_names = {v.display_name for v in value_map.values()}
    assert display_names == {
        "tench",
        "goldfish",
        "great_white_shark",
        "tiger_shark",
        "hammerhead",
        "electric_ray",
        "stingray",
        "cock",
        "hen",
        "ostrich",
    }

    assert len(run.metrics_tables) == 6  # Two passes after epochs 2 and 3, and after training

    # Check that the desired metrics were written
    metrics_df = pd.concat(
        [metrics_table.to_pandas() for metrics_table in run.metrics_tables],
        ignore_index=True,
    )

    assert 0 in metrics_df[TRAINING_PHASE], "Expected metrics from during training"
    assert 1 in metrics_df[TRAINING_PHASE], "Expected metrics from after training"

    # Aggregate per-sample metrics should match the output aggregate metrics
    val_after_metrics_df = run.metrics_tables[-1].to_pandas()  # Val metrics after training should be written last

    metrics_top1_accuracy = val_after_metrics_df["top1_accuracy"].mean()
    metrics_top5_accuracy = val_after_metrics_df["top5_accuracy"].mean()

    assert np.isclose(metrics_top1_accuracy, results_3lc.top1)
    assert np.isclose(metrics_top5_accuracy, results_3lc.top5)

    # PaCMAP embedding reduction does not work on macOS; skip the embedding checks there.
    if not PACMAP_BROKEN_ON_MACOS:
        embeddings_column_name = f"embeddings_{settings.image_embeddings_reducer}"
        assert embeddings_column_name in metrics_df.columns, "Expected embeddings column missing"
        assert len(metrics_df[embeddings_column_name][0]) == settings.image_embeddings_dim, (
            "Embeddings dimension mismatch"
        )

    # Test metrics collection only here with the same weights (since there are no readily available pretrained weights
    # for the ten-class case)
    best = results_3lc.save_dir / "weights" / "best.pt"
    best_model = TLCYOLO(best)
    results_dict = best_model.collect(
        data=TASK2DATASET["classify"],
        splits=("train", "val"),
        settings=settings,
    )

    assert results_dict["val"].results_dict == results_ultralytics.results_dict, (
        "Results validation metrics collection onlywith  3LC different from Ultralytics"
    )

    # model.predict() should work and be the same as vanilla ultralytics
    preds_3lc = model_3lc.predict(imgsz=320)
    preds_ultralytics = model_ultralytics.predict(imgsz=320)

    assert preds_3lc[0].probs.top5 == preds_ultralytics[0].probs.top5, "Predictions mismatch"


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_metrics_collection_only(task) -> None:
    # save_json=True would normally route detect/segment validation through COCO/LVIS JSON
    # evaluation, which reads on-disk annotation files that 3LC Tables don't have (previously
    # crashed with KeyError: 'path'). It must instead be disabled with a warning, and collection
    # must run to completion.
    overrides = {"device": "cpu", "save_json": True}
    settings = Settings(project_name=f"test_{task}_collect", run_name=f"test_{task}_collect", collect_loss=True)
    splits = ("train", "val")

    model = TLCYOLO(TASK2MODEL[task])
    with capture_logs(logging.WARNING) as log_messages:
        results_dict = model.collect(data=TASK2DATASET[task], splits=splits, settings=settings, **overrides)
    assert all(results_dict[split] for split in splits), "Metrics collection failed"

    # save_json was unsupported, so a clear warning was emitted and it was disabled for the run.
    assert any("save_json is not supported with 3LC datasets" in msg for msg in log_messages), (
        "Expected warning about save_json not being supported with 3LC datasets"
    )

    run_urls = [results_dict[split].run_url for split in splits]
    assert run_urls[0] == run_urls[1], "Expected same run URL for both splits"

    run = tlc.Run.from_url(run_urls[0])
    metrics_tables = get_metrics_tables_from_run(run)

    metrics_df = pd.concat(
        [m.to_pandas() for m in metrics_tables["default_stream"]],
        ignore_index=True,
    )
    assert "loss" not in metrics_df.columns, "Expected no loss column"
    assert run.status == RUN_STATUS_COMPLETED, "Run status not set to completed after training"
    assert run.description == DEFAULT_COLLECT_RUN_DESCRIPTION, "Description mismatch"
    assert len(metrics_tables[PER_CLASS_METRICS_STREAM_NAME]) == 2, "Expected 2 per-class metrics tables (train, val)"

    per_class_metrics_df = pd.concat(
        [m.to_pandas() for m in metrics_tables[PER_CLASS_METRICS_STREAM_NAME]],
        ignore_index=True,
    )
    assert TRAINING_PHASE not in per_class_metrics_df.columns, "Expected no training phase column"
    assert EPOCH not in per_class_metrics_df.columns, "Expected no epoch column"


@skip_pacmap_on_macos
def test_embeddings_collection() -> None:
    settings = Settings(
        project_name="test_embeddings_collection_project",
        run_name="test_embeddings_collection_run",
        image_embeddings_dim=2,
    )

    overrides = {
        "batch": 8,
        "device": "cpu",
        "workers": 0,
    }

    model = TLCYOLO(TASK2MODEL["detect"])
    model.collect(data="coco128.yaml", splits=("train",), settings=settings, **overrides)

    run = _get_run_from_settings(settings)
    assert len(run.metrics_tables) == 2, "Expected 2 metrics tables to be written"

    embeddings_table = next(
        (metrics_table for metrics_table in run.metrics_tables if "embeddings_pacmap" in metrics_table.columns),
        None,
    )
    assert embeddings_table is not None, "Expected a metrics table with the reduced embeddings column"

    embeddings_column_arrow = embeddings_table.get_column_as_pyarrow_array("embeddings_pacmap")
    embeddings_column_list = embeddings_column_arrow.tolist()

    assert all(len(embedding) == settings.image_embeddings_dim for embedding in embeddings_column_list), (
        "Expected embeddings to be of correct dimension"
    )


def test_train_collection_val_only() -> None:
    task = "classify"
    model_arg = TASK2MODEL[task]
    overrides = {"data": TASK2DATASET[task], "device": "cpu", "epochs": 1, "batch": 4, "imgsz": 224}

    model = TLCYOLO(model_arg)

    settings = Settings(
        collection_val_only=True,
        project_name="test_train_collect_val_only",
        run_name="test_train_collect_val_only",
    )
    model.train(**overrides, settings=settings)

    # Ensure that only validation metrics are collected after training
    run = _get_run_from_settings(settings)
    assert len(run.metrics_tables) == 1, "Expected only validation metrics to be collected after training"


@pytest.mark.parametrize("task", ["classify", "detect", "segment", "obb"])
def test_train_collection_disabled(task: str) -> None:
    model_arg = TASK2MODEL[task]
    overrides = {"data": TASK2DATASET[task], "device": "cpu", "epochs": 1, "batch": 4, "imgsz": 224}

    model = TLCYOLO(model_arg)

    settings = Settings(
        collection_disable=True,
        project_name=f"test_train_collection_disabled_{task}",
        run_name=f"test_train_collection_disabled_{task}",
    )
    model.train(**overrides, settings=settings)

    # classify never writes per-class tables, so only detect/segment/obb can catch a gating regression here.
    run = _get_run_from_settings(settings)
    assert len(run.metrics_tables) == 0, "Expected no metrics tables to be written"


def test_invalid_tables() -> None:
    # Test that an error is raised if the tables are not formatted as desired
    for model_arg in TASK2MODEL.values():
        model = TLCYOLO(model_arg)
        table = tlc.Table.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(ValueError):
            model.train(tables={"train": table, "val": table})


def test_table_resolving() -> None:
    # Check that repeated runs with 'data' resolve to the same tables, or the latest
    settings = Settings(project_name="test_table_resolving")
    trainer = TASK2TRAINER["detect"](
        overrides={"data": TASK2DATASET["detect"], "model": TASK2MODEL["detect"], "settings": settings},
    )

    # Create initial tables
    train_table = trainer.data["train"]

    # Create an edited version of the train table
    train_table_edited = NullOverlay(
        train_table.url.create_sibling("peter").create_unique(),
        input_table_url=train_table,
    )

    # A new trainer should now use the edited table since it gets latest
    new_trainer = TASK2TRAINER["detect"](
        overrides={"data": TASK2DATASET["detect"], "model": TASK2MODEL["detect"], "settings": settings},
    )
    assert new_trainer.data["train"].url == train_table_edited.url, "Table not resolved correctly"

    # A new trainer should not be able to take the tables directly
    tables = {"train": train_table_edited.url, "val": new_trainer.data.get("val") or new_trainer.data["test"].url}
    trainer_from_tables = TASK2TRAINER["detect"](
        overrides={
            "tables": tables,
            "model": TASK2MODEL["detect"],
            "settings": settings,
        },
    )
    assert trainer_from_tables.data["train"].url == train_table_edited.url, (
        "Table passed directly not resolved correctly"
    )


def test_seg_table_checker() -> None:
    settings = Settings(project_name="test_seg_table_checker")
    trainer = TASK2TRAINER["segment"](
        overrides={"data": TASK2DATASET["segment"], "model": TASK2MODEL["segment"], "settings": settings}
    )

    # A table from a yolo dataset is valid
    check_seg_table(trainer.data["train"], "image", "segmentations")

    # The same data in a new table, but backed by a row cache, is also valid
    overlay_table_url = NullOverlay(
        url=trainer.data["train"].url.create_sibling("overlay_table"), input_table_url=trainer.data["train"]
    ).write_to_url()
    overlay_table = tlc.Table.from_url(overlay_table_url)
    check_seg_table(overlay_table, "image", "segmentations")

    # A table with a wrong schema should be invalid
    invalid_schema_seg_table = tlc.Table.from_dict(
        {"image": [1, 2, 3], "segmentations": [4, 5, 6]},
        project_name=settings.project_name,
        dataset_name="test_seg_table_checker",
        table_name="invalid_seg_table",
    )
    with pytest.raises(ValueError, match="Validation failed"):
        check_seg_table(invalid_schema_seg_table, "image", "segmentations")


def _instance_task_config(task: str, width: int, height: int) -> dict:
    """Per-task building blocks for constructing instance-task tables (detect/segment/obb/pose).

    Returns the column name, schema, label path, dataset ``names``, extra ``data`` kwargs, the
    annotation key to compare on, a labeled row, an unlabeled (empty) row, and copies of both with
    their stored image dimensions zeroed out. The dimensions live in different fields per task:
    segmentation stores ``image_height``/``image_width``, while the box/keypoint tasks carry them
    as the coordinate-space bounds ``x_max``/``y_max`` (annotations are stored as absolute pixels).
    """
    from tlc.constants import IMAGE_HEIGHT, IMAGE_WIDTH, X_MAX, Y_MAX
    from tlc.data_types import BoundingBoxes2D, Keypoints2D, OrientedBoundingBoxes2D, SegmentationPolygons

    if task == "detect":
        column = "bbs"
        schema = BoundingBoxes2D.schema(classes={0: "object"})
        labeled_row = BoundingBoxes2D(
            bounding_boxes=[[width / 2, height / 2, width / 4, height / 4]],
            bounding_box_format="cxywh",
            image_width=width,
            image_height=height,
            labels=[0],
        ).to_row()
        empty_template = BoundingBoxes2D.create_empty(image_width=width, image_height=height).to_row()
        label_container, label_path = "instances_additional_data", "bbs.instances_additional_data.label"
        names, data_extra, compare_key = {0: "object"}, {}, "bboxes"
        dim_zero = {X_MAX: 0.0, Y_MAX: 0.0}
    elif task == "segment":
        column = "segmentations"
        schema = SegmentationPolygons.schema(classes={0: "object"})
        labeled_row = SegmentationPolygons(
            image_width=width,
            image_height=height,
            polygons=[[5.0, 5.0, 40.0, 5.0, 40.0, 30.0, 5.0, 30.0]],
            labels=[0],
        ).to_row()
        empty_template = SegmentationPolygons.create_empty(image_width=width, image_height=height).to_row()
        label_container, label_path = "instance_properties", "segmentations.instance_properties.label"
        names, data_extra, compare_key = {0: "object"}, {}, "segments"
        dim_zero = {IMAGE_HEIGHT: 0, IMAGE_WIDTH: 0}
    elif task == "obb":
        column = "obb"
        schema = OrientedBoundingBoxes2D.schema(classes={0: "object"})
        labeled_row = OrientedBoundingBoxes2D(
            image_width=width,
            image_height=height,
            obbs=[[width / 2, height / 2, width / 4, height / 4, 0.0]],
            labels=[0],
        ).to_row()
        empty_template = OrientedBoundingBoxes2D.create_empty(image_width=width, image_height=height).to_row()
        label_container, label_path = "instances_additional_data", "obb.instances_additional_data.label"
        names, data_extra, compare_key = {0: "object"}, {}, "segments"
        dim_zero = {X_MAX: 0.0, Y_MAX: 0.0}
    elif task == "pose":
        column = "pose"
        schema = Keypoints2D.schema(num_keypoints=2, classes={0: "person"})
        labeled_row = Keypoints2D(
            image_width=width,
            image_height=height,
            keypoints=[[[10, 10], [20, 20]]],
            keypoint_visibilities=[[2, 2]],
            bounding_boxes=[[5, 5, 25, 25]],
            labels=[0],
        ).to_row()
        empty_template = Keypoints2D.create_empty(image_width=width, image_height=height).to_row()
        label_container, label_path = "instances_additional_data", "pose.instances_additional_data.label"
        names, data_extra, compare_key = {0: "person"}, {"kpt_shape": [2, 3]}, "keypoints"
        dim_zero = {X_MAX: 0.0, Y_MAX: 0.0}
    else:
        raise ValueError(f"Unknown task: {task}")

    # A real table writer fills the label sub-column with an empty list for unlabeled rows
    # (rather than the bare ``{}`` that ``create_empty().to_row()`` produces), so mirror that here
    # to keep the column schema consistent across rows.
    empty_row = {**empty_template, label_container: {"label": []}}

    return {
        "column": column,
        "schema": schema,
        "label_path": label_path,
        "names": names,
        "data_extra": data_extra,
        "compare_key": compare_key,
        "labeled_row": labeled_row,
        "empty_row": empty_row,
        "zeroed_row": {**labeled_row, **dim_zero},
        "empty_zeroed_row": {**empty_row, **dim_zero},
    }


def _build_task_dataset(task: str, config: dict, rows: list[dict], table_name: str, class_map: dict | None = None):
    """Build a 3LC YOLO dataset for ``task`` from a single-column table of ``rows``."""
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    from tlc_ultralytics.detect.utils import build_tlc_yolo_dataset

    # Labeled row is passed first by callers so the column schema is inferred from a fully
    # populated row.
    table = tlc.Table.from_dict(
        {"image": [str(DUMMY_IMAGE_FILE)] * len(rows), config["column"]: rows},
        schema={config["column"]: config["schema"]},
        project_name="test_instance_tasks",
        dataset_name=task,
        table_name=table_name,
        if_exists="overwrite",
    )
    cfg = get_cfg(DEFAULT_CFG, overrides={"task": task, "imgsz": 64, "rect": False})
    return build_tlc_yolo_dataset(
        cfg,
        table,
        batch=1,
        data={"channels": 3, "names": config["names"], "nc": 1, **config["data_extra"]},
        mode="val",
        class_map=class_map,
        image_column_name="image",
        label_column_name=config["label_path"],
    )


@pytest.mark.parametrize("task", ["detect", "segment", "obb", "pose"])
def test_missing_image_dimensions_fallback(task: str) -> None:
    """A Table that stores non-positive image dimensions must not crash. The dataset should fall
    back to reading the real image size from disk, recover the annotations against that size, and
    warn once. Regression test for tables authored without valid image dimensions.
    """
    from tlc.helpers import ImageHelper

    # Real image on disk; annotations are encoded against its true size.
    height, width = ImageHelper.get_exif_image_dimensions(str(DUMMY_IMAGE_FILE))
    config = _instance_task_config(task, width, height)

    # Reference dataset: correct dimensions stored.
    reference = _build_task_dataset(task, config, [config["labeled_row"]], "dims_good")

    # Broken dataset: image dimensions zeroed out (valid annotations, but no usable size metadata).
    with capture_logs(logging.WARNING) as log_messages:
        recovered = _build_task_dataset(task, config, [config["zeroed_row"]], "dims_zeroed")

    # The fallback warned about the missing dimensions...
    assert any("non-positive image dimensions" in msg for msg in log_messages), (
        f"Expected a warning about missing image dimensions, got: {log_messages}"
    )

    ref_label, got_label = reference.labels[0], recovered.labels[0]

    # ...recovered the real image shape...
    assert got_label["shape"] == (height, width) == ref_label["shape"]

    # ...and reconstructed identical, normalized ([0, 1]) annotations against that size.
    ref_val, got_val = ref_label[config["compare_key"]], got_label[config["compare_key"]]
    pairs = zip(ref_val, got_val, strict=True) if isinstance(ref_val, list) else [(ref_val, got_val)]
    if isinstance(ref_val, list):
        assert len(got_val) == len(ref_val) >= 1
    for ref_arr, got_arr in pairs:
        np.testing.assert_allclose(got_arr, ref_arr, atol=1e-6)
        assert np.max(got_arr) <= 1.0 + 1e-6
    # The normalized boxes are also recovered identically for every task.
    np.testing.assert_allclose(got_label["bboxes"], ref_label["bboxes"], atol=1e-6)


@pytest.mark.parametrize("rotation_deg", [90.0, 60.0, 30.0, 0.0])
def test_obb_not_squashed_on_non_square_image(rotation_deg: float) -> None:
    """Oriented boxes read from a 3LC table must keep their shape on non-square images.

    The dataset stores each box as normalized corner points. Normalizing the box's local
    `size_x`/`size_y` by image width/height *before* applying the rotation squashes tall boxes
    toward square on non-square images: the two extents are scaled by different factors and the
    rotation then mixes the axes. Corners must be built in pixel space and normalized afterwards.
    """
    from tlc.data_types import OrientedBoundingBoxes2D

    # Deliberately non-square, and independent of the dummy image's real size: the stored
    # dimensions are positive so the dataset uses them directly.
    width, height = 1920, 1080
    cx, cy, size_x, size_y = 960.0, 540.0, 438.0, 167.0
    rotation = np.deg2rad(rotation_deg)

    config = _instance_task_config("obb", width, height)
    tall_row = OrientedBoundingBoxes2D(
        image_width=width,
        image_height=height,
        obbs=[[cx, cy, size_x, size_y, rotation]],
        labels=[0],
    ).to_row()

    dataset = _build_task_dataset("obb", config, [tall_row], f"obb_squash_{int(rotation_deg)}")

    label = dataset.labels[0]
    assert label["shape"] == (height, width)

    # The dataset returns normalized corner points; scale them back to pixel space
    (segment,) = label["segments"]
    corners_px = np.asarray(segment, dtype=np.float64).copy()
    corners_px[:, 0] *= width
    corners_px[:, 1] *= height

    # Recover the rotated box and compare its side lengths to what was stored
    (_, _), (rec_w, rec_h), _ = cv2.minAreaRect(corners_px.astype(np.float32))
    recovered = sorted([rec_w, rec_h])
    expected = sorted([size_x, size_y])
    np.testing.assert_allclose(
        recovered,
        expected,
        atol=1.0,
        err_msg=f"OBB squashed: stored {expected} px but read back {recovered} px",
    )


@pytest.mark.parametrize("task", ["detect", "segment", "obb", "pose"])
def test_unlabeled_row(task: str) -> None:
    """An unlabeled row (no instances) must not crash. The instance dataclasses come back with
    ``labels=None`` for an empty row, which previously raised ``TypeError: 'NoneType' object is
    not iterable`` (segment) or ``AttributeError: 'NoneType' object has no attribute 'astype'``
    (obb/pose). The dataset should instead produce a label with zero instances. Regression test
    for training on revision tables that still have some unlabeled samples.
    """
    from tlc.helpers import ImageHelper

    height, width = ImageHelper.get_exif_image_dimensions(str(DUMMY_IMAGE_FILE))
    config = _instance_task_config(task, width, height)

    dataset = _build_task_dataset(task, config, [config["labeled_row"], config["empty_row"]], "unlabeled")

    # The unlabeled row yields zero instances; the labeled row is unaffected.
    labeled_label, empty_label = dataset.labels[0], dataset.labels[1]
    assert empty_label["cls"].shape == (0, 1)
    assert empty_label["bboxes"].shape == (0, 4)
    assert empty_label["shape"] == (height, width)
    assert labeled_label["cls"].shape == (1, 1)
    if task == "segment":
        assert empty_label["segments"] == []
    elif task == "pose":
        assert empty_label["keypoints"].shape == (0, 2, 3)


@pytest.mark.parametrize("task", ["detect", "segment", "obb", "pose"])
def test_unlabeled_row_with_missing_dimensions(task: str) -> None:
    """The realistic revision-table case: a row that is both unlabeled *and* has non-positive
    image dimensions - exactly what ``create_empty()`` produces by default. The dimension fallback
    and the empty-label handling must compose, so the row decodes to zero instances at the real
    image size and warns once.
    """
    from tlc.helpers import ImageHelper

    height, width = ImageHelper.get_exif_image_dimensions(str(DUMMY_IMAGE_FILE))
    config = _instance_task_config(task, width, height)

    with capture_logs(logging.WARNING) as log_messages:
        dataset = _build_task_dataset(
            task, config, [config["labeled_row"], config["empty_zeroed_row"]], "unlabeled_zeroed"
        )

    assert any("non-positive image dimensions" in msg for msg in log_messages), (
        f"Expected a warning about missing image dimensions, got: {log_messages}"
    )
    empty_label = dataset.labels[1]
    assert empty_label["cls"].shape == (0, 1)
    assert empty_label["bboxes"].shape == (0, 4)
    assert empty_label["shape"] == (height, width)


@pytest.mark.parametrize("task", ["detect", "segment", "obb", "pose"])
def test_label_not_in_value_map_raises(task: str) -> None:
    """A row whose annotation references a class id absent from the label column's value map must
    fail with an actionable `ValueError` instead of a bare `KeyError`. The value map declares
    only class id 0, but the row carries label 1.
    """
    from tlc.constants import IMAGE_HEIGHT, IMAGE_WIDTH, X_MAX, Y_MAX  # noqa: F401
    from tlc.data_types import BoundingBoxes2D, Keypoints2D, OrientedBoundingBoxes2D, SegmentationPolygons
    from tlc.helpers import ImageHelper

    height, width = ImageHelper.get_exif_image_dimensions(str(DUMMY_IMAGE_FILE))
    config = _instance_task_config(task, width, height)

    # An otherwise-valid row, but with a class id (1) that is not in the value map ({0: ...}).
    if task == "detect":
        bad_row = BoundingBoxes2D(
            bounding_boxes=[[width / 2, height / 2, width / 4, height / 4]],
            bounding_box_format="cxywh",
            image_width=width,
            image_height=height,
            labels=[1],
        ).to_row()
    elif task == "segment":
        bad_row = SegmentationPolygons(
            image_width=width,
            image_height=height,
            polygons=[[5.0, 5.0, 40.0, 5.0, 40.0, 30.0, 5.0, 30.0]],
            labels=[1],
        ).to_row()
    elif task == "obb":
        bad_row = OrientedBoundingBoxes2D(
            image_width=width,
            image_height=height,
            obbs=[[width / 2, height / 2, width / 4, height / 4, 0.0]],
            labels=[1],
        ).to_row()
    else:  # pose
        bad_row = Keypoints2D(
            image_width=width,
            image_height=height,
            keypoints=[[[10, 10], [20, 20]]],
            keypoint_visibilities=[[2, 2]],
            bounding_boxes=[[5, 5, 25, 25]],
            labels=[1],
        ).to_row()

    # The value map declares only class id 0, so the class map maps 0 -> 0.
    class_map = {0: 0}

    with pytest.raises(ValueError, match=r"class id 1.*not present in the column's value map") as excinfo:
        _build_task_dataset(task, config, [bad_row], "label_not_in_value_map", class_map=class_map)

    # The error names the offending id, the row and column it came from, and the human-readable
    # value map (id -> name), not the raw class map.
    message = str(excinfo.value)
    assert f"column '{config['column']}'" in message
    assert "Row 0" in message  # the example id of the offending row
    assert next(iter(config["names"].values())) in message  # the class name, e.g. "object" / "person"


@pytest.mark.parametrize("task", ["detect", "segment", "obb", "pose"])
def test_class_map_is_applied(task: str) -> None:
    """The dataset must translate raw 3LC class ids to their contiguous training indices via the
    class map, for every instance task. A non-contiguous id (5) is mapped to training index 1, so
    the emitted `cls` must be 1 (mapped), never 5 (raw). Regression test for cluster A — in
    particular OBB previously ignored the class map entirely (emitting the raw id), and pose
    silently passed unknown ids through.
    """
    from tlc.data_types import BoundingBoxes2D, Keypoints2D, OrientedBoundingBoxes2D, SegmentationPolygons
    from tlc.helpers import ImageHelper

    height, width = ImageHelper.get_exif_image_dimensions(str(DUMMY_IMAGE_FILE))
    config = _instance_task_config(task, width, height)

    # A valid row carrying a non-contiguous 3LC class id (5).
    if task == "detect":
        row = BoundingBoxes2D(
            bounding_boxes=[[width / 2, height / 2, width / 4, height / 4]],
            bounding_box_format="cxywh",
            image_width=width,
            image_height=height,
            labels=[5],
        ).to_row()
    elif task == "segment":
        row = SegmentationPolygons(
            image_width=width,
            image_height=height,
            polygons=[[5.0, 5.0, 40.0, 5.0, 40.0, 30.0, 5.0, 30.0]],
            labels=[5],
        ).to_row()
    elif task == "obb":
        row = OrientedBoundingBoxes2D(
            image_width=width,
            image_height=height,
            obbs=[[width / 2, height / 2, width / 4, height / 4, 0.0]],
            labels=[5],
        ).to_row()
    else:  # pose
        row = Keypoints2D(
            image_width=width,
            image_height=height,
            keypoints=[[[10, 10], [20, 20]]],
            keypoint_visibilities=[[2, 2]],
            bounding_boxes=[[5, 5, 25, 25]],
            labels=[5],
        ).to_row()

    # 3LC class id 5 maps to contiguous training index 1.
    class_map = {5: 1}
    dataset = _build_task_dataset(task, config, [row], "class_map_applied", class_map=class_map)

    cls = dataset.labels[0]["cls"]
    assert cls.shape == (1, 1)
    assert int(cls[0, 0]) == 1, f"expected mapped index 1, got {cls[0, 0]} (class map was not applied)"


def test_legacy_bb_table_default_label_path() -> None:
    # A legacy-format (bb_list) detection table should work with the default label column
    # name, which points at the new-format path (bbs.instances_additional_data.label).
    from tlc.schemas._annotations._bounding_box_list_schema import _BoundingBoxListSchema

    from tlc_ultralytics.constants import DETECTION_LABEL_COLUMN_NAME
    from tlc_ultralytics.detect.utils import check_det_table
    from tlc_ultralytics.utils.dataset import get_value_map_from_table, resolve_label_value_path

    classes = {0: tlc.schemas.MapElement("cat"), 1: tlc.schemas.MapElement("dog")}
    writer = tlc.TableWriter(
        table_name="legacy_bbs",
        dataset_name="test_legacy_bb_table",
        project_name="test_legacy_bb_table",
        schema={"image": tlc.schemas.ImageSchema(), "bbs": _BoundingBoxListSchema(classes=classes)},
    )
    writer.add_row(
        {
            "image": str(DUMMY_IMAGE_FILE),
            "bbs": {
                "image_width": 100,
                "image_height": 100,
                "bb_list": [{"x0": 10.0, "y0": 10.0, "x1": 50.0, "y1": 50.0, "label": 1, "segmentation": []}],
            },
        }
    )
    table = writer.finalize()

    # The default (new-format) label path resolves to the legacy path
    assert resolve_label_value_path(table, DETECTION_LABEL_COLUMN_NAME) == "bbs.bb_list.label"

    # Table check and value map lookup work with the default label column name
    check_det_table(table, "image", DETECTION_LABEL_COLUMN_NAME)
    value_map = get_value_map_from_table(table, DETECTION_LABEL_COLUMN_NAME, "detect")
    assert value_map is not None
    assert {k: v.internal_name for k, v in value_map.items()} == {0: "cat", 1: "dog"}

    # An explicitly provided legacy path also works and is left untouched
    assert resolve_label_value_path(table, "bbs.bb_list.label") == "bbs.bb_list.label"
    check_det_table(table, "image", "bbs.bb_list.label")


def _make_annotation_table(task: str, column_name: str, table_name: str) -> tlc.Table:
    """Build a single-row 3LC table for `task` whose annotation column is named `column_name`.

    Covers all four annotation tasks so column-resolution behaviour can be exercised uniformly.
    """
    from tlc.data_types import BoundingBoxes2D, Keypoints2D, OrientedBoundingBoxes2D, SegmentationPolygons

    classes = {0: tlc.schemas.MapElement("object")}
    if task == "detect":
        row = BoundingBoxes2D(
            bounding_boxes=[[50.0, 50.0, 25.0, 25.0]],
            bounding_box_format="cxywh",
            image_width=100,
            image_height=100,
            labels=[0],
        ).to_row()
        schema = BoundingBoxes2D.schema(classes=classes)
    elif task == "segment":
        row = SegmentationPolygons(
            image_width=100,
            image_height=100,
            polygons=[[5.0, 5.0, 40.0, 5.0, 40.0, 30.0, 5.0, 30.0]],
            labels=[0],
        ).to_row()
        schema = SegmentationPolygons.schema(classes=classes)
    elif task == "pose":
        row = Keypoints2D(
            keypoints=np.array([[[10, 10], [20, 20]]], dtype=np.float32),
            keypoint_visibilities=np.array([[2, 2]]),
            labels=np.array([0]),
            bounding_boxes=np.array([[5, 5, 30, 30]], dtype=np.float32),
            bounding_box_format="xyxy",
            image_width=100,
            image_height=100,
            normalized=False,
        ).to_row()
        schema = Keypoints2D.schema(num_keypoints=2, classes=classes)
    elif task == "obb":
        row = OrientedBoundingBoxes2D(
            obbs=np.array([[50, 50, 20, 10, 0.3]], dtype=np.float32),
            labels=np.array([0]),
            image_width=100,
            image_height=100,
            normalized=False,
        ).to_row()
        schema = OrientedBoundingBoxes2D.schema(classes=classes)
    else:
        raise ValueError(f"Unsupported task: {task}")

    return tlc.Table.from_dict(
        {"image": [str(DUMMY_IMAGE_FILE)], column_name: [row]},
        schema={column_name: schema},
        project_name="test_annotation_column_resolution",
        dataset_name=task,
        table_name=table_name,
        if_exists="overwrite",
    )


def _make_detection_table_with_column(column_name: str, table_name: str) -> tlc.Table:
    """Build a single-row 3LC detection table whose bounding-box column is `column_name`."""
    return _make_annotation_table("detect", column_name, table_name)


# Per-task fixtures for the column-resolution cross-product: the non-default column name to author the
# table with, the task default label path, the resolved path expected from inference (full value path
# for detect/segment, root column for pose/obb), and the annotation type name surfaced in messages.
_ANNOTATION_RESOLUTION_CASES = {
    "detect": ("boxes", DETECTION_LABEL_COLUMN_NAME, "boxes.instances_additional_data.label", "BOUNDING_BOXES"),
    "segment": ("masks", SEGMENTATION_LABEL_COLUMN_NAME, "masks.instance_properties.label", "SEGMENTATION"),
    "pose": ("kpts", POSE_LABEL_COLUMN_NAME, "kpts", "KEYPOINTS"),
    "obb": ("my_obb", OBB_LABEL_COLUMN_NAME, "my_obb", "ORIENTED_BOUNDING_BOXES"),
}


def test_detection_non_default_bbox_column_end_to_end() -> None:
    # A detection table whose bounding-box column is NOT named "bbs" should work end-to-end:
    # check_tlc_dataset must infer the column, propagate the resolved label path into Settings,
    # build a value map, and the dataset must resolve labels against the actual column.
    from tlc_ultralytics.detect.utils import build_tlc_yolo_dataset, check_det_table
    from tlc_ultralytics.utils.dataset import check_tlc_dataset, resolve_annotation_label_path

    table = _make_detection_table_with_column("boxes", "boxes_table")

    # The resolver maps the default `bbs...` path — and None (no configured column) — to the actual
    # `boxes...` path via structural inference.
    assert (
        resolve_annotation_label_path(table, DETECTION_LABEL_COLUMN_NAME, "detect")
        == "boxes.instances_additional_data.label"
    )
    assert resolve_annotation_label_path(table, None, "detect") == "boxes.instances_additional_data.label"

    # The checker accepts the table even though the configured/default column is "bbs".
    check_det_table(table, "image", DETECTION_LABEL_COLUMN_NAME)

    # check_tlc_dataset propagates the inferred label path into the provided Settings object so
    # downstream dataset construction uses the right column.
    settings = Settings(project_name="test_detection_non_default_bbox_column")
    settings.label_column_name = DETECTION_LABEL_COLUMN_NAME
    data = check_tlc_dataset(
        data="ignored",
        tables={"val": table},
        image_column_name="image",
        label_column_name=settings.label_column_name,
        splits=("val",),
        task="detect",
        settings=settings,
    )
    assert settings.label_column_name == "boxes.instances_additional_data.label"
    assert data["names"] == {0: "object"}

    # Building the dataset with the propagated label path resolves labels against the "boxes" column.
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    cfg = get_cfg(DEFAULT_CFG, overrides={"task": "detect", "imgsz": 64, "rect": False})
    dataset = build_tlc_yolo_dataset(
        cfg,
        table,
        batch=1,
        data=data,
        mode="val",
        class_map=data["3lc_class_to_range"],
        image_column_name="image",
        label_column_name=settings.label_column_name,
    )
    label = dataset.labels[0]
    assert label["cls"].shape == (1, 1)
    assert label["bboxes"].shape == (1, 4)


def test_detection_table_without_bounding_boxes_precise_message() -> None:
    # A table with segmentation (but no bounding-box) annotations should raise a precise message
    # naming the annotation type present and the task to use instead, plus the columns present.
    from tlc.data_types import SegmentationPolygons

    from tlc_ultralytics.detect.utils import check_det_table
    from tlc_ultralytics.utils.dataset import resolve_annotation_label_path

    seg_row = SegmentationPolygons(
        image_width=100,
        image_height=100,
        polygons=[[5.0, 5.0, 40.0, 5.0, 40.0, 30.0, 5.0, 30.0]],
        labels=[0],
    ).to_row()
    table = tlc.Table.from_dict(
        {"image": [str(DUMMY_IMAGE_FILE)], "segmentations": [seg_row]},
        schema={"segmentations": SegmentationPolygons.schema(classes={0: tlc.schemas.MapElement("object")})},
        project_name="test_detection_no_bboxes",
        dataset_name="d",
        table_name="seg_table",
        if_exists="overwrite",
    )

    with pytest.raises(ValueError, match="SEGMENTATION annotations in column 'segmentations'"):
        resolve_annotation_label_path(table, DETECTION_LABEL_COLUMN_NAME, "detect")

    # The same precise message surfaces through the checker, listing the columns present.
    with pytest.raises(ValueError, match=r"Columns present:.*'segmentations'"):
        check_det_table(table, "image", DETECTION_LABEL_COLUMN_NAME)

    # A table with no annotation columns at all gets the "no annotation columns" message.
    no_ann = tlc.Table.from_dict(
        {"a": [1], "b": [2]},
        project_name="test_detection_no_bboxes",
        dataset_name="d",
        table_name="no_ann_table",
        if_exists="overwrite",
    )
    with pytest.raises(ValueError, match="no annotation columns were found"):
        resolve_annotation_label_path(no_ann, DETECTION_LABEL_COLUMN_NAME, "detect")


def test_segment_non_default_column_and_cross_task_message() -> None:
    # The same resolution applies to segmentation: a differently-named segmentation column is
    # inferred (from None or the default), and a detection table yields a precise message pointing
    # at the bounding-box task.
    from tlc.data_types import SegmentationPolygons

    from tlc_ultralytics.segment.utils import check_seg_table
    from tlc_ultralytics.utils.dataset import resolve_annotation_label_path

    seg_row = SegmentationPolygons(
        image_width=100,
        image_height=100,
        polygons=[[5.0, 5.0, 40.0, 5.0, 40.0, 30.0, 5.0, 30.0]],
        labels=[0],
    ).to_row()
    table = tlc.Table.from_dict(
        {"image": [str(DUMMY_IMAGE_FILE)], "masks": [seg_row]},
        schema={"masks": SegmentationPolygons.schema(classes={0: tlc.schemas.MapElement("object")})},
        project_name="test_segment_non_default_column",
        dataset_name="d",
        table_name="masks_table",
        if_exists="overwrite",
    )

    # Both the default `segmentations...` path and None resolve to the actual `masks...` path.
    assert resolve_annotation_label_path(table, SEGMENTATION_LABEL_COLUMN_NAME, "segment") == (
        "masks.instance_properties.label"
    )
    assert resolve_annotation_label_path(table, None, "segment") == "masks.instance_properties.label"

    # The checker accepts the table even though the configured/default column is "segmentations".
    check_seg_table(table, "image", SEGMENTATION_LABEL_COLUMN_NAME)
    check_seg_table(table, "image", None)

    # A detection table routed to the segment task names bounding boxes and the right task to use.
    det_table = _make_detection_table_with_column("bbs", "bbs_for_segment")
    with pytest.raises(ValueError, match="BOUNDING_BOXES annotations in column 'bbs'"):
        resolve_annotation_label_path(det_table, SEGMENTATION_LABEL_COLUMN_NAME, "segment")


@pytest.mark.parametrize("task", ["detect", "segment", "pose", "obb"])
def test_resolve_infers_non_default_column(task: str) -> None:
    # Path 1+4: a non-default annotation column is inferred from both the task default and None,
    # uniformly across all four annotation tasks, and the per-task checker accepts it unconfigured.
    from tlc_ultralytics.utils.dataset import get_dataset_functions, resolve_annotation_label_path

    column, default, expected, _ = _ANNOTATION_RESOLUTION_CASES[task]
    table = _make_annotation_table(task, column, f"{task}_infer")

    assert resolve_annotation_label_path(table, default, task) == expected
    assert resolve_annotation_label_path(table, None, task) == expected

    _, table_checker = get_dataset_functions(task)
    table_checker(table, "image", None)
    table_checker(table, "image", default)


@pytest.mark.parametrize("task", ["detect", "segment", "pose", "obb"])
def test_resolve_honors_existing_named_column(task: str) -> None:
    # Path 2+3: an explicitly-named existing column (a bare root) is honored — completed to a full
    # value path for detect/segment — without falling back to inference.
    from tlc_ultralytics.utils.dataset import resolve_annotation_label_path

    column, _, expected, _ = _ANNOTATION_RESOLUTION_CASES[task]
    table = _make_annotation_table(task, column, f"{task}_honor")

    resolved = resolve_annotation_label_path(table, column, task)
    assert resolved == expected
    assert resolved.split(".")[0] == column


@pytest.mark.parametrize(
    "task,other_task",
    [("detect", "segment"), ("segment", "detect"), ("pose", "detect"), ("obb", "segment")],
)
def test_resolve_cross_task_message(task: str, other_task: str) -> None:
    # Path 5a: a table whose only annotation column is a different type raises a precise message
    # naming that type and column, both directly and through the task checker.
    from tlc_ultralytics.utils.dataset import get_dataset_functions, resolve_annotation_label_path

    other_column, _, _, other_type = _ANNOTATION_RESOLUTION_CASES[other_task]
    _, default, _, _ = _ANNOTATION_RESOLUTION_CASES[task]
    table = _make_annotation_table(other_task, other_column, f"{task}_from_{other_task}")

    with pytest.raises(ValueError, match=f"{other_type} annotations in column '{other_column}'"):
        resolve_annotation_label_path(table, default, task)

    _, table_checker = get_dataset_functions(task)
    with pytest.raises(ValueError, match=other_column):
        table_checker(table, "image", None)


def test_resolve_multiple_annotation_columns_message() -> None:
    # Path 5b: a table with several annotation columns, none of the required type, must report that
    # — not the misleading "no annotation columns were found".
    from tlc.data_types import Keypoints2D, SegmentationPolygons

    from tlc_ultralytics.utils.dataset import resolve_annotation_label_path

    seg_row = SegmentationPolygons(
        image_width=100, image_height=100, polygons=[[5.0, 5.0, 40.0, 5.0, 40.0, 30.0, 5.0, 30.0]], labels=[0]
    ).to_row()
    kp_row = Keypoints2D(
        keypoints=np.array([[[10, 10], [20, 20]]], dtype=np.float32),
        keypoint_visibilities=np.array([[2, 2]]),
        labels=np.array([0]),
        bounding_boxes=np.array([[5, 5, 30, 30]], dtype=np.float32),
        bounding_box_format="xyxy",
        image_width=100,
        image_height=100,
        normalized=False,
    ).to_row()
    table = tlc.Table.from_dict(
        {"image": [str(DUMMY_IMAGE_FILE)], "seg": [seg_row], "kp": [kp_row]},
        schema={
            "seg": SegmentationPolygons.schema(classes={0: tlc.schemas.MapElement("object")}),
            "kp": Keypoints2D.schema(num_keypoints=2, classes={0: tlc.schemas.MapElement("object")}),
        },
        project_name="test_annotation_column_resolution",
        dataset_name="multi",
        table_name="multi_ann",
        if_exists="overwrite",
    )

    with pytest.raises(ValueError, match="multiple annotation columns, but none are the BOUNDING_BOXES"):
        resolve_annotation_label_path(table, DETECTION_LABEL_COLUMN_NAME, "detect")


def test_explicit_missing_label_column_warns_but_none_is_quiet() -> None:
    # A non-None label_column_name whose root is absent warns (likely a typo or stale config) before
    # falling back to inference; deferred resolution (None) is the normal path and stays quiet.
    from tlc_ultralytics.utils.dataset import resolve_annotation_label_path

    table = _make_annotation_table("detect", "boxes", "warn_case")

    with capture_logs(logging.WARNING) as messages:
        resolved = resolve_annotation_label_path(table, "nonexistent_column", "detect")
    assert resolved == "boxes.instances_additional_data.label"
    assert any("was not found in the table" in m for m in messages), messages

    with capture_logs(logging.WARNING) as messages:
        resolve_annotation_label_path(table, None, "detect")
    assert not any("was not found in the table" in m for m in messages), messages


def test_sampling_weights() -> None:
    # Test that sampling weights are correctly applied, with worker processes enabled
    settings = Settings(project_name="test_sampling_weights", sampling_weights=True)
    trainer = TASK2TRAINER["detect"](
        overrides={
            "data": TASK2DATASET["detect"],
            "model": TASK2MODEL["detect"],
            "settings": settings,
            "workers": 4,
        },
    )

    # Model is normally set up in train(); build_dataset only needs its stride, so use a stub.
    trainer.model = stub_model_with_stride()

    epochs = 1000

    # Create edited table where one sample has weight increased to 2
    train_table = trainer.data["train"]
    edited_table = EditedTable(
        url=train_table.url.create_sibling("jonas"),
        input_table_url=train_table,
        edits={train_table.weights_column_name: {"runs_and_values": [[0], 2.0]}},
    )

    dataloader = trainer.get_dataloader(edited_table, batch_size=2, rank=-1)

    sampled_example_ids = []
    for _epoch in range(epochs):
        for batch in dataloader:
            sampled_example_ids.extend(batch["example_id"])

    # Check other samples are sampled within [0.45, 0.55] of the time of the first
    counts = np.bincount(sampled_example_ids)
    relative_counts = counts[1:] / counts[0]
    assert np.allclose(
        relative_counts,
        np.full_like(relative_counts, 0.5),
        atol=0.05,
    ), f"First sample should be sampled twice as often as others, got {counts}"
    assert len(sampled_example_ids) == len(edited_table) * epochs, "Expected no change in the number of samples"


def test_exclude_zero_weight_training() -> None:
    # Test that sampling weights are correctly applied, with worker processes enabled
    settings = Settings(project_name="test_exclude_zero_weight_training", exclude_zero_weight_training=True)
    trainer = TASK2TRAINER["detect"](
        overrides={
            "data": TASK2DATASET["detect"],
            "model": TASK2MODEL["detect"],
            "settings": settings,
            "workers": 4,
        },
    )

    # Model is normally set up in train(); build_dataset only needs its stride, so use a stub.
    trainer.model = stub_model_with_stride()

    # Create edited table where one sample has weight increased to 2
    train_table = trainer.data["train"]
    edited_table = EditedTable(
        url=train_table.url.create_sibling("jonas"),
        input_table_url=train_table,
        edits={train_table.weights_column_name: {"runs_and_values": [[0], 0.0]}},
    )

    dataloader = trainer.get_dataloader(edited_table, batch_size=2, rank=-1)
    sampled_example_ids = []
    for batch in dataloader:
        sampled_example_ids.extend(batch["example_id"])

    assert 0 not in sampled_example_ids, "Sample with zero weight should not be included in training"
    assert len(sampled_example_ids) == len(edited_table) - 1, "Expected one sample to be excluded"


@pytest.mark.parametrize("task", ["detect", "classify", "segment"])
def test_exclude_zero_weight_collection(task) -> None:
    # Test that sampling weights are correctly applied during metrics collection
    settings = Settings(project_name=f"test_sampling_weights_collection_{task}", exclude_zero_weight_collection=True)
    trainer = TASK2TRAINER[task](
        overrides={
            "model": TASK2MODEL[task],
            "data": TASK2DATASET[task],
            "settings": settings,
            "workers": 2,
        }
    )

    # Model is normally set up in train(); the dataloader build only needs the model's stride
    # (classify uses a ClassificationDataset that doesn't read stride, so a Mock suffices there).
    trainer.model = Mock() if task == "classify" else stub_model_with_stride()

    # Create edited table where several samples have weight 0
    train_table = trainer.data["train"]
    edited_table = EditedTable(
        url=train_table.url.create_sibling(f"erna_{task}"),
        input_table_url=train_table,
        edits={train_table.weights_column_name: {"runs_and_values": [[0, 3], 0.0]}},
    )

    dataloader = trainer.get_dataloader(edited_table, batch_size=2, rank=-1, mode="val")
    sampled_example_ids = []
    for batch in dataloader:
        sampled_example_ids.extend(batch["example_id"])

    assert 0 not in sampled_example_ids, "Sample with zero weight should not be included in collection"
    assert 3 not in sampled_example_ids, "Sample with zero weight should not be included in collection"
    assert len(sampled_example_ids) == len(edited_table) - 2, "Expected two samples to be excluded"


@pytest.mark.skipif(tlc.__version__ < "2.14.0", reason="Test requires 3LC 2.14.0 or higher")
@pytest.mark.parametrize("task", ["detect", "classify"])
def test_train_no_weight_column_in_table(task) -> None:
    # Test that training with a table that has no weight column works
    model = TLCYOLO(TASK2MODEL[task])

    settings = Settings(project_name=f"test_train_no_weight_column_in_table_{task}")
    model.train(data=TASK2DATASET[task], settings=settings, epochs=1, device="cpu", workers=0)
    table = model.trainer.data["train"]

    no_weight_column_table = table.delete_column(table.weights_column_name)
    tables = {"train": no_weight_column_table, "val": model.trainer.data.get("val") or model.trainer.data["test"]}

    model.train(tables=tables, settings=settings, epochs=1, device="cpu", workers=0)

    # Should fail to train with weights enabled on table with no weight column
    with pytest.raises(ValueError):
        settings = Settings(project_name=f"test_train_no_weight_column_in_table_{task}", sampling_weights=True)
        model.train(tables=tables, settings=settings, workers=0, epochs=1, device="cpu")

    # Should collect with exclusion enabled and no weight column
    settings = Settings(
        project_name=f"test_train_no_weight_column_in_table_{task}", exclude_zero_weight_collection=True
    )
    model.collect(tables=tables, settings=settings, workers=0, device="cpu")


def test_collect_with_string_tables_raises() -> None:
    # Passing a string for `tables` (instead of a {split: table} mapping) should fail fast
    model = TLCYOLO(TASK2MODEL["detect"])
    with pytest.raises(TypeError, match=r"Tables must be a mapping of \{split_name: table\}"):
        model.collect(tables="some/path")


def test_illegal_reducer() -> None:
    settings = Settings(image_embeddings_dim=2, image_embeddings_reducer="illegal_reducer")
    with pytest.raises(ValueError):
        settings.verify(training=False)


@pytest.mark.skipif(UMAP_AVAILABLE, reason="Test assumes umap is not installed")
def test_missing_reducer() -> None:
    # umap-learn not installed in the test env, so using it should fail
    settings = Settings(image_embeddings_dim=2, image_embeddings_reducer="umap")
    with pytest.raises(ValueError):
        settings.verify(training=False)


@pytest.mark.parametrize(
    "start,interval,epochs,disable,expected",
    [
        (1, 1, 10, False, list(range(1, 11))),  # Start at 1, interval 1, 10 epochs
        (1, 2, 10, False, [1, 3, 5, 7, 9]),  # Start at 1, interval 2, 5 epochs
        (None, 2, 10, False, []),  # No start means no collection
        (0, 1, 10, False, ValueError),  # Start must be positive
        (1, 0, 10, False, ValueError),  # Interval must be positive
        (1, 1, 10, True, []),  # Disable collection, no mc
    ],
)
def test_get_metrics_collection_epochs(start, interval, epochs, disable, expected) -> None:
    settings = Settings(collection_epoch_start=start, collection_epoch_interval=interval, collection_disable=disable)
    if isinstance(expected, list):
        collection_epochs = settings.get_metrics_collection_epochs(epochs)
        assert collection_epochs == expected, f"Expected {expected}, got {collection_epochs}"
    else:
        with pytest.raises(expected):
            settings.get_metrics_collection_epochs(epochs)


@pytest.mark.parametrize("task", ["detect", "classify", "segment"])
def test_arbitrary_class_indices(task) -> None:  # noqa: C901
    # Test that arbitrary class indices can be used
    settings = Settings(
        project_name=f"test_arbitrary_class_indices_{task}",
        run_name=f"test_arbitrary_class_indices_{task}",
    )

    label_column_name = TASK2LABEL_COLUMN_NAME[task]
    predicted_label_column_name = TASK2PREDICTED_LABEL_COLUMN_NAME[task]

    if task == "detect":
        data_dict = check_tlc_dataset(
            data=TASK2DATASET["detect"],
            tables=None,
            image_column_name="image",
            label_column_name=label_column_name,
            project_name=settings.project_name,
            task="detect",
        )
    elif task == "classify":
        data_dict = check_tlc_dataset(
            data=TASK2DATASET["classify"],
            tables=None,
            image_column_name="image",
            label_column_name=label_column_name,
            project_name=settings.project_name,
            task="classify",
        )

    elif task == "segment":
        data_dict = check_tlc_dataset(
            data=TASK2DATASET["segment"],
            tables=None,
            image_column_name="image",
            label_column_name=label_column_name,
            project_name=settings.project_name,
            task="segment",
        )

    # Create edited tables where class indices are changed
    edited_tables = {}
    for split in ("train", "val"):
        table = data_dict[split]
        table_value_map = table.get_value_map(label_column_name)
        label_map = {k: -(k**2) for k in table_value_map.keys()}  # 0, 1, 2, ... -> 0, -1, -4, ...
        edited_value_map = {label_map[k]: v for k, v in table_value_map.items()}
        edited_schema_table = table.set_value_map(label_column_name, edited_value_map)

        if task == "detect":
            bbs_edits = []
            for i, row in enumerate(edited_schema_table.table_rows):
                remapped_labels = [label_map[label] for label in row["bbs"]["instances_additional_data"]["label"]]
                bbs_edits.append([i])
                bbs_edits.append(
                    {
                        **row["bbs"],
                        "instances_additional_data": {
                            **row["bbs"]["instances_additional_data"],
                            "label": remapped_labels,
                        },
                    }
                )

            edited_tables[split] = EditedTable(
                url=edited_schema_table.url.create_sibling(f"edited_value_map_and_values_{task}"),
                input_table_url=edited_schema_table,
                edits={"bbs": {"runs_and_values": bbs_edits}},
            )
        elif task == "classify":
            edits = []
            for i, row in enumerate(edited_schema_table.table_rows):
                edits.append([i])
                edits.append(label_map[row[label_column_name]])

            edited_tables[split] = EditedTable(
                url=edited_schema_table.url.create_sibling(f"edited_value_map_and_values_{task}"),
                input_table_url=edited_schema_table,
                edits={label_column_name: {"runs_and_values": edits}},
            )

        elif task == "segment":
            edits = []
            for i, row in enumerate(edited_schema_table.table_rows):
                edits.append([i])

                instance_properties_override = deepcopy(row["segmentations"]["instance_properties"])
                instance_properties_override["label"] = [label_map[i] for i in instance_properties_override["label"]]

                segmentations_edit = {
                    "rles": [rle.decode() for rle in row["segmentations"]["rles"]],
                    "instance_properties": instance_properties_override,
                }

                edits.append(segmentations_edit)

            edited_tables[split] = EditedTable(
                url=edited_schema_table.url.create_sibling(f"edited_value_map_and_values_{task}"),
                input_table_url=edited_schema_table,
                edits={
                    "segmentations": {"runs_and_values": edits},
                },
            )

    # Check that the edited table can be used for training and validation
    model = TLCYOLO(TASK2MODEL[task])
    results = model.train(tables=edited_tables, settings=settings, epochs=1, device="cpu")

    assert results, f"{task} training with arbitrary class indices failed"

    run = _get_run_from_settings(settings)

    # Verify metrics have the expected class indices
    sample_metrics_tables = [
        m for m in run.metrics_tables if TASK2PREDICTED_LABEL_COLUMN_NAME[task].split(".")[0] in m.columns
    ]
    metrics_df = pd.concat(
        [metrics_table.to_pandas() for metrics_table in sample_metrics_tables],
        ignore_index=True,
    )

    if task == "detect":
        for i in range(len(metrics_df)):
            assert all(label <= 0 for label in metrics_df["bbs_predicted"][i]["instances_additional_data"]["label"])

        # Verify that a giraffe is predicted in the second image
        predicted_label = np.sqrt(-metrics_df["bbs_predicted"][1]["instances_additional_data"]["label"][0])
        assert table_value_map[predicted_label]["internal_name"] == "giraffe"
    elif task == "classify":
        assert all(label <= 0 for label in metrics_df[predicted_label_column_name]), "Predicted label indices mismatch"

    elif task == "segment":
        predicted_labels = (x["instance_properties"]["label"] for x in metrics_df["segmentations_predicted"])
        assert all(label <= 0 for predicted_row in predicted_labels for label in predicted_row), (
            "Predicted label indices mismatch"
        )

    # Verify that the metrics schema is correct
    label_value_map = edited_tables["train"].get_value_map(label_column_name)
    predicted_label_value_map = sample_metrics_tables[0].get_value_map(predicted_label_column_name)
    assert label_value_map == predicted_label_value_map, "Predicted label value map mismatch"


@pytest.mark.parametrize(
    "train_classes,val_classes,description,expected_error",
    [
        (
            ["a", "b", "c"],
            ["a", "b"],
            "Extra class in train table",
            "All splits must have the same categories, but 'train' has categories that 'val' does not: {2: 'c'}",
        ),
        (
            ["a", "b"],
            ["a", "b", "c"],
            "Extra class in val table",
            "All splits must have the same categories, but 'val' has categories that 'train' does not: {2: 'c'}",
        ),
        (
            ["a", "b", "c"],
            ["a", "b", "d"],
            "Different extra classes in both tables",
            "All splits must have the same categories, but 'train' has categories that 'val' does not: {2: 'c'} "
            "and 'val' has categories that 'train' does not: {2: 'd'}",
        ),
    ],
)
def test_check_tlc_dataset_different_categories(train_classes, val_classes, description, expected_error) -> None:
    # Test that an error is raised if the categories of the tables are different
    project_name = f"test_check_tlc_dataset_different_categories_{description.lower().replace(' ', '_')}"

    train_structure = {
        "image": tlc.schemas.ImageSchema(),
        "label": tlc.schemas.CategoricalLabelSchema(classes=train_classes),
    }
    val_structure = {
        "image": tlc.schemas.ImageSchema(),
        "label": tlc.schemas.CategoricalLabelSchema(classes=val_classes),
    }

    train_table = tlc.Table.from_dict(
        {"image": ["a.jpg", "b.jpg"], "label": [0, 1]},
        schema=train_structure,
        project_name=project_name,
        dataset_name="train",
    )
    val_table = tlc.Table.from_dict(
        {"image": ["c.jpg", "d.jpg"], "label": [0, 1]},
        schema=val_structure,
        project_name=project_name,
        dataset_name="val",
    )

    with pytest.raises(ValueError, match=expected_error):
        check_tlc_dataset(
            data="",
            tables={
                "train": train_table,
                "val": val_table,
            },
            image_column_name="image",
            label_column_name="label",
            task="classify",
        )


def test_check_tlc_dataset_bad_tables() -> None:
    # Test that an error is raised if tables or urls are not provided properly
    tables = {"train": [1, 2, 3], "val": [4, 5, 6]}

    with pytest.raises(ValueError):
        check_tlc_dataset(data="", tables=tables, image_column_name="a", label_column_name="b", task="detect")


def test_check_tlc_dataset_bad_url() -> None:
    # Test that an error is raised if a non-valid url is provided
    tables = {"train": "some_url", "val": "some_other_url"}

    with pytest.raises(ValueError):
        check_tlc_dataset(data="", tables=tables, image_column_name="a", label_column_name="b", task="detect")


def test_check_tlc_dataset_string_tables_converted_before_split_filter() -> None:
    """Regression test: string table entries must be converted to tlc.Table before the splits filter is applied.

    Previously, when only a "train" table was passed as a URL string and check_tlc_dataset was called with
    splits=("test",), the "train" entry stayed as a string because conversion was gated by the splits filter.
    Later code then called .get_value_map() on the string, causing AttributeError.
    """
    # Create a minimal table to use as the "train" split
    train_schema = {
        "image": tlc.schemas.ImageSchema(),
        "label": tlc.schemas.CategoricalLabelSchema(classes=["a", "b"]),
    }
    train_table = tlc.Table.from_dict(
        {"image": [str(DUMMY_IMAGE_FILE)], "label": [0]},
        schema=train_schema,
        project_name="test_string_conversion_bug",
        dataset_name="train",
        table_name="initial",
        if_exists="overwrite",
    )

    # Pass the table URL as a string (simulating tables={"train": "s3://..."})
    tables = {"train": train_table.url.to_str()}

    # Call with splits=("train",) — the bug was that string entries not matching splits were
    # never converted to tlc.Table, causing AttributeError on .get_value_map() later.
    # Using task="classify" since it accepts simple "label" column names.
    result = check_tlc_dataset(
        data="",
        tables=tables,
        image_column_name="image",
        label_column_name="label",
        task="classify",
        splits=("train",),
    )

    # The key assertion: the train entry should be a tlc.Table, not a string
    assert isinstance(result["train"], tlc.Table)


def test_small_segmentations() -> None:
    # Test that small segmentations are skipped properly
    structure = {
        "image": tlc.schemas.ImageSchema(),
        "segmentations": tlc.data_types.SegmentationPolygons.schema(
            classes=["a", "b", "c"],
            relative=True,
        ),
    }
    zidane_image_path = DUMMY_IMAGE_FILE.as_posix()

    relative_polygons_sample = {
        "image": zidane_image_path,
        "segmentations": tlc.data_types.SegmentationPolygons(
            image_width=10,
            image_height=10,
            relative=True,
            labels=[0, 1, 2],
            polygons=[
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],  # Should be fine
                [0.0, 0.0, 0.5, 0.0, 1.0, 0.0],  # A line with no area, should be ignored
                [0.0, 0.0, 0.01, 0.0, 0.01, 0.01, 0.0, 0.01],  # Should become a one pixel mask, which should be ignored
            ],
        ),
    }

    table_writer = tlc.TableWriter(
        schema=structure,
        project_name="test_small_segmentations",
        dataset_name="test",
        table_name="initial",
    )
    table_writer.add_row(relative_polygons_sample)
    table = table_writer.finalize()

    first_row = table[0]
    assert len(first_row["segmentations"].polygons) == 3  # All three instances should be present in some way
    assert len(first_row["segmentations"].polygons[0]) == 6  # Expecting a full polygon
    assert len(first_row["segmentations"].polygons[1]) < 6  # Expecting some kind of zero area polygon
    assert len(first_row["segmentations"].polygons[2]) == 0  # Expecting an empty list

    dataset = TLCYOLODataset(
        table,
        task="segment",
        data={"channels": 3},
        image_column_name="image",
        label_column_name=TASK2LABEL_COLUMN_NAME["segment"],
    )
    assert len(dataset.labels[0]["segments"]) == 1
    assert len(dataset.labels[0]["cls"]) == 1


def test_absolute_segmentation_polygons() -> None:
    # Test that absolute segmentation polygons are handled correctly
    structure = {
        "image": tlc.schemas.ImageSchema(),
        "segmentations": tlc.data_types.SegmentationPolygons.schema(
            classes=["a", "b", "c"],
            relative=False,
        ),
    }

    table_writer = tlc.TableWriter(
        schema=structure,
        project_name="test_absolute_segmentation_polygons",
        dataset_name="test",
        table_name="initial",
    )

    zidane_image_path = DUMMY_IMAGE_FILE.as_posix()
    im = Image.open(zidane_image_path)
    width, height = im.size
    table_writer.add_row(
        {
            "image": zidane_image_path,
            "segmentations": {
                "image_width": width,
                "image_height": height,
                "instance_properties": {
                    "label": [0],
                },
                "polygons": [
                    [0, 0, 0, height, width, 0],
                ],
            },
        }
    )
    table = table_writer.finalize()

    # Should pass the seg table checker
    check_seg_table(table, "image", TASK2LABEL_COLUMN_NAME["segment"])

    # Should be able to populate the dataset with relative polygons
    dataset = TLCYOLODataset(
        table,
        task="segment",
        data={"channels": 3},
        image_column_name="image",
        label_column_name=TASK2LABEL_COLUMN_NAME["segment"],
    )
    assert len(dataset.labels[0]["segments"]) == 1
    assert all(polygon.max() <= 1.0 and polygon.min() >= 0.0 for polygon in dataset.labels[0]["segments"])

    # Should be able to train and collect metrics on this dataset
    model = TLCYOLO(TASK2MODEL["segment"])
    tables = {"train": table, "val": table}
    results = model.train(
        tables=tables,
        settings=Settings(project_name="test_absolute_segmentation_polygons", run_name="test"),
        epochs=1,
        device="cpu",
        imgsz=640,
        batch=1,
    )
    assert results, "Training should succeed"


def _decode_rles(rles):
    """Decode COCO RLEs to a `(H, W, N)` uint8 tensor."""
    import pycocotools.mask as mask_utils
    import torch

    return torch.from_numpy(np.ascontiguousarray(mask_utils.decode(rles)))


def test_segment_masks_built_only_for_filtered_predictions(monkeypatch) -> None:
    # Masks must be generated for the filtered predictions only, at the original image resolution, and stay
    # index-aligned with the other per-instance columns.
    import torch
    from ultralytics.utils import ops

    from tlc_ultralytics.engine.validator import PREDICTION_INDEX
    from tlc_ultralytics.segment.validator import TLCSegmentationValidator

    imgsz = [64, 64]  # model input size, four times the prototype resolution below
    ori_shape = (50, 80)
    num_predictions = 8

    # Each prediction gets its own vertical stripe: prototype channel j is positive only in the columns its own
    # box covers, and coefficient j selects channel j. A mask is therefore non-empty only if it was built from the
    # coefficients that belong to the box it was cropped with — pairing prediction j's box with any other
    # prediction's coefficients crops the stripe away entirely.
    proto = torch.full((num_predictions, 16, 16), -1.0)
    for j in range(num_predictions):
        proto[j, :, 2 * j : 2 * j + 2] = 1.0
    coefficients = torch.eye(num_predictions)
    bboxes = torch.tensor([[8.0 * j, 14.0, 8.0 * j + 8.0, 44.0] for j in range(num_predictions)])
    conf = torch.tensor([0.10, 0.20, 0.30, 0.55, 0.60, 0.70, 0.80, 0.90])
    pred = {
        "bboxes": bboxes,
        "conf": conf,
        "cls": torch.zeros(num_predictions),
        # Ultralytics' own masks, at the prototype resolution — not what is written to 3LC.
        "masks": torch.zeros(num_predictions, 16, 16, dtype=torch.uint8),
    }
    pbatch = {"imgsz": imgsz, "ori_shape": ori_shape, "ratio_pad": None}

    validator = TLCSegmentationValidator.__new__(TLCSegmentationValidator)
    validator._settings = Settings(conf_thres=0.5, max_det=4)
    validator._mask_sources = [(proto, coefficients)]
    validator._mask_imgsz = imgsz
    validator._mask_chunk_instances = 2  # force several chunks for the four surviving predictions

    filtered = validator._filter_top_predictions(pred)
    kept = filtered[PREDICTION_INDEX]
    assert len(kept) == 4, "Expected the confidence threshold and max_det to leave four predictions"

    # Reference: the same masks, generated in one go for the same instances.
    reference = (
        ops.scale_masks(
            ops.process_mask_native(proto, coefficients[kept], bboxes[kept], shape=imgsz)[None],
            ori_shape,
            ratio_pad=None,
        )[0]
        .byte()
        .permute(1, 2, 0)
    )  # (H, W, N), the layout the RLEs decode to

    processed_instances = []
    real_process_mask_native = ops.process_mask_native

    def counting_process_mask_native(protos, masks_in, boxes, shape):
        processed_instances.append(masks_in.shape[0])
        return real_process_mask_native(protos, masks_in, boxes, shape)

    monkeypatch.setattr(ops, "process_mask_native", counting_process_mask_native)

    scaled = validator._scale_filtered_pred(0, filtered, pbatch)

    # Only the filtered instances are turned into full-resolution masks, and never more than a chunk at a time.
    assert sum(processed_instances) == 4, f"Expected masks for the four filtered predictions, got {processed_instances}"
    assert max(processed_instances) <= validator._mask_chunk_instances, "Mask generation was not chunked"

    # The masks arrive RLE-encoded, one COCO RLE per instance at the original image resolution.
    assert scaled["masks"].shape[1:] == (16, 16), "Only Ultralytics' prototype-resolution masks may stay dense"
    assert all(rle["size"] == list(ori_shape) for rle in scaled["rles"]), "RLEs must be at the original resolution"
    masks = _decode_rles(scaled["rles"])
    assert masks.shape == (*ori_shape, 4)
    assert torch.equal(masks, reference), "Chunked masks differ from the unchunked reference"

    # Per-instance columns stay aligned: one mask per confidence/class, in the same order.
    assert len(scaled["conf"]) == len(scaled["cls"]) == len(scaled["rles"])
    assert torch.equal(scaled["conf"], conf[kept])

    # Every mask survives the crop to its own box, which by construction (see the stripes above) only happens if
    # the coefficients PREDICTION_INDEX selected belong to the same instances as the boxes they were cropped with.
    for mask, box in zip(masks.permute(2, 0, 1), scaled["bboxes"], strict=True):
        rows, cols = torch.nonzero(mask, as_tuple=True)
        assert rows.numel() > 0, "Mask is empty, so its coefficients do not belong to the box it was cropped with"
        x0, y0, x1, y1 = box.tolist()
        assert rows.min() >= y0 - 1 and rows.max() <= y1 + 1, "Mask extends outside its bounding box vertically"
        assert cols.min() >= x0 - 1 and cols.max() <= x1 + 1, "Mask extends outside its bounding box horizontally"

    # On large images the pixel budget sizes the chunks instead of `_mask_chunk_instances`. A budget of two image
    # areas stands in for a large image here, and must give chunks of two regardless of the instance cap.
    processed_instances.clear()
    validator._mask_chunk_instances = 32
    validator._mask_chunk_pixels = 2 * ori_shape[0] * ori_shape[1]

    rescaled = validator._scale_filtered_pred(0, filtered, pbatch)

    assert processed_instances == [2, 2], f"Expected chunks sized by the pixel budget, got {processed_instances}"
    assert torch.equal(_decode_rles(rescaled["rles"]), reference), (
        "Pixel-budget chunks differ from the unchunked reference"
    )


def _edge_case_masks(height, width):
    """Binary `(H, W, N)` masks covering the RLE edge cases, plus random ones."""
    masks = [np.zeros((height, width), np.uint8), np.ones((height, width), np.uint8)]
    for y, x in ((0, 0), (height - 1, width - 1), (height // 2, width // 2)):
        single = np.zeros((height, width), np.uint8)
        single[y, x] = 1
        masks.append(single)
    border = np.zeros((height, width), np.uint8)
    border[:, 0] = border[-1, :] = 1
    masks.append(border)
    rng = np.random.default_rng(0)
    masks.extend((rng.random((height, width)) > p).astype(np.uint8) for p in (0.1, 0.5, 0.97))
    return np.stack(masks, axis=-1)


@pytest.mark.parametrize("shape", [(7, 5), (1, 9), (9, 1), (1, 1), (64, 48)])
def test_rles_from_column_major_masks_match_pycocotools(shape) -> None:
    # The run-boundary encoder must produce exactly the RLEs pycocotools encodes from the same dense masks.
    import pycocotools.mask as mask_utils
    import torch

    from tlc_ultralytics.utils.rle import rles_from_column_major_masks

    height, width = shape
    dense = _edge_case_masks(height, width)  # (H, W, N)
    expected = mask_utils.encode(np.asfortranarray(dense))
    column_major = torch.from_numpy(np.ascontiguousarray(dense.transpose(2, 1, 0)))  # (N, W, H)

    rles = rles_from_column_major_masks(column_major, height, width)

    assert [r["size"] for r in rles] == [r["size"] for r in expected]
    assert [r["counts"] for r in rles] == [r["counts"] for r in expected]
    assert rles_from_column_major_masks(column_major[:0], height, width) == []


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="needs a CUDA device")
def test_segment_rles_on_cuda_match_pycocotools() -> None:
    # On CUDA the validator finds mask runs on the GPU, a chunk at a time. Its RLEs must be exactly what pycocotools
    # encodes from the same masks built in one go on the same device. (CUDA and CPU masks themselves differ by a few
    # boundary pixels, from Ultralytics' device-dependent `crop_mask`, so the reference must come from CUDA too.)
    import pycocotools.mask as mask_utils
    import torch
    from ultralytics.utils import ops

    from tlc_ultralytics.segment.validator import TLCSegmentationValidator

    torch.manual_seed(0)
    num_instances, imgsz, ori_shape = 40, [64, 96], (150, 230)
    proto = torch.randn(32, 16, 24, device="cuda")
    coefficients = torch.randn(num_instances, 32, device="cuda")
    xy = torch.rand(num_instances, 2, device="cuda") * torch.tensor([80.0, 50.0], device="cuda")
    bboxes = torch.cat([xy, xy + 4 + torch.rand(num_instances, 2, device="cuda") * 30], dim=1)

    validator = TLCSegmentationValidator.__new__(TLCSegmentationValidator)
    validator._mask_imgsz = imgsz
    validator._mask_chunk_instances = 16  # several chunks
    rles = validator._rles_at_original_resolution(
        proto, coefficients, bboxes, {"ori_shape": ori_shape, "ratio_pad": None}
    )

    dense = ops.scale_masks(ops.process_mask_native(proto, coefficients, bboxes, shape=imgsz)[None], ori_shape)[0]
    expected = mask_utils.encode(np.asfortranarray(dense.byte().cpu().numpy().transpose(1, 2, 0)))
    assert len(rles) == num_instances
    assert [r["counts"] for r in rles] == [r["counts"] for r in expected]


def test_segment_rles_on_mps_match_pycocotools() -> None:
    # `_rles_at_original_resolution` must copy each chunk to host memory before handing it to pycocotools on any
    # non-CUDA accelerator, not just CPU: an MPS tensor raises on `.numpy()` without an explicit `.cpu()` first.
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("Requires an MPS device")

    from tlc.helpers import SegmentationHelper
    from ultralytics.utils import ops

    from tlc_ultralytics.segment.validator import TLCSegmentationValidator

    device = torch.device("mps")
    generator = torch.Generator(device=device).manual_seed(0)

    imgsz = [64, 64]  # model input size, four times the prototype resolution below
    ori_shape = (50, 80)
    num_instances = 6

    proto = torch.rand((32, 16, 16), generator=generator, device=device)
    coefficients = torch.rand((num_instances, 32), generator=generator, device=device)
    bboxes = torch.tensor([[8.0 * j, 14.0, 8.0 * j + 8.0, 44.0] for j in range(num_instances)], device=device)
    pbatch = {"ori_shape": ori_shape, "ratio_pad": None}

    validator = TLCSegmentationValidator.__new__(TLCSegmentationValidator)
    validator._mask_imgsz = imgsz
    validator._mask_chunk_instances = 2  # force several chunks

    rles = validator._rles_at_original_resolution(proto, coefficients, bboxes, pbatch)
    assert len(rles) == num_instances

    # Reference: the same masks, built in one go and encoded from MPS too - masks built from identical inputs
    # differ slightly across devices, so a CPU-built reference would not be a fair comparison.
    reference_native = ops.process_mask_native(proto, coefficients, bboxes, shape=imgsz)
    reference_scaled = ops.scale_masks(reference_native[None], ori_shape, ratio_pad=None)[0]
    reference = reference_scaled.byte().permute(1, 2, 0).cpu().numpy()  # (H, W, N)
    reference_rles = SegmentationHelper.rles_from_masks(reference)

    for rle, reference_rle in zip(rles, reference_rles, strict=True):
        assert rle["counts"] == reference_rle["counts"], "Chunked MPS encoding differs from the unchunked reference"


def test_segment_postprocess_stashes_mask_sources(monkeypatch) -> None:
    # postprocess must stash one (prototypes, coefficients) pair per image, in order, and start over on the next
    # batch - a stale or misaligned stash would silently hand an image another image's masks.
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import ops

    from tlc_ultralytics.segment.validator import TLCSegmentationValidator

    mask_dim = 4
    nms_outputs = []

    def fake_nms_postprocess(self, preds):
        return nms_outputs.pop(0)

    monkeypatch.setattr(DetectionValidator, "postprocess", fake_nms_postprocess)

    def queue_batch(proto_values, instance_counts):
        """Queue one batch: distinguishable prototypes, and coefficients distinguishable per image."""
        proto = torch.stack([torch.full((mask_dim, 8, 8), value) for value in proto_values])
        coefficients = [torch.full((n, mask_dim), float(i + 1)) for i, n in enumerate(instance_counts)]
        nms_outputs.append(
            [
                {
                    "bboxes": torch.tensor([[1.0, 1.0, 20.0, 20.0]] * n).reshape(n, 4),
                    "conf": torch.full((n,), 0.9),
                    "cls": torch.zeros(n),
                    "extra": coefficients[i],
                }
                for i, n in enumerate(instance_counts)
            ]
        )
        return [torch.zeros(len(proto_values), 1), proto], proto, coefficients

    validator = TLCSegmentationValidator.__new__(TLCSegmentationValidator)
    validator._settings = Settings(collect_loss=True)
    validator.process = ops.process_mask  # Ultralytics' default: masks at the prototype resolution

    # One image with instances and one without, so the empty-coefficient branch cannot shift the stash.
    preds, proto, coefficients = queue_batch([1.0, 2.0], [3, 0])
    outputs = validator.postprocess(preds)

    assert validator._curr_raw_preds is preds, "The raw predictions must still be stashed for loss collection"
    assert validator._mask_imgsz == [32, 32], "Model input size is four times the prototype resolution"
    assert len(validator._mask_sources) == 2, "One stash entry per image in the batch"
    for i, (stashed_proto, stashed_coefficients) in enumerate(validator._mask_sources):
        assert torch.equal(stashed_proto, proto[i]), f"Image {i} stashed another image's prototypes"
        assert torch.equal(stashed_coefficients, coefficients[i]), f"Image {i} stashed another image's coefficients"

    # The coefficients are consumed from the predictions, and Ultralytics' masks stay at prototype resolution.
    assert all("extra" not in pred for pred in outputs)
    assert outputs[0]["masks"].shape == (3, 8, 8)
    assert outputs[1]["masks"].shape == (0, 8, 8)

    # A second batch replaces the stash rather than appending to it.
    preds, proto, coefficients = queue_batch([7.0], [2])
    validator.postprocess(preds)

    assert len(validator._mask_sources) == 1, "The stash must be reset for each batch"
    assert torch.equal(validator._mask_sources[0][0], proto[0])
    assert torch.equal(validator._mask_sources[0][1], coefficients[0])


def test_segment_annotation_masks_at_original_resolution(monkeypatch) -> None:
    # End-to-end counterpart of the unit test above: every segmentation annotation written during a real
    # collection pass carries one mask per written instance, at the original image resolution, and reaches the
    # metrics writer already RLE-encoded.
    from tlc.constants import MASKS, RLES

    from tlc_ultralytics.engine.validator import PREDICTION_INDEX
    from tlc_ultralytics.segment.validator import TLCSegmentationValidator

    recorded = []
    build_annotation = TLCSegmentationValidator._build_annotation

    def recording_build_annotation(self, scaled, mapped_classes, h, w):
        assert PREDICTION_INDEX not in scaled, "The prediction index is bookkeeping and must not reach annotations"
        sizes = [tuple(rle["size"]) for rle in scaled["rles"]]
        recorded.append((sizes, (int(h), int(w)), scaled["conf"].tolist(), mapped_classes))
        annotation = build_annotation(self, scaled, mapped_classes, h, w)
        assert MASKS not in annotation, "Annotations must be in row form, without dense masks"
        assert len(annotation[RLES]) == len(mapped_classes)
        return annotation

    monkeypatch.setattr(TLCSegmentationValidator, "_build_annotation", recording_build_annotation)

    settings = Settings(
        project_name="test_segment_mask_resolution",
        run_name="test_segment_mask_resolution",
        conf_thres=0.25,
    )
    model = TLCYOLO(TASK2MODEL["segment"])
    model.collect(data=TASK2DATASET["segment"], splits=("val",), settings=settings, device="cpu", workers=0)

    assert recorded, "Expected at least one image with predictions above the confidence threshold"
    for sizes, ori_shape, confidences, labels in recorded:
        assert all(size == ori_shape for size in sizes), f"Masks at {set(sizes)}, expected original shape {ori_shape}"
        assert len(sizes) == len(confidences) == len(labels), "One mask per written instance"
        assert len(sizes) <= settings.max_det
        assert all(confidence >= settings.conf_thres for confidence in confidences)


def test_segment_row_form_annotation_matches_tlc_encoding() -> None:
    # The segmentation validator hands the metrics writer annotations in row form, with masks it RLE-encoded
    # itself. That row must be exactly what 3LC produces from the same dense masks, and the writer must store it
    # unchanged next to sample-form values and read it back as the same masks.
    import torch
    from tlc.data_types import SegmentationMasks
    from tlc.helpers import SegmentationHelper
    from tlc.schemas import ConfidenceSchema

    from tlc_ultralytics.segment.validator import TLCSegmentationValidator

    h, w = 30, 40
    rng = np.random.default_rng(0)
    dense = np.asfortranarray((rng.random((h, w, 3)) > 0.6).astype(np.uint8))  # (H, W, N)
    labels = [2, 0, 1]
    conf = torch.tensor([0.9, 0.45, 0.3])
    sample = SegmentationMasks(
        image_height=h, image_width=w, masks=dense, mask_format="hwn", labels=labels, confidences=conf.tolist()
    )

    validator = TLCSegmentationValidator.__new__(TLCSegmentationValidator)
    scaled = {"rles": SegmentationHelper.rles_from_masks(dense), "conf": conf}
    row = validator._build_annotation(scaled, labels, h, w)

    schema = SegmentationMasks.schema(
        classes={0: "a", 1: "b", 2: "c"}, per_instance_schemas={"confidence": ConfidenceSchema(writable=False)}
    )
    assert row == schema.to_row(sample), "The row form differs from what 3LC encodes from the dense masks"

    run = tlc.init(project_name="test_segment_row_form", run_name="test_segment_row_form")
    writer = tlc.MetricsTableWriter(run_url=run.url, foreign_table_url=run.url, schema={"seg": schema})
    writer.add_batch({"example_id": [0, 1], "seg": [row, sample]})
    table = writer.finalize()

    for i in range(2):
        written = table[i]["seg"]
        assert isinstance(written, SegmentationMasks)
        assert np.array_equal(written.masks, dense), f"Row {i} does not read back as the original masks"
        assert written.labels.tolist() == labels


@pytest.mark.parametrize("task", ["detect", "pose"])
def test_prediction_index_does_not_reach_annotations(task, monkeypatch) -> None:
    # PREDICTION_INDEX is bookkeeping for the scaling step, so it must be gone from the scaled predictions the
    # task validators turn into annotations.
    import torch

    from tlc_ultralytics.detect.validator import TLCDetectionValidator
    from tlc_ultralytics.engine.validator import PREDICTION_INDEX
    from tlc_ultralytics.pose.validator import TLCPoseValidator

    validator_class = {"detect": TLCDetectionValidator, "pose": TLCPoseValidator}[task]
    pbatch = {"imgsz": [64, 64], "ori_shape": (50, 80), "ratio_pad": None}
    pred = {
        "bboxes": torch.tensor([[4.0, 4.0, 20.0, 20.0], [8.0, 8.0, 24.0, 24.0]]),
        "conf": torch.tensor([0.9, 0.1]),  # the second prediction is filtered out
        "cls": torch.zeros(2),
        "keypoints": torch.zeros(2, 1, 3),
    }

    scaled_preds = []
    monkeypatch.setattr(validator_class, "_prepare_batch", lambda self, i, batch: pbatch)
    monkeypatch.setattr(
        validator_class, "_build_annotation", lambda self, scaled, mapped_classes, h, w: scaled_preds.append(scaled)
    )

    validator = validator_class.__new__(validator_class)
    validator._settings = Settings(conf_thres=0.5)
    validator._cur_pbatches = {}
    validator._cur_filtered_preds = {}
    validator.data = {"range_to_3lc_class": {0: 0}}

    validator._process_predictions([pred], {})

    assert len(scaled_preds) == 1
    assert PREDICTION_INDEX not in scaled_preds[0], "The prediction index must not reach annotation building"
    assert len(scaled_preds[0]["conf"]) == 1, "Expected only the prediction above the confidence threshold"


def test_absolutize_image_url() -> None:
    # Unexpanded aliases should fail
    url = tlc.Url("<UNEXPANDED_ALIAS>/in/my/url.png")
    with pytest.raises(ValueError):
        TLCDatasetMixin._absolutize_image_url(url, tlc.Url("some_table_url"))

    # Non-file schemes should fail
    for scheme in (tlc.url.Scheme.S3, tlc.url.Scheme.GS, tlc.url.Scheme.ABFS):
        url = tlc.Url(f"{scheme}://some/remote/url.png")
        with pytest.raises(ValueError):
            TLCDatasetMixin._absolutize_image_url(url, tlc.Url("some_table_url"))

    # Aliases should be expanded
    url = tlc.Url("<TEST_ALIAS>/in/my/url.png")
    result = TLCDatasetMixin._absolutize_image_url(url, tlc.Url("some_table_url"))
    assert result == "/test/alias/in/my/url.png"
    assert tlc.Url(result).scheme == tlc.url.Scheme.FILE

    # Relative URLs should be made absolute
    relative_url = tlc.Url("../some/relative/url.png")
    assert relative_url.scheme == tlc.url.Scheme.RELATIVE
    result = TLCDatasetMixin._absolutize_image_url(relative_url, tlc.Url("/one/two/table"))
    assert result == "/one/two/some/relative/url.png"
    assert tlc.Url(result).scheme == tlc.url.Scheme.FILE

    # Absolute URLs should remain unchanged
    absolute_url = tlc.Url("/some/absolute/url.png")
    assert absolute_url.scheme == tlc.url.Scheme.FILE
    result = TLCDatasetMixin._absolutize_image_url(absolute_url, tlc.Url("/one/two/table"))
    assert result == "/some/absolute/url.png"
    assert tlc.Url(result).scheme == tlc.url.Scheme.FILE


def test_cache_write_failure_degrades_gracefully() -> None:
    # A failing cache write should not stop the dataset from being constructed
    from tlc.data_types import BoundingBoxes2D

    classes = {0: "cat", 1: "dog"}
    writer = tlc.TableWriter(
        table_name="initial",
        dataset_name="test",
        project_name="test_cache_write_failure",
        schema={"image": tlc.schemas.ImageSchema(), "bbs": BoundingBoxes2D.schema(classes=classes)},
    )
    writer.add_row(
        {
            "image": str(DUMMY_IMAGE_FILE),
            "bbs": BoundingBoxes2D(
                bounding_boxes=[[10.0, 10.0, 50.0, 50.0]],
                bounding_box_format="xyxy",
                image_width=100,
                image_height=100,
                labels=[1],
            ).to_row(),
        }
    )
    table = writer.finalize()

    # Simulate the FileExistsError raised when a parent path component is a file.
    def raise_file_exists(self, *args, **kwargs):
        raise FileExistsError(17, "File exists")

    # The ultralytics LOGGER does not propagate to the root logger, so capture its warnings directly.
    warnings: list[str] = []

    from tlc_ultralytics.engine import dataset as dataset_module

    with patch.object(tlc.Url, "write_text", raise_file_exists):
        with patch.object(dataset_module.LOGGER, "warning", side_effect=lambda msg, *a, **k: warnings.append(msg)):
            dataset = TLCYOLODataset(
                table,
                task="detect",
                data={"channels": 3},
                image_column_name="image",
                label_column_name=TASK2LABEL_COLUMN_NAME["detect"],
            )

    # Construction succeeds despite the cache-write failure, and the in-memory flow is unaffected.
    assert len(dataset.labels) == 1
    assert any("Could not write the images cache" in msg for msg in warnings)


def test_dataset_cache_is_invalidated_when_image_files_change() -> None:
    """A cached missing or corrupt image must not make a Table permanently unusable.

    This is the same failure mode as an alias that initially points at an unavailable mount and is later fixed
    without changing the Table's image URL. The cache stores a hash of the image paths and their file sizes
    (matching `ultralytics.data.utils.get_hash`); any change to the files on disk - an image appearing,
    disappearing, or being replaced with a differently-sized file - changes the hash and triggers a full rescan,
    so a corrupt verdict is not permanent either.
    """
    from tlc.data_types import BoundingBoxes2D

    from tlc_ultralytics.engine import dataset as dataset_module

    alias = "<CACHE_RECOVERY_IMAGES>"
    image_root = TMP / "cache_recovery_images"
    image_root.mkdir(parents=True, exist_ok=True)
    image_paths = [image_root / "image_0.png", image_root / "image_1.png"]
    for image_path in image_paths:
        image_path.unlink(missing_ok=True)
    tlc.url.register_url_alias(alias, str(image_root), force=True)

    def make_dataset() -> TLCYOLODataset:
        return TLCYOLODataset(
            table,
            task="detect",
            data={"channels": 3},
            image_column_name="image",
            label_column_name=TASK2LABEL_COLUMN_NAME["detect"],
        )

    def read_cache() -> dict:
        cache_paths = list(Path(table.url.to_str()).glob("yolo_*.json"))
        assert len(cache_paths) == 1
        return json.loads(cache_paths[0].read_text())

    try:
        writer = tlc.TableWriter(
            table_name="initial",
            dataset_name="cache recovery",
            project_name="test_dataset_cache_recovery",
            schema={"image": tlc.schemas.ImageSchema(), "bbs": BoundingBoxes2D.schema(classes={0: "cat"})},
        )
        for image_path in image_paths:
            writer.add_row(
                {
                    "image": f"{alias}/{image_path.name}",
                    "bbs": BoundingBoxes2D(
                        bounding_boxes=[[10.0, 10.0, 50.0, 50.0]],
                        bounding_box_format="xyxy",
                        image_width=100,
                        image_height=100,
                        labels=[0],
                    ).to_row(),
                }
            )
        table = writer.finalize()

        with pytest.raises(ValueError, match="are missing"):
            make_dataset()

        cache_data = read_cache()
        assert cache_data["version"] == 2
        assert cache_data["corrupt_example_ids"] == []
        assert cache_data["missing_example_ids"] == [0, 1]
        assert "hash" in cache_data

        # One image appears: the changed hash triggers a full rescan, but only the still-missing image is caught by
        # the stat before verify_image is reached
        image_paths[0].write_bytes(DUMMY_IMAGE_FILE.read_bytes())
        with patch.object(dataset_module, "verify_image", wraps=dataset_module.verify_image) as verify_image_mock:
            dataset = make_dataset()

        assert len(dataset.labels) == 1
        verify_image_mock.assert_called_once()
        assert read_cache()["missing_example_ids"] == [1]

        # The other image appears: a full rescan verifies both images
        image_paths[1].write_bytes(DUMMY_IMAGE_FILE.read_bytes())
        with patch.object(dataset_module, "verify_image", wraps=dataset_module.verify_image) as verify_image_mock:
            dataset = make_dataset()

        assert len(dataset.labels) == 2
        assert verify_image_mock.call_count == 2
        assert read_cache()["missing_example_ids"] == []

        # Nothing changed, so a warm cache verifies no images
        with patch.object(dataset_module, "verify_image", wraps=dataset_module.verify_image) as verify_image_mock:
            dataset = make_dataset()

        assert len(dataset.labels) == 2
        verify_image_mock.assert_not_called()

        # One image becomes corrupt (a different size than the dummy image, so the hash changes)
        corrupt_bytes = b"not an image"
        assert len(corrupt_bytes) != DUMMY_IMAGE_FILE.stat().st_size
        image_paths[1].write_bytes(corrupt_bytes)
        with patch.object(dataset_module, "verify_image", wraps=dataset_module.verify_image) as verify_image_mock:
            dataset = make_dataset()

        assert len(dataset.labels) == 1
        assert verify_image_mock.call_count == 2
        assert read_cache()["corrupt_example_ids"] == [1]

        # The corrupt image is restored: the cache is invalidated again, showing that corrupt is not permanent
        image_paths[1].write_bytes(DUMMY_IMAGE_FILE.read_bytes())
        dataset = make_dataset()

        assert len(dataset.labels) == 2
        assert read_cache()["corrupt_example_ids"] == []
    finally:
        tlc.url.unregister_url_alias(alias)


def test_dataset_cache_from_older_version_is_regenerated() -> None:
    """A cache written by an older version is discarded and all images are verified again.

    Version 1 caches recorded missing images as corrupt, so upgrading must not reuse their verdicts. The cache key
    did not change between versions, so the old cache sits at exactly the path the new version reads.
    """
    from tlc.data_types import BoundingBoxes2D

    from tlc_ultralytics.engine import dataset as dataset_module

    image_root = TMP / "cache_version_upgrade_images"
    image_root.mkdir(parents=True, exist_ok=True)
    image_paths = [image_root / "image_0.png", image_root / "image_1.png"]
    for image_path in image_paths:
        image_path.write_bytes(DUMMY_IMAGE_FILE.read_bytes())

    writer = tlc.TableWriter(
        table_name="initial",
        dataset_name="cache version upgrade",
        project_name="test_dataset_cache_version_upgrade",
        schema={"image": tlc.schemas.ImageSchema(), "bbs": BoundingBoxes2D.schema(classes={0: "cat"})},
    )
    for image_path in image_paths:
        writer.add_row(
            {
                "image": str(image_path),
                "bbs": BoundingBoxes2D(
                    bounding_boxes=[[10.0, 10.0, 50.0, 50.0]],
                    bounding_box_format="xyxy",
                    image_width=100,
                    image_height=100,
                    labels=[0],
                ).to_row(),
            }
        )
    table = writer.finalize()

    def make_dataset() -> TLCYOLODataset:
        return TLCYOLODataset(
            table,
            task="detect",
            data={"channels": 3},
            image_column_name="image",
            label_column_name=TASK2LABEL_COLUMN_NAME["detect"],
        )

    make_dataset()
    cache_paths = list(Path(table.url.to_str()).glob("yolo_*.json"))
    assert len(cache_paths) == 1
    cache_path = cache_paths[0]

    # Replace it with a version 1 cache that, like one written while the images were unavailable, marks them corrupt
    cache_path.write_text(json.dumps({"version": 1, "corrupt_example_ids": [0, 1]}))

    with patch.object(dataset_module, "verify_image", wraps=dataset_module.verify_image) as verify_image_mock:
        dataset = make_dataset()

    assert len(dataset.labels) == 2
    assert verify_image_mock.call_count == len(image_paths)

    assert list(Path(table.url.to_str()).glob("yolo_*.json")) == [cache_path]
    cache_data = json.loads(cache_path.read_text())
    assert set(cache_data.keys()) == {"version", "hash", "corrupt_example_ids", "missing_example_ids"}
    assert cache_data["version"] == 2
    assert cache_data["corrupt_example_ids"] == []
    assert cache_data["missing_example_ids"] == []
    assert isinstance(cache_data["hash"], str) and cache_data["hash"]


def test_dataset_cache_with_out_of_range_id_is_regenerated() -> None:
    """A hand-edited cache with an out-of-range example id must not crash dataset construction.

    This can happen if a cache file is corrupted or edited outside of the normal write path; the id-range check in
    `_load_cached_example_ids` must catch it and fall back to a full rescan rather than raising an IndexError later.
    """
    from tlc.data_types import BoundingBoxes2D

    image_root = TMP / "cache_out_of_range_images"
    image_root.mkdir(parents=True, exist_ok=True)
    image_paths = [image_root / "image_0.png", image_root / "image_1.png"]
    for image_path in image_paths:
        image_path.write_bytes(DUMMY_IMAGE_FILE.read_bytes())

    writer = tlc.TableWriter(
        table_name="initial",
        dataset_name="cache out of range",
        project_name="test_dataset_cache_out_of_range",
        schema={"image": tlc.schemas.ImageSchema(), "bbs": BoundingBoxes2D.schema(classes={0: "cat"})},
    )
    for image_path in image_paths:
        writer.add_row(
            {
                "image": str(image_path),
                "bbs": BoundingBoxes2D(
                    bounding_boxes=[[10.0, 10.0, 50.0, 50.0]],
                    bounding_box_format="xyxy",
                    image_width=100,
                    image_height=100,
                    labels=[0],
                ).to_row(),
            }
        )
    table = writer.finalize()

    def make_dataset() -> TLCYOLODataset:
        return TLCYOLODataset(
            table,
            task="detect",
            data={"channels": 3},
            image_column_name="image",
            label_column_name=TASK2LABEL_COLUMN_NAME["detect"],
        )

    make_dataset()
    cache_paths = list(Path(table.url.to_str()).glob("yolo_*.json"))
    assert len(cache_paths) == 1
    cache_path = cache_paths[0]
    valid_hash = json.loads(cache_path.read_text())["hash"]

    # Hand-edit the cache to reference an out-of-range example id, keeping the hash valid
    cache_path.write_text(
        json.dumps({"version": 2, "hash": valid_hash, "corrupt_example_ids": [99], "missing_example_ids": []})
    )

    dataset = make_dataset()

    assert len(dataset.labels) == 2
    assert list(Path(table.url.to_str()).glob("yolo_*.json")) == [cache_path]
    cache_data = json.loads(cache_path.read_text())
    assert cache_data["corrupt_example_ids"] == []
    assert cache_data["missing_example_ids"] == []


def test_extra_metrics() -> None:
    """Test providing extra metrics callback and schemas work as expected"""

    _yolo_dataset_path, (table_train, table_val) = _create_test_image_and_table()

    BATCH_SIZE = 2

    def extra_metrics(preds, batch):
        return {
            "constant_metric": [1] * BATCH_SIZE * 2,
            "metric_with_schema": list(range(BATCH_SIZE * 2)),
        }

    metric_schemas = {
        "metric_with_schema": tlc.schemas.CategoricalLabelSchema(
            classes=[f"value_{i}" for i in range(BATCH_SIZE * 2)],
        ),
    }

    settings = Settings(
        metrics_collection_function=extra_metrics,
        metrics_schemas=metric_schemas,
        project_name="test_extra_metrics",
        run_name="test_extra_metrics",
        run_description="Test extra metrics",
    )

    model = TLCYOLO("yolo11n.pt")
    results = model.train(
        tables={"train": table_train, "val": table_val},
        settings=settings,
        epochs=1,
        device="cpu",
        imgsz=640,
        batch=BATCH_SIZE,
    )
    assert results, "Training should succeed"

    run = _get_run_from_settings(settings)

    sample_metrics_tables = [m for m in run.metrics_tables if "constant_metric" in m.columns]
    assert len(sample_metrics_tables) == 2, "Should have two metrics tables"

    for metrics_table in sample_metrics_tables:
        assert "constant_metric" in metrics_table.columns, "Constant metric should be present"

        assert "metric_with_schema" in metrics_table.columns, "Metric with schema should be present"

        constant_column = metrics_table.get_column_as_pyarrow_array("constant_metric").to_numpy()
        assert np.all(constant_column == 1), "Constant metric should be 1"
        metric_with_schema_column = metrics_table.get_column_as_pyarrow_array("metric_with_schema").to_numpy()
        assert np.all(
            metric_with_schema_column == np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        )
        assert metrics_table.rows_schema["metric_with_schema"].value.map is not None


def test_no_predictions() -> None:
    yolo_dataset_path, (table_train, table_val) = _create_test_image_and_table()

    overrides = {
        "data": yolo_dataset_path,
        "epochs": 1,
        "batch": 4,
        "device": "cpu",
        "save": False,
        "conf": 1.0,
        "plots": False,
    }

    model_ultralytics = YOLO("yolo11n.pt")
    results_ultralytics = model_ultralytics.train(**overrides)
    assert results_ultralytics, "Detection yolo training failed"

    settings = Settings(
        collection_epoch_start=1,
        collect_loss=True,
        image_embeddings_dim=2,
        image_embeddings_reducer="pacmap",
        project_name="test_no_predictions_project",
        run_name="test_no_predictions",
        run_description="Test no predictions training",
        conf_thres=1.0,
    )

    model_3lc = TLCYOLO("yolo11n.pt")
    tables = {"train": table_train, "val": table_val}
    results_3lc = model_3lc.train(**overrides, settings=settings, tables=tables)
    assert results_3lc, "Detection training failed"


def test_complete_label_column_name() -> None:
    assert _complete_label_column_name("a", "a") == "a"
    assert _complete_label_column_name("a", "a.b.c") == "a.b.c"
    assert _complete_label_column_name("a.b.c", "d.e.f") == "a.b.c"
    assert _complete_label_column_name("", "a.b.c") == "a.b.c"


@pytest.mark.parametrize("mode", ["train", "val"])
@pytest.mark.parametrize("task", ["detect", "pose", "obb"])
def test_dataset_determinism(mode, task) -> None:
    """Test that datasets are deterministic with the same seed across separate processes."""
    from dataset_determinism import _compare_dataset_rows, create_dataset_samples

    if task == "obb":
        # FIXME: Some boxes are identical but rotated by pi/2 and w-h are swapped
        pytest.skip("Known issue with obb in train mode")

    rows_3lc, rows_ultralytics = create_dataset_samples(mode, task)

    assert len(rows_3lc) == len(rows_ultralytics), "Number of batches should be the same"

    for row_3lc, row_ultralytics in zip(rows_3lc, rows_ultralytics, strict=False):
        _compare_dataset_rows(row_ultralytics, row_3lc)


@pytest.mark.parametrize("mode", ["train", "val"])
@pytest.mark.parametrize("task", ["detect", "pose", "obb"])
def test_dataset_determinism_with_random_tracking(mode, task) -> None:
    """Test that datasets are deterministic and don't make unexpected random calls.

    This test spawns a subprocess, enables random tracking, and checks that no
    unexpected random calls are made.
    """
    import json
    import subprocess
    import sys
    import tempfile

    if task == "obb":
        # FIXME: Known issue with obb in train mode
        pytest.skip("Known issue with obb in train mode")

    TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(TMP)) as temp_dir:
        output_file = (Path(temp_dir) / "output.json").as_posix()
        cmd = [
            sys.executable,
            "-c",
            "from dataset_determinism import create_dataset_samples_with_tracking;"
            f"create_dataset_samples_with_tracking('{mode}', '{task}', '{output_file}')",
        ]
        subprocess.run(cmd, check=True, cwd=str(Path(__file__).parent))

        with open(output_file) as f:
            tracking_result = json.load(f)

        assert "error" not in tracking_result, f"Subprocess failed:\n{tracking_result['error']}"

        assert tracking_result["rows_count_3lc"] == tracking_result["rows_count_ultralytics"], (
            "Number of batches should be the same"
        )


@pytest.mark.parametrize("task", ["classify", "detect", "segment"])
def test_dataset_cache(task) -> None:
    """Test that the dataset cache is used correctly."""
    from dataset_determinism import _compare_dataset_rows

    # Create a table to use
    settings = Settings(project_name=f"test_dataset_cache_{task}")
    trainer = TASK2TRAINER[task](
        overrides={"data": TASK2DATASET[task], "model": TASK2MODEL[task], "settings": settings},
    )
    trainer.model = stub_model_with_stride()

    # Check that there is no cache
    cache_paths = list(Path(trainer.data["train"].url.to_str()).glob("yolo_*.json"))
    assert len(cache_paths) == 0, "There should be no cache files"

    # Get a dataset for the table
    dataset_first = trainer.build_dataset(trainer.data["train"], mode="val", batch=1)

    cache_paths = list(Path(trainer.data["train"].url.to_str()).glob("yolo_*.json"))
    assert len(cache_paths) == 1, "There should be one cache file"

    # Get the dataset again, make sure verify_image is not called here
    with patch("tlc_ultralytics.engine.dataset.verify_image") as verify_image_mock:
        dataset_second = trainer.build_dataset(trainer.data["train"], mode="val", batch=1)
        verify_image_mock.assert_not_called()

    # Check that the dataset has the same rows
    assert len(dataset_first) == len(dataset_second), "Number of rows should be the same"
    for row_first, row_second in zip(dataset_first, dataset_second, strict=False):
        _compare_dataset_rows(row_second, row_first)

    cache_paths = list(Path(trainer.data["train"].url.to_str()).glob("yolo_*.json"))
    assert len(cache_paths) == 1, "There should still be one cache file"

    cache_path = cache_paths[0]
    cache_data = json.loads(cache_path.read_text())
    assert cache_data["version"] == 2, "Cache version should be 2"
    assert cache_data["corrupt_example_ids"] == []
    assert cache_data["missing_example_ids"] == []
    assert "hash" in cache_data


@pytest.mark.parametrize("task", ["detect", "segment", "classify", "obb", "pose"])
def test_dataset_does_not_pickle_table(task: str) -> None:
    """No `tlc.Table` may cross the pickle boundary into a dataloader worker - a single surviving reference costs
    one full copy of the annotations per worker on `spawn` platforms.
    """
    settings = Settings(project_name=f"test_dataset_pickle_{task}")
    trainer = TASK2TRAINER[task](
        overrides={"data": TASK2DATASET[task], "model": TASK2MODEL[task], "settings": settings},
    )
    trainer.model = stub_model_with_stride()

    table = trainer.data["train"]
    dataset = trainer.build_dataset(table, mode="val", batch=1)

    # `reducer_override` sees every object the pickler writes, so this catches a Table reached by any path.
    pickled_tables: list[tlc.Table] = []

    class TableDetectingPickler(pickle.Pickler):
        def reducer_override(self, obj):
            if isinstance(obj, tlc.Table):
                pickled_tables.append(obj)
            return NotImplemented

    buffer = io.BytesIO()
    TableDetectingPickler(buffer).dump(dataset)
    assert not pickled_tables, f"Tables written into the worker payload: {[t.url.to_str() for t in pickled_tables]}"

    # The main process is unaffected: the Table is still there and the shared data dict is not mutated
    assert dataset.table is table, "The dataset should still hold the Table it was built from"
    assert dataset.display_name == table.dataset_name, "display_name should survive"
    assert dataset.table.url == table.url, "The validator reads dataset.table.url"
    assert trainer.data["train"] is table, "__getstate__ must not mutate the shared data dict"

    # A worker can produce samples without ever materializing the Table
    worker_dataset = pickle.loads(buffer.getvalue())
    assert worker_dataset.__dict__["_table"] is None, "The unpickled dataset should not carry a Table"
    assert len(worker_dataset) == len(dataset), "The unpickled dataset should have the same length"
    sample = worker_dataset[0]
    assert "example_id" in sample, "The unpickled dataset should still produce example ids"
    assert worker_dataset.__dict__["_table"] is None, "__getitem__ must not reload the Table"

    # ...but it is restored on demand if anything asks for it
    assert worker_dataset.table.url == table.url, "The Table should be restored lazily from its URL"


def test_bad_arguments() -> None:
    """Test that bad arguments are caught early and an error is raised"""
    model = TLCYOLO(TASK2MODEL["detect"])

    train_table = tlc.Table.from_dict(
        {"col": []}, project_name="test_bad_arguments", dataset_name="train", table_name="initial"
    )
    val_table = tlc.Table.from_dict(
        {"col": []}, project_name="test_bad_arguments", dataset_name="val", table_name="initial"
    )

    # Data is used instead of tables
    with pytest.raises(ValueError):
        model.train(data={"train": train_table, "val": val_table})


def test_settings_serialization() -> None:
    settings = Settings(
        project_name="test_settings_serialization",
        run_name="test_settings_serialization",
        image_embeddings_reducer="umap",
        image_embeddings_dim=2,
        exclude_zero_weight_training=True,
        metrics_collection_function=lambda x, y: {"test_metric": [1] * len(x)},
    )

    settings_dict = settings.to_dict()
    settings_from_dict = Settings(**settings_dict)

    assert settings_from_dict.project_name == settings.project_name
    assert settings_from_dict.run_name == settings.run_name
    assert settings_from_dict.image_embeddings_reducer == settings.image_embeddings_reducer


# HELPERS


def _get_run_from_settings(settings: Settings) -> tlc.Run:
    run_url = TMP_PROJECT_ROOT_URL / settings.project_name / "runs" / settings.run_name
    return tlc.Run.from_url(run_url)


def _create_no_predictions_data_yaml(dataset_path: pathlib.Path) -> pathlib.Path:
    data_yaml_content = f"""
path: {dataset_path.as_posix()}
train: train/images
val: val/images
names:
  0: a
  1: b
  2: c
    """
    data_yaml_path = dataset_path / "data.yaml"
    with open(data_yaml_path, "w") as f:
        f.write(data_yaml_content)
    return data_yaml_path


def _create_test_image_and_table() -> tuple[pathlib.Path, tuple[tlc.Table, tlc.Table]]:  # noqa: C901
    data_set_path = TMP / "no_predictions"

    train_dir = data_set_path / "train" / "images"
    val_dir = data_set_path / "val" / "images"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    labels_dir = data_set_path / "train" / "labels"
    os.makedirs(labels_dir, exist_ok=True)

    def circles_overlap(c1, c2, r1, r2):
        return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) < (r1 + r2)

    rng = np.random.default_rng(seed=42)

    for i in range(16):
        img = np.full((640, 640, 3), 255, dtype=np.uint8)
        img_name = f"train_test_{i}.jpg"
        h, w = img.shape[:2]

        # Generate two non-overlapping circles
        circles = []
        for _ in range(2):
            while True:
                radius = int(0.1 * min(h, w))
                center_x = rng.integers(radius, w - radius)
                center_y = rng.integers(radius, h - radius)

                # Check if new circle overlaps with existing ones
                if not any(circles_overlap((center_x, center_y), (c[0], c[1]), radius, radius) for c in circles):
                    break

            circles.append((center_x, center_y, radius))

            def is_similar_to_grey(color, threshold=30):
                # Check if color components are too close to each other (indicating greyness)
                r, g, b = color
                avg = (r + g + b) / 3
                return all(abs(c - avg) < threshold for c in (r, g, b))

            # Generate color that's not too similar to grey
            while True:
                color = tuple(rng.integers(0, 255, 3).tolist())
                if not is_similar_to_grey(color):
                    break
            cv2.circle(img, (center_x, center_y), radius, color, -1)

        # Save training image
        img_path = train_dir / img_name
        cv2.imwrite(img_path.as_posix(), img)

        # Create label file for train image
        label_path = labels_dir / f"{img_name[:-4]}.txt"
        with open(label_path, "w") as f:
            for cx, cy, r in circles:
                # Convert to YOLO format (x_center, y_center, width, height) normalized
                x_center = cx / w
                y_center = cy / h
                width = (2 * r) / w
                height = (2 * r) / h
                f.write(f"0 {x_center} {y_center} {width} {height}\n")

    # Generate validation images (all gray) with random labels
    val_labels_dir = data_set_path / "val" / "labels"
    os.makedirs(val_labels_dir, exist_ok=True)

    for i in range(16):
        img = np.full((640, 640, 3), 128, dtype=np.uint8)
        img_name = f"val_test_{i}.jpg"
        img_path = val_dir / img_name
        cv2.imwrite(img_path.as_posix(), img)

        # Create random labels for validation images
        if i == 1:
            h, w = img.shape[:2]
            label_path = val_labels_dir / f"{img_name[:-4]}.txt"
            with open(label_path, "w") as f:
                for _ in range(1):
                    x_center = rng.random()
                    y_center = rng.random()
                    width = rng.random() * 0.2  # Max 20% of image width
                    height = rng.random() * 0.2  # Max 20% of image height
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")

    yolo_dataset_file = _create_no_predictions_data_yaml(data_set_path)

    from tlc_ultralytics import create_tables_from_yaml_file

    tables = create_tables_from_yaml_file(
        str(yolo_dataset_file),
        task="detect",
        if_exists="overwrite",
        splits=("train", "val"),
    )

    return yolo_dataset_file, (tables["train"], tables["val"])


@pytest.mark.skip(reason="TODO: Fix test")
def test_pose_flip_and_oks_overrides(mocker) -> None:
    """Verify that flip augmentation uses provided flip indices and loss uses provided OKS sigmas."""
    from ultralytics.data.augment import RandomFlip
    from ultralytics.utils.loss import KeypointLoss
    from ultralytics.utils.metrics import kpt_iou

    from tlc_ultralytics.pose.loss import v8UnreducedPoseLoss

    # Spies for the various components of the pose loss.
    random_flip_spy = mocker.spy(RandomFlip, "__init__")
    keypoint_loss_init_spy = mocker.spy(KeypointLoss, "__init__")
    keypoint_loss_call_spy = mocker.spy(KeypointLoss, "__call__")
    unreduced_keypoint_loss_call_spy = mocker.spy(v8UnreducedPoseLoss, "__call__")
    kpt_iou_spy = mocker.patch("ultralytics.models.yolo.pose.val.kpt_iou", wraps=kpt_iou)

    # Settings with custom flip indices, OKS sigmas, and point/line properties.
    settings = Settings(
        project_name="test_pose_overrides_project",
        run_name="test_pose_overrides",
        collect_loss=True,
        oks_sigmas=OKS_SIGMAS.tolist(),  # Sigmas provided in Settings; will only be used for loss, not for kpt_iou
        flip_indices=list(range(17)),
        **COCO_POSE_SETTINGS_OVERRIDES,
    )

    overrides = {
        "data": TASK2DATASET["pose"],
        "model": TASK2MODEL["pose"],
        "device": "cpu",
        "epochs": 1,
        "batch": 2,
        "imgsz": 64,
        "workers": 0,
        "deterministic": True,
        "seed": 0,
        "flipud": 0.0,
        "fliplr": 1.0,
    }

    model = TLCYOLO(TASK2MODEL["pose"])

    model.train(settings=settings, **overrides)

    ## Data / metrics checks

    # Check the run, training table, and first metrics table are all correct
    run = _get_run_from_settings(settings)
    first_metrics_table = run.metrics_tables[0]
    train_table = tlc.Table.from_url(first_metrics_table.get_foreign_table_url().to_absolute(first_metrics_table.url))
    check_pose_table_and_metrics_tables(train_table, first_metrics_table, COCO_POSE_SETTINGS_OVERRIDES)

    ## Tests for flip augmentation.

    # Assert that RandomFlip is instantiated with the provided flip indices
    assert random_flip_spy.call_count == 2
    assert random_flip_spy.call_args_list[0][1] == {
        "p": 0.0,
        "direction": "vertical",
        "flip_idx": settings.flip_indices,
    }
    assert random_flip_spy.call_args_list[1][1] == {
        "p": 1.0,
        "direction": "horizontal",
        "flip_idx": settings.flip_indices,
    }

    ## Tests for OKS sigmas - Settings override should only affect loss, not kpt_iou or Table OKS sigmas.

    # Assert that kpt_iou is called with the Table OKS sigmas
    assert kpt_iou_spy.call_count == 9
    for args in kpt_iou_spy.call_args_list:
        assert np.allclose(args[1]["sigma"], [1 / 17] * 17)

    # Assert that KeypointLoss is instantiated with the provided OKS sigmas
    assert keypoint_loss_init_spy.call_count == 8
    for call in [1, 3, 5, 7]:
        # The other calls are from super.__init__, where other sigmas are used
        call_kwargs = keypoint_loss_init_spy.call_args_list[call][1]
        assert np.allclose(call_kwargs["sigmas"].numpy(), OKS_SIGMAS)

    # Assert that keypoint loss is called with the provided OKS sigmas
    assert keypoint_loss_call_spy.call_count != 0
    for args in keypoint_loss_call_spy.call_args_list:
        assert np.allclose(args[0][0].sigmas.numpy(), OKS_SIGMAS)

    # Assert that unreduced keypoint loss is called with the provided OKS sigmas
    assert unreduced_keypoint_loss_call_spy.call_count == 2
    for arg in unreduced_keypoint_loss_call_spy.call_args_list:
        assert np.allclose(arg[0][0].keypoint_loss.sigmas.numpy(), OKS_SIGMAS)

    # Assert that the sigmas are set on the model and validators
    assert np.allclose(model.trainer.model.criterion.keypoint_loss.sigmas.numpy(), OKS_SIGMAS)
    assert np.allclose(model.trainer.validator.loss_fn.keypoint_loss.sigmas.numpy(), OKS_SIGMAS)
    assert np.allclose(model.trainer.train_validator.loss_fn.keypoint_loss.sigmas.numpy(), OKS_SIGMAS)


@pytest.mark.parametrize("task", ["pose", "obb", "detect", "segment"])
@pytest.mark.parametrize("mode", ["train", "val"])
def test_single_sample_equality(task: str, mode: str) -> None:
    """Test that a single sample from the dataset is equal between 3LC and Ultralytics."""

    if task == "segment" and mode == "train":
        # FIXME: known issue with out of order instances in train mode for segment
        pytest.skip("Fails because of out of order instances")

    NUM_SAMPLES = 4
    settings = Settings(project_name=f"test_dataset_determinism_mode_{mode}_task_{task}")
    overrides = {
        "data": TASK2DATASET[task],
        "model": TASK2MODEL[task],
        "seed": 42,
        "deterministic": True,
    }

    overrides_3lc = overrides.copy()
    overrides_3lc["settings"] = settings

    # Set up Ultralytics dataset
    trainer_ultralytics = TASK2ULTRALYTICS_TRAINER[task](overrides=overrides)
    trainer_ultralytics.model = stub_model_with_stride()
    dataset_ultralytics = trainer_ultralytics.build_dataset(trainer_ultralytics.data["train"], mode=mode, batch=1)

    # Set up 3LC dataset
    trainer_3lc = TASK2TRAINER[task](overrides=overrides_3lc)
    trainer_3lc.model = stub_model_with_stride()
    dataset_3lc = trainer_3lc.build_dataset(trainer_3lc.data["train"], mode=mode, batch=1)

    plot = False  # Turn on to enable debug viz.

    for i in range(NUM_SAMPLES):
        if mode == "train":
            # FIXME: known issue with random seed in train mode for obb and pose
            # Samples will not be equal unless we explicitly seed before fetching data
            random.seed(42)

        sample_3lc = dataset_3lc[i]
        if mode == "train":
            random.seed(42)
        sample_ultralytics = dataset_ultralytics[i]

        if plot and i == 0:
            plot_ultralytics(sample_3lc, "3LC")
            plot_ultralytics(sample_ultralytics, "Ultralytics")

        compare_dataset_values(sample_ultralytics, sample_3lc, task, mode)


def test_embeddings_dim_settings() -> None:
    settings = Settings(image_embeddings_dim=-1, label_column_name="test")

    with pytest.raises(AssertionError):
        settings.verify(training=False)

    for dim in [1, 2, 3, 4]:
        settings.image_embeddings_dim = dim

        with capture_logs() as tlc_messages:
            settings.verify(training=False)

        if dim in [1, 4]:
            assert len(tlc_messages) == 1
        else:
            assert len(tlc_messages) == 0


@contextmanager
def capture_logs(loglevel: int = logging.INFO):
    # Capture ultralytics logger output specifically
    from ultralytics.utils import LOGGER

    ultralytics_logger = LOGGER

    # Create handler for 3LC run
    tlc_handler = CapturingHandler()
    tlc_handler.setLevel(loglevel)
    formatter = logging.Formatter("%(message)s")
    tlc_handler.setFormatter(formatter)

    # Add handler to ultralytics logger
    ultralytics_logger.addHandler(tlc_handler)

    try:
        yield tlc_handler.log_messages
    finally:
        ultralytics_logger.removeHandler(tlc_handler)


# === Tests for _get_default_names and table reuse functionality ===


class TestGetDefaultNames:
    """Tests for the _get_default_names function."""

    def test_default_names_no_overrides(self):
        """Test default naming without any overrides."""
        from tlc_ultralytics.utils.dataset import _get_default_names

        project, dataset = _get_default_names("coco128.yaml", "train")
        assert project == "coco128-YOLO"
        assert dataset == "coco128-train"

    def test_default_names_with_path(self):
        """Test default naming with a full path."""
        from tlc_ultralytics.utils.dataset import _get_default_names

        project, dataset = _get_default_names("/path/to/my_dataset.yaml", "val")
        assert project == "my_dataset-YOLO"
        assert dataset == "my_dataset-val"

    def test_default_names_with_project_override(self):
        """Test that project_name override is respected."""
        from tlc_ultralytics.utils.dataset import _get_default_names

        project, dataset = _get_default_names("coco128.yaml", "train", project_name="custom-project")
        assert project == "custom-project"
        assert dataset == "coco128-train"

    def test_default_names_with_dataset_override(self):
        """Test that dataset_name override is respected."""
        from tlc_ultralytics.utils.dataset import _get_default_names

        project, dataset = _get_default_names("coco128.yaml", "train", dataset_name="custom-dataset")
        assert project == "coco128-YOLO"
        assert dataset == "custom-dataset"

    def test_default_names_with_both_overrides(self):
        """Test that both overrides are respected."""
        from tlc_ultralytics.utils.dataset import _get_default_names

        project, dataset = _get_default_names(
            "coco128.yaml", "train", project_name="my-project", dataset_name="my-dataset"
        )
        assert project == "my-project"
        assert dataset == "my-dataset"

    def test_default_names_empty_split(self):
        """Test naming with empty split (used for project-only lookup)."""
        from tlc_ultralytics.utils.dataset import _get_default_names

        project, dataset = _get_default_names("coco128.yaml", "")
        assert project == "coco128-YOLO"
        assert dataset == "coco128-"

    def test_default_names_with_pathlib_path(self):
        """Test that pathlib.Path objects work correctly."""
        from tlc_ultralytics.utils.dataset import _get_default_names

        project, dataset = _get_default_names(Path("/some/path/dataset.yaml"), "test")
        assert project == "dataset-YOLO"
        assert dataset == "dataset-test"


class TestGetExistingTable:
    """Tests for the _get_existing_table function."""

    def test_reuse_nonexistent_table_returns_none(self):
        """Test that reuse mode returns None for nonexistent tables."""
        from tlc_ultralytics.utils.dataset import _get_existing_table

        result = _get_existing_table("nonexistent-project-xyz", "nonexistent-dataset", "reuse")
        assert result is None

    def test_overwrite_nonexistent_table_returns_none(self):
        """Test that overwrite mode returns None for nonexistent tables."""
        from tlc_ultralytics.utils.dataset import _get_existing_table

        result = _get_existing_table("nonexistent-project-xyz", "nonexistent-dataset", "overwrite")
        assert result is None

    def test_rename_nonexistent_table_returns_none(self):
        """Test that rename mode returns None for nonexistent tables."""
        from tlc_ultralytics.utils.dataset import _get_existing_table

        result = _get_existing_table("nonexistent-project-xyz", "nonexistent-dataset", "rename")
        assert result is None

    def test_raise_nonexistent_table_returns_none(self):
        """Test that raise mode returns None for nonexistent tables (no error if table doesn't exist)."""
        from tlc_ultralytics.utils.dataset import _get_existing_table

        result = _get_existing_table("nonexistent-project-xyz", "nonexistent-dataset", "raise")
        assert result is None

    def test_reuse_existing_table(self):
        """Test that reuse mode returns the existing table."""
        from tlc_ultralytics.utils.dataset import _get_existing_table

        # Create a table first
        project_name = "test-reuse-project"
        dataset_name = "test-reuse-dataset"
        table = tlc.Table.from_dict(
            {"image": ["a.jpg", "b.jpg"]},
            project_name=project_name,
            dataset_name=dataset_name,
            table_name="initial",
            if_exists="overwrite",
        )

        # Now test reuse
        result = _get_existing_table(project_name, dataset_name, "reuse")
        assert result is not None
        assert result.url == table.url

    def test_raise_existing_table_raises_error(self):
        """Test that raise mode raises FileExistsError for existing tables."""
        from tlc_ultralytics.utils.dataset import _get_existing_table

        # Create a table first
        project_name = "test-raise-project"
        dataset_name = "test-raise-dataset"
        tlc.Table.from_dict(
            {"image": ["a.jpg", "b.jpg"]},
            project_name=project_name,
            dataset_name=dataset_name,
            table_name="initial",
            if_exists="overwrite",
        )

        # Now test raise mode
        with pytest.raises(FileExistsError, match="Table already exists"):
            _get_existing_table(project_name, dataset_name, "raise")

    def test_overwrite_existing_table_returns_none(self):
        """Test that overwrite mode returns None (allowing table to be recreated)."""
        from tlc_ultralytics.utils.dataset import _get_existing_table

        # Create a table first
        project_name = "test-overwrite-project"
        dataset_name = "test-overwrite-dataset"
        tlc.Table.from_dict(
            {"image": ["a.jpg", "b.jpg"]},
            project_name=project_name,
            dataset_name=dataset_name,
            table_name="initial",
            if_exists="overwrite",
        )

        # Overwrite mode should return None so table gets recreated
        result = _get_existing_table(project_name, dataset_name, "overwrite")
        assert result is None

    def test_rename_existing_table_returns_none(self):
        """Test that rename mode returns None (allowing new table with different name)."""
        from tlc_ultralytics.utils.dataset import _get_existing_table

        # Create a table first
        project_name = "test-rename-project"
        dataset_name = "test-rename-dataset"
        tlc.Table.from_dict(
            {"image": ["a.jpg", "b.jpg"]},
            project_name=project_name,
            dataset_name=dataset_name,
            table_name="initial",
            if_exists="overwrite",
        )

        # Rename mode should return None so a new table gets created with different name
        result = _get_existing_table(project_name, dataset_name, "rename")
        assert result is None


class TestCreateTablesFromYamlFileReuse:
    """Integration tests for table creation and reuse with create_tables_from_yaml_file."""

    def test_tables_reused_on_second_call(self):
        """Test that tables are reused when calling create_tables_from_yaml_file twice."""
        from tlc_ultralytics.utils.dataset import create_tables_from_yaml_file

        # First call - creates tables
        tables1 = create_tables_from_yaml_file(
            "coco8.yaml",
            task="detect",
            project_name="test-reuse-yaml",
            if_exists="overwrite",
            splits=("train", "val"),
        )

        train_url1 = tables1["train"].url
        val_url1 = tables1["val"].url

        # Second call - should reuse tables
        tables2 = create_tables_from_yaml_file(
            "coco8.yaml",
            task="detect",
            project_name="test-reuse-yaml",
            if_exists="reuse",
            splits=("train", "val"),
        )

        # URLs should match (same tables reused)
        assert tables2["train"].url == train_url1
        assert tables2["val"].url == val_url1

    def test_default_naming_scheme_consistency(self):
        """Test that default naming scheme is consistent between creation and reuse."""
        from tlc_ultralytics.utils.dataset import create_tables_from_yaml_file

        # Create with default naming (no project_name specified)
        tables1 = create_tables_from_yaml_file(
            "coco8.yaml",
            task="detect",
            if_exists="overwrite",
            splits=("train",),
        )

        # Verify default naming was applied
        assert "coco8-YOLO" in str(tables1["train"].url)
        assert "coco8-train" in str(tables1["train"].url)

        # Second call should find the table with default naming
        tables2 = create_tables_from_yaml_file(
            "coco8.yaml",
            task="detect",
            if_exists="reuse",
            splits=("train",),
        )

        assert tables2["train"].url == tables1["train"].url

    def test_raise_on_existing_table(self):
        """Test that if_exists='raise' raises error when table exists."""
        from tlc_ultralytics.utils.dataset import create_tables_from_yaml_file

        # First call - creates tables
        create_tables_from_yaml_file(
            "coco8.yaml",
            task="detect",
            project_name="test-raise-yaml",
            if_exists="overwrite",
            splits=("train",),
        )

        # Second call with raise should error
        with pytest.raises(FileExistsError):
            create_tables_from_yaml_file(
                "coco8.yaml",
                task="detect",
                project_name="test-raise-yaml",
                if_exists="raise",
                splits=("train",),
            )

    def test_split_with_multiple_paths(self):
        """Test that a split with multiple paths creates a single table containing all data."""
        from ultralytics.data.utils import check_det_dataset

        from tlc_ultralytics.utils.dataset import create_tables_from_yaml_file

        # Get the actual paths from coco8
        data = check_det_dataset("coco8.yaml")
        train_path = data["train"]
        val_path = data["val"]

        # Create a YAML where train split is a list containing both train and val paths
        yaml_data = {
            "train": [train_path, val_path],  # Two entries for the train split
            "val": None,
            "test": None,
            "names": data["names"],
            "nc": data["nc"],
        }

        yaml_path = TMP / "coco8-multi-path-split.yaml"
        yaml_path.write_text(yaml.safe_dump(yaml_data))

        # Create tables - multiple paths are passed directly to from_yolo_url
        tables = create_tables_from_yaml_file(
            str(yaml_path),
            task="detect",
            project_name="test-multi-path-split",
            if_exists="overwrite",
            splits=("train",),
        )

        assert "train" in tables
        combined_table = tables["train"]

        # Create separate tables to compare row counts
        tables_train_only = create_tables_from_yaml_file(
            "coco8.yaml",
            task="detect",
            project_name="test-multi-path-split-train-only",
            if_exists="overwrite",
            splits=("train",),
        )
        tables_val_only = create_tables_from_yaml_file(
            "coco8.yaml",
            task="detect",
            project_name="test-multi-path-split-val-only",
            if_exists="overwrite",
            splits=("val",),
        )

        train_row_count = len(tables_train_only["train"])
        val_row_count = len(tables_val_only["val"])

        # The combined table should have all rows from both paths
        assert len(combined_table) == train_row_count + val_row_count

        # Verify the table name is "initial"
        assert combined_table.name == "initial"


# === Instance embeddings tests ===

INSTANCE_EMB_OVERRIDES = {"batch": 4, "device": "cpu", "workers": 0}


@pytest.mark.parametrize("task", ["detect", "segment", "pose", "obb"])
def test_instance_embeddings_collection(task: str) -> None:
    """Test that predicted instance embeddings are collected as a top-level column."""
    dim = 2
    settings = Settings(
        project_name=f"test_instance_emb_{task}",
        run_name=f"test_instance_emb_{task}",
        instance_embeddings_dim=dim,
        instance_embeddings_reducer="pca",
        label_column_name=TASK2LABEL_COLUMN_NAME[task],
    )

    model = TLCYOLO(TASK2MODEL[task])
    model.collect(data=TASK2DATASET[task], splits=("train",), settings=settings, **INSTANCE_EMB_OVERRIDES)

    run = _get_run_from_settings(settings)
    metrics_tables = get_metrics_tables_from_run(run)
    default_tables = metrics_tables["default_stream"]
    assert len(default_tables) >= 1, "Expected at least one default_stream metrics table"

    metrics_df = pd.concat([m.to_pandas() for m in default_tables], ignore_index=True)

    # Predicted instance embeddings should be a top-level column
    assert "predicted_instance_embedding" in metrics_df.columns, (
        f"Expected 'predicted_instance_embedding' column in metrics for task {task}"
    )

    # Each row should contain a list of embeddings (one per instance)
    for row_embs in metrics_df["predicted_instance_embedding"]:
        assert isinstance(row_embs, (list, np.ndarray)), "Expected list or array of embeddings per image"
        for emb in row_embs:
            assert len(emb) == dim, f"Expected embedding dimension {dim}, got {len(emb)}"


@pytest.mark.parametrize("task", ["detect", "segment", "pose", "obb"])
def test_gt_instance_embeddings_collection(task: str) -> None:
    """Test that both predicted and ground-truth instance embeddings are collected."""
    dim = 2
    settings = Settings(
        project_name=f"test_gt_instance_emb_{task}",
        run_name=f"test_gt_instance_emb_{task}",
        instance_embeddings_dim=dim,
        ground_truth_instance_embeddings=True,
        instance_embeddings_reducer="pca",
        label_column_name=TASK2LABEL_COLUMN_NAME[task],
    )

    model = TLCYOLO(TASK2MODEL[task])
    model.collect(data=TASK2DATASET[task], splits=("train",), settings=settings, **INSTANCE_EMB_OVERRIDES)

    run = _get_run_from_settings(settings)
    metrics_tables = get_metrics_tables_from_run(run)
    default_tables = metrics_tables["default_stream"]
    assert len(default_tables) >= 1, "Expected at least one default_stream metrics table"

    metrics_df = pd.concat([m.to_pandas() for m in default_tables], ignore_index=True)

    # Both predicted and GT instance embeddings should be top-level columns
    assert "predicted_instance_embedding" in metrics_df.columns, (
        f"Expected 'predicted_instance_embedding' column for task {task}"
    )
    assert "ground_truth_instance_embedding" in metrics_df.columns, (
        f"Expected 'ground_truth_instance_embedding' column for task {task}"
    )

    # Validate predicted embeddings
    for row_embs in metrics_df["predicted_instance_embedding"]:
        assert isinstance(row_embs, (list, np.ndarray)), "Expected list of embeddings"
        for emb in row_embs:
            assert len(emb) == dim, f"Expected predicted embedding dim {dim}, got {len(emb)}"

    # Validate GT embeddings
    for row_embs in metrics_df["ground_truth_instance_embedding"]:
        assert isinstance(row_embs, (list, np.ndarray)), "Expected list of embeddings"
        for emb in row_embs:
            assert len(emb) == dim, f"Expected GT embedding dim {dim}, got {len(emb)}"

    # At least some images should have GT annotations
    gt_counts = [len(row_embs) for row_embs in metrics_df["ground_truth_instance_embedding"]]
    assert sum(gt_counts) > 0, "Expected at least some GT instance embeddings"


def test_all_embeddings_combined() -> None:
    """Test that predicted instance and GT instance embeddings can be collected together."""
    dim = 2
    settings = Settings(
        project_name="test_all_embeddings_combined",
        run_name="test_all_embeddings_combined",
        instance_embeddings_dim=dim,
        ground_truth_instance_embeddings=True,
        instance_embeddings_reducer="pca",
        label_column_name=TASK2LABEL_COLUMN_NAME["detect"],
    )

    model = TLCYOLO(TASK2MODEL["detect"])
    model.collect(data=TASK2DATASET["detect"], splits=("train",), settings=settings, **INSTANCE_EMB_OVERRIDES)

    run = _get_run_from_settings(settings)
    metrics_tables = get_metrics_tables_from_run(run)
    default_tables = metrics_tables["default_stream"]
    metrics_df = pd.concat([m.to_pandas() for m in default_tables], ignore_index=True)

    # Both embedding types should be present
    assert "predicted_instance_embedding" in metrics_df.columns, "Expected predicted instance embeddings column"
    assert "ground_truth_instance_embedding" in metrics_df.columns, "Expected GT instance embeddings column"


@pytest.mark.parametrize("reducer", ["pca", "umap", "pacmap"])
def test_instance_reducer_fit_then_transform(reducer: str) -> None:
    """Unit test: each reducer must survive a fit followed by a fresh .transform().

    Exercises ``_fit_embeddings_reducer`` and ``_transform_embeddings``
    directly on synthetic data so the test doesn't depend on a full model run or
    the size of the YOLO test dataset. This is the scenario that catches pacmap's
    ``save_tree=True`` requirement — without it the fitted reducer can't project
    instances outside the fit sample into the fitted space.
    """
    pytest.importorskip(reducer if reducer != "pca" else "sklearn")

    from tlc_ultralytics.utils._instance_reduce import (
        _fit_embeddings_reducer,
        _transform_embeddings,
    )

    rng = np.random.default_rng(0)
    sample = rng.normal(size=(200, 32)).astype(np.float32)

    try:
        # random_state is a raw constructor kwarg for all three reducers; passing it through
        # exercises that instance_embeddings_reducer_kwargs are forwarded to the constructor.
        fitted = _fit_embeddings_reducer(
            sample,
            method=reducer,
            n_components=2,
            random_state=42,
        )
    except ValueError as exc:
        # pacmap on macOS ARM currently fails during fit with a
        # broadcast/shape error from its internal KNN. Skip rather than fail —
        # the post-fit .transform() path (the save_tree=True regression guard)
        # can only be checked when fit itself works.
        pytest.skip(f"{reducer} fit failed in this environment: {exc}")

    assert fitted is not None
    # The forwarded kwarg reached the underlying reducer constructor.
    assert fitted.random_state == 42

    # Transform a disjoint batch with the fitted reducer — this crashes on
    # pacmap when save_tree=False, which is the bug the in-process reducer guards.
    new_raw = [rng.normal(size=(5, 32)).astype(np.float32) for _ in range(3)]
    projected = _transform_embeddings(new_raw, fitted, n_components=2)
    assert all(r.shape == (5, 2) for r in projected)


def test_gt_instance_embeddings_requires_instance_dim() -> None:
    """Test that ground_truth_instance_embeddings requires instance_embeddings_dim > 0."""
    settings = Settings(
        ground_truth_instance_embeddings=True,
        instance_embeddings_dim=0,
        label_column_name="test",
    )
    with pytest.raises(AssertionError, match="ground_truth_instance_embeddings requires instance_embeddings_dim"):
        settings.verify(training=False)


def test_gt_instance_embeddings_incompatible_with_collection_disable() -> None:
    """Test that ground_truth_instance_embeddings can't be used with collection_disable."""
    settings = Settings(
        ground_truth_instance_embeddings=True,
        instance_embeddings_dim=2,
        collection_disable=True,
        label_column_name="test",
    )
    with pytest.raises(AssertionError, match="Cannot disable collection"):
        settings.verify(training=True)


def test_reducer_validation_split() -> None:
    """pca is supported by the in-process reduction for both image and instance embeddings."""
    settings = Settings(image_embeddings_dim=2, image_embeddings_reducer="pca", label_column_name="test")
    settings.verify(training=False)

    settings = Settings(instance_embeddings_dim=2, instance_embeddings_reducer="pca", label_column_name="test")
    settings.verify(training=False)

    settings = Settings(instance_embeddings_dim=2, instance_embeddings_reducer="illegal", label_column_name="test")
    with pytest.raises(ValueError, match="instance_embeddings_reducer"):
        settings.verify(training=False)

    settings = Settings(image_embeddings_dim=2, image_embeddings_reducer="illegal", label_column_name="test")
    with pytest.raises(ValueError, match="image_embeddings_reducer"):
        settings.verify(training=False)


def test_build_rewritten_arrow_table() -> None:
    """Unit test: the rewrite drops the raw embedding columns, carries the rest through by reference and types
    the reduced columns from their schemas."""
    import pyarrow as pa

    from tlc_ultralytics.utils._table_rewrite import (
        _build_rewritten_arrow_table,
        _fixed_size_list_array,
        _nested_list_array,
    )
    from tlc_ultralytics.utils.schemas import _instance_embeddings_list_schema, _reduced_image_embeddings_schema

    source = pa.table(
        {
            "example_id": pa.array([0, 1], type=pa.int32()),
            "rles": pa.array([[b"abc"], [b"de", b"f"]]),
            "input_table_id": pa.nulls(2, type=pa.int32()),  # declared with a default, no data in the parquet
            "embeddings": pa.array([[0.0] * 4, [1.0] * 4], type=pa.list_(pa.float32(), 4)),
            "predicted_instance_embedding_raw": pa.array([[[0.0] * 4], []], type=pa.list_(pa.list_(pa.float32(), 4))),
        }
    )
    schema = {
        "example_id": tlc.schemas.ExampleIdSchema(),
        "rles": tlc.Schema(),
        "input_table_id": tlc.schemas.ForeignTableIdSchema(foreign_table_url="../table"),
        "predicted_instance_embedding": _instance_embeddings_list_schema(2),
        "embeddings_pca": _reduced_image_embeddings_schema(2, "pca"),
    }
    reduced_columns = {
        "predicted_instance_embedding": _nested_list_array(
            [np.array([[0.1, 0.2]], dtype=np.float32), np.empty((0, 2), dtype=np.float32)], 2
        ),
        "embeddings_pca": _fixed_size_list_array(np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32), 2),
    }

    rewritten = _build_rewritten_arrow_table(
        source, schema, reduced_columns, {"embeddings", "predicted_instance_embedding_raw"}
    )

    assert rewritten.column_names == [
        "example_id",
        "rles",
        "input_table_id",
        "predicted_instance_embedding",
        "embeddings_pca",
    ]
    # Pass-through columns are the source's own arrow data, not a re-encoded copy
    assert rewritten.column("rles").to_pylist() == source.column("rles").to_pylist()
    # A column with no data in the source parquet gets its schema default, as a row-by-row copy would have
    assert rewritten.column("input_table_id").to_pylist() == [0, 0]
    # The reduced columns are typed from their schemas, exactly as tlc's own writer would type them
    assert rewritten.schema.field("predicted_instance_embedding").type == pa.list_(pa.list_(pa.float32(), 2))
    assert rewritten.schema.field("embeddings_pca").type == pa.list_(pa.float32(), 2)
    assert np.allclose(rewritten.column("embeddings_pca").to_pylist(), [[0.3, 0.4], [0.5, 0.6]])
    pred_rows = rewritten.column("predicted_instance_embedding").to_pylist()
    assert [len(row) for row in pred_rows] == [1, 0]
    assert np.allclose(pred_rows[0], [[0.1, 0.2]])


def test_rewrite_default_fill_tolerates_mistyped_default() -> None:
    """Unit test: a default whose type the column cannot hold leaves the column alone rather than failing the
    rewrite, which would strand the run's raw metrics tables."""
    import pyarrow as pa

    from tlc_ultralytics.utils._table_rewrite import _filled_with_default

    column = pa.chunked_array([pa.nulls(2, type=pa.int32())])
    mistyped = tlc.schemas.Int32Schema(default_value="not an int")

    assert _filled_with_default(column, mistyped).to_pylist() == [None, None]
    assert _filled_with_default(column, tlc.schemas.Int32Schema(default_value=7)).to_pylist() == [7, 7]


def test_write_rewritten_metrics_table_refuses_empty() -> None:
    """Unit test: an empty rewrite must raise before a table url is allocated."""
    import pyarrow as pa

    from tlc_ultralytics.utils._table_rewrite import _write_rewritten_metrics_table

    run = tlc.init(project_name="test_rewrite_empty", run_name="test_rewrite_empty")
    empty = pa.table({"example_id": pa.array([], type=pa.int32())})

    with pytest.raises(ValueError, match="empty metrics table"):
        _write_rewritten_metrics_table(
            arrow_table=empty,
            schema={"example_id": tlc.schemas.ExampleIdSchema()},
            run_url=run.url,
            foreign_table_url=run.url / "dummy_table",
        )

    assert not tlc.Run.from_url(run.url).metrics, "No metrics table should have been registered on the run"


def test_rolling_metrics_writer_rolls_by_bytes() -> None:
    """Unit test: the rolling writer flushes to a new metrics table when the buffer threshold is crossed,
    and the flushed tables together hold all rows in order."""
    from tlc_ultralytics.utils._rolling_writer import _RollingMetricsWriter

    run = tlc.init(project_name="test_rolling_writer", run_name="test_rolling_writer")

    writer = _RollingMetricsWriter(
        run_url=run.url,
        foreign_table_url=run.url / "dummy_table",
        schema={"value": tlc.schemas.Float32Schema()},
        max_buffer_bytes=1,  # every batch crosses the threshold -> one table per batch
    )

    assert writer.num_flushed_tables == 0
    for i in range(3):
        writer.add_batch({"example_id": [2 * i, 2 * i + 1], "value": [0.5, 1.5]})
        assert writer.num_flushed_tables == i + 1

    table_urls, metrics_infos = writer.finalize()
    assert len(table_urls) == 3
    assert len(metrics_infos) == 3
    assert all(info["stream_name"] == "default_stream" for info in metrics_infos)

    df = pd.concat([tlc.Table.from_url(url).to_pandas() for url in table_urls], ignore_index=True)
    assert sorted(df["example_id"].tolist()) == list(range(6))

    # Each flushed table was registered on the run as it was written
    run = tlc.Run.from_url(run.url)
    registered_urls = {info["url"] for info in run.metrics}
    assert {info["url"] for info in metrics_infos} <= registered_urls

    # A finalized writer must reject further batches and repeated finalize calls
    with pytest.raises(RuntimeError):
        writer.add_batch({"example_id": [0], "value": [0.0]})
    with pytest.raises(RuntimeError):
        writer.finalize()


def test_metrics_flushing_end_to_end() -> None:
    """Collect with a zero buffer threshold: metrics are flushed to multiple tables that together
    hold one row per dataset image."""
    settings = Settings(
        project_name="test_metrics_flushing",
        run_name="test_metrics_flushing",
        metrics_max_buffer_mb=0,  # flush after every batch
    )

    model = TLCYOLO(TASK2MODEL["detect"])
    model.collect(data=TASK2DATASET["detect"], splits=("train",), settings=settings, batch=2, device="cpu", workers=0)

    run = _get_run_from_settings(settings)
    default_tables = get_metrics_tables_from_run(run)["default_stream"]
    assert len(default_tables) >= 2, "Expected the zero buffer threshold to flush multiple metrics tables"

    df = pd.concat([t.to_pandas() for t in default_tables], ignore_index=True)
    assert sorted(df["example_id"].tolist()) == [0, 1, 2, 3], "Flushed tables should cover every image exactly once"


def test_metrics_flushing_with_instance_embeddings() -> None:
    """Collect with a zero buffer threshold and instance embeddings: every raw table is rewritten with a
    reduced embedding column, raw columns and tables are gone, and rows are preserved."""
    dim = 2
    settings = Settings(
        project_name="test_metrics_flushing_instance_emb",
        run_name="test_metrics_flushing_instance_emb",
        metrics_max_buffer_mb=0,  # flush after every batch
        instance_embeddings_dim=dim,
        instance_embeddings_reducer="pca",
        # A user-supplied n_components must be ignored in favor of instance_embeddings_dim
        instance_embeddings_reducer_kwargs={"n_components": 7},
        instance_embeddings_fit_sample_size=5,  # force the sampled-fit path
        label_column_name=TASK2LABEL_COLUMN_NAME["detect"],
    )

    model = TLCYOLO(TASK2MODEL["detect"])
    model.collect(data=TASK2DATASET["detect"], splits=("train",), settings=settings, batch=2, device="cpu", workers=0)

    run = _get_run_from_settings(settings)
    default_tables = get_metrics_tables_from_run(run)["default_stream"]
    assert len(default_tables) >= 2, "Expected the zero buffer threshold to flush multiple metrics tables"

    df = pd.concat([t.to_pandas() for t in default_tables], ignore_index=True)
    assert sorted(df["example_id"].tolist()) == [0, 1, 2, 3], "Rewritten tables should cover every image exactly once"

    assert "predicted_instance_embedding" in df.columns, "Expected the reduced embedding column in every table"
    assert "predicted_instance_embedding_raw" not in df.columns, "Raw embedding columns should have been rewritten"

    total_instances = 0
    for row_embs in df["predicted_instance_embedding"]:
        for emb in row_embs:
            assert len(emb) == dim
            total_instances += 1
    assert total_instances > 0, "Expected at least some reduced instance embeddings"


def test_metrics_flushing_with_image_embeddings() -> None:
    """Collect with a zero buffer threshold and image embeddings: the reducer is fitted on a sample drawn
    across the flushed tables and every table is rewritten with the reduced column in place of the raw one."""
    dim = 2
    settings = Settings(
        project_name="test_metrics_flushing_image_emb",
        run_name="test_metrics_flushing_image_emb",
        metrics_max_buffer_mb=0,  # flush after every batch
        image_embeddings_dim=dim,
        image_embeddings_reducer="pca",
        # A user-supplied n_components must be ignored in favor of image_embeddings_dim
        image_embeddings_reducer_args={"n_components": 7},
        image_embeddings_fit_sample_size=3,  # force the sampled-fit path (fewer than the 4 images)
    )

    model = TLCYOLO(TASK2MODEL["detect"])
    model.collect(data=TASK2DATASET["detect"], splits=("train",), settings=settings, batch=2, device="cpu", workers=0)

    run = _get_run_from_settings(settings)
    default_tables = get_metrics_tables_from_run(run)["default_stream"]
    assert len(default_tables) >= 2, "Expected the zero buffer threshold to flush multiple metrics tables"

    df = pd.concat([t.to_pandas() for t in default_tables], ignore_index=True)
    assert sorted(df["example_id"].tolist()) == [0, 1, 2, 3], "Rewritten tables should cover every image exactly once"

    assert "embeddings_pca" in df.columns, "Expected the reduced image-embeddings column"
    assert "embeddings" not in df.columns, "The raw image-embeddings column should have been rewritten"
    for emb in df["embeddings_pca"]:
        assert len(emb) == dim


def test_metrics_flushing_segment_all_embeddings() -> None:
    """The bug-report scenario: segmentation masks plus image, predicted and ground-truth instance embeddings,
    with a zero buffer threshold. Every flushed table is rewritten with all three reduced columns while the
    heavy mask column is carried through the rewrite."""
    dim = 2
    settings = Settings(
        project_name="test_metrics_flushing_seg_all",
        run_name="test_metrics_flushing_seg_all",
        metrics_max_buffer_mb=0,  # flush after every batch
        image_embeddings_dim=dim,
        image_embeddings_reducer="pca",
        instance_embeddings_dim=dim,
        instance_embeddings_reducer="pca",
        ground_truth_instance_embeddings=True,
        label_column_name=TASK2LABEL_COLUMN_NAME["segment"],
    )

    model = TLCYOLO(TASK2MODEL["segment"])
    model.collect(data=TASK2DATASET["segment"], splits=("train",), settings=settings, batch=2, device="cpu", workers=0)

    run = _get_run_from_settings(settings)
    default_tables = get_metrics_tables_from_run(run)["default_stream"]
    assert len(default_tables) >= 2, "Expected the zero buffer threshold to flush multiple metrics tables"

    df = pd.concat([t.to_pandas() for t in default_tables], ignore_index=True)
    assert sorted(df["example_id"].tolist()) == [0, 1, 2, 3], "Rewritten tables should cover every image exactly once"

    # The heavy mask column survived the rewrite alongside all three reduced embedding columns
    assert "segmentations_predicted" in df.columns, "Expected the predicted segmentations column"
    for column in ("embeddings_pca", "predicted_instance_embedding", "ground_truth_instance_embedding"):
        assert column in df.columns, f"Expected reduced column '{column}'"
    for raw_column in ("embeddings", "predicted_instance_embedding_raw", "ground_truth_instance_embedding_raw"):
        assert raw_column not in df.columns, f"Raw column '{raw_column}' should have been rewritten"

    for emb in df["embeddings_pca"]:
        assert len(emb) == dim
    for row_embs in df["predicted_instance_embedding"]:
        for emb in row_embs:
            assert len(emb) == dim


def _predicted_mask_facts(tables: list[tlc.Table]) -> dict[int, tuple[int, int, int]]:
    """Per example id, the (instance count, image height, image width) of `segmentations_predicted`.

    Read off the tables' arrow data, so the RLEs are never decoded into masks - which is the whole point of the
    rewrite this is used to check.
    """
    import pyarrow.compute as pc

    facts: dict[int, tuple[int, int, int]] = {}
    for table in tables:
        arrow_table = table._to_pyarrow_table()
        masks = arrow_table.column(PREDICTED_SEGMENTATIONS).combine_chunks()
        counts = pc.fill_null(pc.list_value_length(masks.field("rles")), 0).to_pylist()
        heights = masks.field("image_height").to_pylist()
        widths = masks.field("image_width").to_pylist()
        for example_id, count, height, width in zip(
            arrow_table.column(EXAMPLE_ID).to_pylist(), counts, heights, widths, strict=True
        ):
            facts[example_id] = (count, height, width)
    return facts


def test_reduce_and_rewrite_all_empty_keeps_raw_tables() -> None:
    """When the rewrite produces nothing, the raw metrics tables must stay registered on the run and on disk.

    `_reduce_and_rewrite_raw_tables` returning None is the signal for that; returning an empty list of metrics
    infos instead would make the caller deregister and delete the run's only metrics tables.
    """
    from tlc_ultralytics.engine.validator import TLCValidatorMixin

    settings = Settings(
        project_name="test_rewrite_all_empty",
        run_name="test_rewrite_all_empty",
        image_embeddings_dim=2,
        image_embeddings_reducer="pca",
    )

    model = TLCYOLO(TASK2MODEL["detect"])
    # Every raw table looks empty to the rewrite, so no reduced table is written for any of them.
    with patch.object(TLCValidatorMixin, "_rewrite_raw_table", lambda *args, **kwargs: []):
        model.collect(
            data=TASK2DATASET["detect"], splits=("train",), settings=settings, batch=2, device="cpu", workers=0
        )

    run = _get_run_from_settings(settings)
    default_tables = get_metrics_tables_from_run(run)["default_stream"]
    assert default_tables, "The raw metrics tables should still be registered on the run"
    for table in default_tables:
        assert table.url.exists(), f"The raw metrics table at {table.url} should not have been deleted"

    df = pd.concat([t.to_pandas() for t in default_tables], ignore_index=True)
    assert sorted(df[EXAMPLE_ID].tolist()) == [0, 1, 2, 3]
    assert "embeddings" in df.columns, "The raw image-embeddings column should have been kept"
    assert "embeddings_pca" not in df.columns, "No reduced column should have been written"


def test_metrics_rewrite_does_not_decode_masks() -> None:
    """The embedding rewrite must carry the RLE mask column through without decoding it.

    Decoding an RLE row (`SegmentationHelper.masks_from_rles`) inflates it to a dense (H, W, N) uint8 array at
    original image resolution, and re-encoding it (`rles_from_masks`) throws that away again - for a column the
    rewrite is only copying. Counting both while the rewrite runs locks the arrow-level passthrough in place.
    """
    from tlc.helpers.segmentation_helper import SegmentationHelper

    from tlc_ultralytics.engine.validator import TLCValidatorMixin

    calls = {"decode": 0, "encode": 0, "rewrites": 0}
    collected: dict[int, tuple[int, int, int]] = {}
    original_masks_from_rles = SegmentationHelper.masks_from_rles
    original_rles_from_masks = SegmentationHelper.rles_from_masks
    original_reduce = TLCValidatorMixin._reduce_and_rewrite_raw_tables
    rewriting = False

    def counting_masks_from_rles(*args, **kwargs):
        if rewriting:
            calls["decode"] += 1
        return original_masks_from_rles(*args, **kwargs)

    def counting_rles_from_masks(*args, **kwargs):
        if rewriting:
            calls["encode"] += 1
        return original_rles_from_masks(*args, **kwargs)

    def counting_reduce(self, raw_table_urls, *args, **kwargs):
        # Only mask work done by the rewrite itself counts; the pass that produced the raw tables encodes masks
        # legitimately, and reading the tables back afterwards decodes them again.
        nonlocal rewriting
        calls["rewrites"] += 1
        collected.update(_predicted_mask_facts([tlc.Table.from_url(url) for url in raw_table_urls]))
        rewriting = True
        try:
            return original_reduce(self, raw_table_urls, *args, **kwargs)
        finally:
            rewriting = False

    settings = Settings(
        project_name="test_metrics_rewrite_no_decode",
        run_name="test_metrics_rewrite_no_decode",
        metrics_max_buffer_mb=0,  # flush after every batch, so several tables are rewritten
        image_embeddings_dim=2,
        image_embeddings_reducer="pca",
        instance_embeddings_dim=2,
        instance_embeddings_reducer="pca",
        ground_truth_instance_embeddings=True,
        label_column_name=TASK2LABEL_COLUMN_NAME["segment"],
    )

    model = TLCYOLO(TASK2MODEL["segment"])
    with (
        patch.object(SegmentationHelper, "masks_from_rles", staticmethod(counting_masks_from_rles)),
        patch.object(SegmentationHelper, "rles_from_masks", staticmethod(counting_rles_from_masks)),
        patch.object(TLCValidatorMixin, "_reduce_and_rewrite_raw_tables", counting_reduce),
    ):
        model.collect(
            data=TASK2DATASET["segment"],
            splits=("train",),
            settings=settings,
            batch=2,
            device="cpu",
            workers=0,
        )

    assert calls["rewrites"] == 1, "Expected the rewrite to run once for the single collected split"
    assert calls["decode"] == 0, "The rewrite decoded RLE masks instead of copying the column through"
    assert calls["encode"] == 0, "The rewrite re-encoded masks instead of copying the column through"

    # ... and the masks came through unchanged: same instance count and same (H, W) per image
    run = _get_run_from_settings(settings)
    default_tables = get_metrics_tables_from_run(run)["default_stream"]
    rewritten = _predicted_mask_facts(default_tables)
    assert sorted(rewritten) == [0, 1, 2, 3], "Rewritten tables should cover every image exactly once"
    assert rewritten == collected, "The rewritten mask column differs from what was collected"
    assert any(count > 0 for count, _, _ in rewritten.values()), "Expected at least one predicted mask to compare"


def test_instance_embeddings_cross_split_shared_space() -> None:
    """Multi-split collect: train fits the reducer, val is transformed into the same space."""
    dim = 2
    settings = Settings(
        project_name="test_instance_emb_cross_split",
        run_name="test_instance_emb_cross_split",
        instance_embeddings_dim=dim,
        instance_embeddings_reducer="pca",
        label_column_name=TASK2LABEL_COLUMN_NAME["detect"],
    )

    model = TLCYOLO(TASK2MODEL["detect"])
    model.collect(data=TASK2DATASET["detect"], splits=("train", "val"), settings=settings, **INSTANCE_EMB_OVERRIDES)

    run = _get_run_from_settings(settings)
    metrics_tables = get_metrics_tables_from_run(run)
    default_tables = metrics_tables["default_stream"]
    assert len(default_tables) >= 2, "Expected one metrics table per split"

    for table in default_tables:
        df = table.to_pandas()
        assert "predicted_instance_embedding" in df.columns
        for row_embs in df["predicted_instance_embedding"]:
            for emb in row_embs:
                assert len(emb) == dim

    # The run's reducers must not leak past collect()
    from tlc_ultralytics.utils._instance_reduce import _get_fitted_reducer

    assert _get_fitted_reducer(run.url.to_str(), "instance") is None
    assert _get_fitted_reducer(run.url.to_str(), "image") is None


def test_instance_embeddings_explicit_layer() -> None:
    """Collect instance embeddings from an explicit neck layer instead of the cls-head default.

    Exercises the instance_embeddings_layer path (_add_feature_map_hook / _infer_layer_channels),
    which the default cls-head tests do not cover.
    """
    from tlc_ultralytics.utils.embeddings import _auto_detect_p3_layer

    dim = 2
    model = TLCYOLO(TASK2MODEL["detect"])
    # Pick a valid neck layer the same way the auto-detect default does, so the index isn't
    # hardcoded against a specific model architecture.
    layer_index = _auto_detect_p3_layer(model.model.model)

    settings = Settings(
        project_name="test_instance_emb_explicit_layer",
        run_name="test_instance_emb_explicit_layer",
        instance_embeddings_dim=dim,
        instance_embeddings_layer=layer_index,
        instance_embeddings_reducer="pca",
        label_column_name=TASK2LABEL_COLUMN_NAME["detect"],
    )

    model.collect(data=TASK2DATASET["detect"], splits=("train",), settings=settings, **INSTANCE_EMB_OVERRIDES)

    run = _get_run_from_settings(settings)
    metrics_df = pd.concat(
        [m.to_pandas() for m in get_metrics_tables_from_run(run)["default_stream"]], ignore_index=True
    )
    assert "predicted_instance_embedding" in metrics_df.columns
    for row_embs in metrics_df["predicted_instance_embedding"]:
        for emb in row_embs:
            assert len(emb) == dim


def test_instance_embeddings_warns_without_predictions() -> None:
    """With no predictions passing the confidence threshold, a clear warning is logged and the
    embedding columns are empty rather than raising."""
    settings = Settings(
        project_name="test_instance_emb_no_preds",
        run_name="test_instance_emb_no_preds",
        instance_embeddings_dim=2,
        instance_embeddings_reducer="pca",
        conf_thres=1.0,  # confidences are strictly < 1.0, so nothing passes the filter
        label_column_name=TASK2LABEL_COLUMN_NAME["detect"],
    )
    model = TLCYOLO(TASK2MODEL["detect"])

    with patch("tlc_ultralytics.engine.validator.LOGGER") as mock_logger:
        model.collect(data=TASK2DATASET["detect"], splits=("train",), settings=settings, **INSTANCE_EMB_OVERRIDES)

    warnings = [str(call.args[0]) for call in mock_logger.warning.call_args_list if call.args]
    assert any("No predicted instances were available" in w for w in warnings), warnings


# Semantic segmentation

SEMANTIC_OVERRIDES = {"device": "cpu", "imgsz": 256, "batch": 2, "workers": 0}
"""Keep semantic tests fast: cityscapes8 images are 2048x1024."""


def _semantic_table(name: str, masks: list[np.ndarray], **kwargs) -> tlc.Table:
    """Write a semantic segmentation table with one blank image per mask, of the mask's size."""
    images = []
    for i, mask in enumerate(masks):
        image_path = TMP / f"{name}_{i}.png"
        Image.new("RGB", (mask.shape[1], mask.shape[0])).save(image_path)
        images.append(image_path.as_posix())
    return tlc.Table.from_semantic_segmentation(
        images=images,
        masks=masks,
        project_name=f"test_{name}",
        dataset_name=name,
        table_name="initial",
        if_exists="overwrite",
        **kwargs,
    )


def test_semantic_tables_from_yaml() -> None:
    # Tables created from an Ultralytics semantic dataset hold exactly the masks Ultralytics trains on, in its class
    # indices, with its ignore label as 3LC's void class.
    from tlc.data_types.semantic_segmentation import TLC_SEMSEG_VOID
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    from tlc_ultralytics.utils.dataset import create_tables_from_yaml_file

    tables = create_tables_from_yaml_file(
        TASK2DATASET["semantic"], task="semantic", project_name="test_semantic_tables_from_yaml", splits=("val",)
    )
    table = tables["val"]
    assert table.rows_schema.values["mask"].sample_type == "semantic_segmentation"

    data = check_det_dataset(TASK2DATASET["semantic"])
    value_map = table.get_value_map("mask.instance_properties.label")
    assert {int(k): v.internal_name for k, v in value_map.items() if int(k) != 255} == data["names"]
    assert value_map[255].internal_name == TLC_SEMSEG_VOID, "Ultralytics' ignore label must be 3LC's void class"

    dataset = build_yolo_dataset(get_cfg(overrides={"task": "semantic"}), data["val"], 1, data, mode="val")
    index_by_image = {Path(f).name: i for i, f in enumerate(dataset.im_files)}
    for i, row in enumerate(table.table_rows):
        expected = dataset.load_mask(index_by_image[Path(row["image"]).name])
        assert np.array_equal(table[i]["mask"].mask, expected), f"Row {i} differs from Ultralytics' mask"


def test_semantic_classes_from_yaml() -> None:
    # Polygon datasets gain a background class from `add_polygon_background`, which becomes 3LC's background. Binary
    # datasets name only the foreground, painted 1 on a background of 0.
    from ultralytics.data.utils import add_polygon_background

    from tlc_ultralytics.semantic.utils import semantic_classes_from_yaml

    polygons = add_polygon_background({"names": {0: "a", 1: "b"}, "nc": 2})
    assert semantic_classes_from_yaml(polygons) == ({0: "a", 1: "b", 2: "background", 255: "ignore"}, 2, 255)

    binary = add_polygon_background({"names": {0: "crack"}, "nc": 1})
    assert semantic_classes_from_yaml(binary) == ({0: "background", 1: "crack", 255: "ignore"}, 0, 255)

    masks = add_polygon_background({"names": {0: "road", 1: "car"}, "nc": 2, "masks_dir": "masks"})
    assert semantic_classes_from_yaml(masks) == ({0: "road", 1: "car", 255: "ignore"}, None, 255)


def test_semantic_class_mapping_and_masks() -> None:
    # Non-contiguous 3LC ids map to contiguous training indices in id order, the background (dropped from 3LC's value
    # map) is trained as a class of its own, and void pixels are ignored. Decoded masks match 3LC's own decoding.
    from tlc_ultralytics.semantic.utils import IGNORE_INDEX

    rng = np.random.default_rng(0)
    ids = np.array([2, 7, 9, 255])  # 2 is the background, 255 void
    masks = [ids[rng.integers(0, len(ids), size=shape)] for shape in ((24, 40), (32, 18))]
    table = _semantic_table(
        "semantic_mapping", masks, classes={2: "bg", 7: "road", 9: "car", 255: "border"}, background=2, void=255
    )

    data = check_tlc_dataset("", {"val": table}, "image", None, task="semantic", splits=("val",))
    assert data["names"] == {0: "background", 1: "road", 2: "car"}
    assert data["range_to_3lc_class"] == {0: 2, 1: 7, 2: 9}
    assert (data["semantic_background"], data["semantic_void"]) == (2, 255)
    assert {int(k) for k in data["names_3lc"]} == {7, 9}, "The predicted column has neither background nor void"

    dataset = TLCYOLODataset(
        table,
        task="semantic",
        data=data,
        class_map=data["3lc_class_to_range"],
        image_column_name="image",
        label_column_name="mask",
        augment=False,
    )
    lut = np.full(256, -1)
    lut[[2, 7, 9, 255]] = [0, 1, 2, IGNORE_INDEX]
    for i in range(len(table)):
        index = next(j for j, label in enumerate(dataset.labels) if label["example_id"] == i)
        assert np.array_equal(dataset.load_mask(index), lut[table[i]["mask"].mask]), f"Row {i} decoded differently"

    # A validation sample carries its load-time resized shape for the validator, and none of the compact segmentation
    # it was decoded from.
    sample = dataset[0]
    assert "tlc_resized_shape" in sample
    assert not any(key.startswith("tlc_semantic") for key in sample)


def test_semantic_dataset_without_data() -> None:
    # Without `data` the column's background and void are unknown: pixels no layer covers are ignored, and the void
    # layer is an ordinary one. Layer ids still map through a class map.
    from tlc_ultralytics.semantic.utils import IGNORE_INDEX

    ids = np.array([0, 1, 2, 5])  # 0 is the background, 5 void
    mask = ids[np.random.default_rng(0).integers(0, len(ids), size=(12, 20))]
    table = _semantic_table(
        "semantic_without_data", [mask], classes={0: "bg", 1: "a", 2: "b", 5: "border"}, background=0, void=5
    )

    for class_map, mapped in ((None, [1, 2, 5]), ({1: 0, 2: 1, 5: 2}, [0, 1, 2])):
        dataset = TLCYOLODataset(
            table,
            task="semantic",
            class_map=class_map,
            image_column_name="image",
            label_column_name="mask",
            augment=False,
        )
        lut = np.zeros(256, dtype=np.uint8)
        lut[ids] = [IGNORE_INDEX, *mapped]
        assert np.array_equal(dataset.load_mask(0), lut[mask]), f"Decoded differently with class_map={class_map}"


def test_semantic_needs_two_classes() -> None:
    table = _semantic_table("semantic_one_class", [np.ones((16, 16), dtype=np.int32)], classes={1: "only"})
    with pytest.raises(ValueError, match="at least two are needed"):
        check_tlc_dataset("", {"val": table}, "image", None, task="semantic", splits=("val",))


def test_segment_rejects_semantic_table() -> None:
    # A semantic segmentation column is stored in the instance segmentation RLE layout. The instance segmentation task
    # must neither infer it nor accept it when named explicitly.
    table = _semantic_table(
        "segment_rejects_semantic", [np.zeros((16, 16), dtype=np.int32)], classes={0: "bg", 1: "a"}, background=0
    )
    with pytest.raises(ValueError, match="task='semantic'"):
        check_seg_table(table, "image", None)
    with pytest.raises(ValueError, match="task='semantic'"):
        check_seg_table(table, "image", "mask")


def test_segment_infers_instance_column_next_to_semantic_column() -> None:
    # 3LC matches a semantic segmentation column as instance segmentation too. Next to one instance segmentation column
    # (not named `segmentations`), segment inference must pick the instance column rather than find two, and the
    # semantic task the semantic one.
    from tlc.data_types import SegmentationPolygons
    from tlc.schemas import SemanticSegmentationRleSchema

    from tlc_ultralytics.utils.dataset import resolve_annotation_label_path

    semantic_schema = SemanticSegmentationRleSchema(classes={0: "bg", 1: "a"}, background=0)
    instances = SegmentationPolygons(
        image_width=16, image_height=16, polygons=[[2.0, 2.0, 10.0, 2.0, 10.0, 10.0]], labels=[0]
    ).to_row()
    table = tlc.Table.from_dict(
        {
            "image": [str(DUMMY_IMAGE_FILE)],
            "instances": [instances],
            "mask": [semantic_schema.to_row(np.eye(16, dtype=np.int32))],
        },
        schema={
            "instances": SegmentationPolygons.schema(classes={0: tlc.schemas.MapElement("object")}),
            "mask": semantic_schema,
        },
        project_name="test_segment_infers_instance_column",
        dataset_name="d",
        table_name="instances_and_mask",
        if_exists="overwrite",
    )

    assert resolve_annotation_label_path(table, None, "segment") == "instances.instance_properties.label"
    check_seg_table(table, "image", None)
    data = check_tlc_dataset("", {"val": table}, "image", None, task="semantic", splits=("val",))
    assert data["names"] == {0: "background", 1: "a"}


def test_semantic_tables_from_yaml_reject_kwargs() -> None:
    from tlc_ultralytics.utils.dataset import create_tables_from_yaml_file

    with pytest.raises(TypeError, match="task='semantic': categories"):
        create_tables_from_yaml_file("unused.yaml", task="semantic", categories={})


def test_semantic_prediction_row_matches_tlc_encoding() -> None:
    # The validator writes predictions in row form, encoded per chunk of classes. The row must be exactly what 3LC
    # produces from the dense map in 3LC ids, and read back as the same map, next to a sample-form value.
    import torch
    from tlc.constants import RLES
    from tlc.schemas import SemanticSegmentationRleSchema

    from tlc_ultralytics.semantic.validator import TLCSemanticSegmentationValidator

    ids = [2, 7, 9, 11]  # training index -> 3LC id, 2 is the background
    h, w = 13, 17
    class_map = torch.from_numpy(np.random.default_rng(0).integers(0, len(ids), size=(h, w)))
    class_map[0] = 3  # a class present in one row only
    dense = np.array(ids)[class_map.numpy()]

    validator = TLCSemanticSegmentationValidator.__new__(TLCSemanticSegmentationValidator)
    validator.data = {"semantic_background": 2}
    validator._index_to_3lc_class = ids
    validator._chunk_pixels = 2 * h * w  # two classes per chunk
    row = validator._prediction_row(class_map, h, w)

    schema = SemanticSegmentationRleSchema(classes={7: "a", 9: "b", 11: "c"}, background=2)
    assert row == schema.to_row(dense), "The row form differs from what 3LC encodes from the dense map"
    assert len(row[RLES]) == 3, "The background is the fill, not a layer"

    run = tlc.init(project_name="test_semantic_prediction_row", run_name="test_semantic_prediction_row")
    writer = tlc.MetricsTableWriter(run_url=run.url, foreign_table_url=run.url, schema={"seg": schema})
    writer.add_batch({"example_id": [0, 1], "seg": [row, dense]})
    table = writer.finalize()
    for i in range(2):
        assert np.array_equal(table[i]["seg"].mask, dense), f"Row {i} does not read back as the dense map"


def test_semantic_class_map_chunking_matches_argmax() -> None:
    # Upsampling a few classes at a time with a running maximum must give the argmax over all classes at once.
    import torch
    import torch.nn.functional as F
    from ultralytics.utils import ops

    from tlc_ultralytics.semantic.validator import TLCSemanticSegmentationValidator

    torch.manual_seed(0)
    logits = torch.randn(7, 8, 12)
    imgsz, ori_shape = (64, 96), (70, 150)
    reference = ops.scale_masks(F.interpolate(logits[None], imgsz, mode="bilinear"), ori_shape)[0].argmax(0)

    validator = TLCSemanticSegmentationValidator.__new__(TLCSemanticSegmentationValidator)
    validator._chunk_pixels = 2 * imgsz[0] * imgsz[1]  # two classes per chunk
    assert torch.equal(validator._class_map_at_original_resolution(logits, imgsz, ori_shape, None).long(), reference)


def test_semantic_collect() -> None:
    # End-to-end collection with the pretrained model: one prediction per image at its original resolution, with
    # per-sample losses and a per-class metrics table. At imgsz 256 cityscapes8's 2048x1024 images scale by exactly
    # 1/8, so validation and `model.predict` letterbox them identically and the predictions must be equal.
    import ultralytics
    from packaging.version import Version
    from ultralytics import YOLO as UltralyticsYOLO

    settings = Settings(
        project_name="test_semantic_collect",
        run_name="test_semantic_collect",
        collect_loss=True,
        instance_embeddings_dim=2,
    )
    model = TLCYOLO(TASK2MODEL["semantic"])
    with capture_logs() as log_messages:
        model.collect(data=TASK2DATASET["semantic"], splits=("val",), settings=settings, **SEMANTIC_OVERRIDES)
    assert any("Instance embeddings are not supported for the 'semantic' task" in m for m in log_messages)

    run = _get_run_from_settings(settings)
    metrics_tables = get_metrics_tables_from_run(run)
    (metrics_table,) = metrics_tables["default_stream"]
    assert not any("embedding" in column for column in metrics_table.rows_schema.values)

    input_table = tlc.Table.from_url(run.url / run.constants["inputs"][0]["input_table_url"])
    assert len(metrics_table) == len(input_table) == 4

    # Before 8.4.57 Ultralytics' predictor resized the stride-8 logits straight to the original image, so the letterbox
    # padding it cropped was off by a fraction of a pixel. Collection follows the fixed predictor on every version.
    exact = Version(ultralytics.__version__) >= Version("8.4.57")
    predictor = UltralyticsYOLO(TASK2MODEL["semantic"])
    for row in metrics_table:
        prediction = row[PREDICTED_SEMANTIC_SEGMENTATION].mask
        image = input_table.table_rows[row["example_id"]]["image"]
        expected = predictor.predict(image, imgsz=SEMANTIC_OVERRIDES["imgsz"], device="cpu", verbose=False)[0]
        agreement = (prediction == expected.semantic_mask.data.cpu().numpy()).mean()
        # Bilinear interpolation rounds differently across CPU architectures (e.g. macOS arm64 vs. Linux x86_64),
        # occasionally flipping the argmax of a near-tie pixel score. That shows up as a handful of pixels out of a
        # few million, nowhere near the ~1-2% a geometry bug (e.g. a one-row letterbox shift) would produce, so this
        # threshold stays far tighter than the pre-8.4.57 tolerance below.
        assert agreement > 0.999 if exact else agreement > 0.98, f"Prediction differs from predict(): {agreement:.4f}"
        assert row["loss"] == pytest.approx(row["ce_loss"] + row["dice_loss"], rel=1e-5)

    (per_class_table,) = metrics_tables[PER_CLASS_METRICS_STREAM_NAME]
    assert len(per_class_table) == len(model.names) + 1, "One row per class plus 'all'"
    assert {"iou", "pixel_accuracy", "num_pixels", NUM_IMAGES} <= set(per_class_table.rows_schema.values)


def test_semantic_training() -> None:
    # Training on 3LC tables matches plain Ultralytics training on the same data, with metrics collected during and
    # after training.
    overrides = {
        "data": TASK2DATASET["semantic"],
        "epochs": 2,
        "plots": False,
        "seed": 3 + ord("L") + ord("C"),
        "deterministic": True,
        **SEMANTIC_OVERRIDES,
    }
    settings = Settings(
        collection_epoch_start=1,
        project_name="test_semantic_project",
        run_name="test_semantic",
        collect_loss=True,
    )

    results_ultralytics = YOLO(TASK2MODEL["semantic"]).train(**overrides)
    model_3lc = TLCYOLO(TASK2MODEL["semantic"])
    results_3lc = model_3lc.train(**overrides, settings=settings)

    for k in results_ultralytics.results_dict:
        assert np.isclose(results_ultralytics.results_dict[k], results_3lc.results_dict[k], atol=0.01), k
    assert results_ultralytics.names == results_3lc.names

    run = _get_run_from_settings(settings)
    assert run.status == RUN_STATUS_COMPLETED
    assert "val_mIoU" in run.constants["outputs"][-1]

    metrics_tables = get_metrics_tables_from_run(run)
    # Train and val after each of the two epochs and after training
    assert len(metrics_tables["default_stream"]) == len(metrics_tables[PER_CLASS_METRICS_STREAM_NAME]) == 6
    metrics_df = pd.concat([t.to_pandas() for t in metrics_tables["default_stream"]], ignore_index=True)
    assert {PREDICTED_SEMANTIC_SEGMENTATION, "loss", EPOCH, TRAINING_PHASE} <= set(metrics_df.columns)


def test_semantic_prediction_inverts_letterbox_exactly() -> None:
    # Validation resizes an image on load, rounding up, before `LetterBox` pads it, rounding to nearest. Predictions
    # must be cropped out of exactly the padded image's content: at 333x500 and imgsz 256 the padding derived from the
    # original shape alone misses the content by a row.
    import torch
    import torch.nn.functional as F
    from ultralytics.utils import ops

    from tlc_ultralytics.semantic.dataset import RESIZED_SHAPE
    from tlc_ultralytics.semantic.validator import TLCSemanticSegmentationValidator

    ori_shape = (333, 500)
    table = _semantic_table("semantic_letterbox", [np.ones(ori_shape, dtype=np.int32)], classes={0: "a", 1: "b"})
    data = check_tlc_dataset("", {"val": table}, "image", None, task="semantic", splits=("val",))
    dataset = TLCYOLODataset(
        table,
        task="semantic",
        data=data,
        class_map=data["3lc_class_to_range"],
        image_column_name="image",
        label_column_name="mask",
        augment=False,
        imgsz=256,
        rect=True,
        batch_size=1,
        pad=0.0,
    )
    sample = dataset[0]
    imgsz = tuple(sample["img"].shape[1:])
    rows, cols = np.nonzero(sample["semantic_mask"].numpy() != 255)  # the content, all of class "b"

    torch.manual_seed(0)
    logits = torch.randn(3, *imgsz)
    content = logits[:, rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    expected = F.interpolate(content[None], ori_shape, mode="bilinear")[0].argmax(0)
    assert not torch.equal(ops.scale_masks(logits[None], ori_shape)[0].argmax(0), expected), "Expected rounding to bite"

    validator = TLCSemanticSegmentationValidator.__new__(TLCSemanticSegmentationValidator)
    validator._chunk_pixels = 2 * ori_shape[0] * ori_shape[1]  # two classes per chunk
    ratio_pad = validator._letterbox_ratio_pad(sample[RESIZED_SHAPE], imgsz)
    class_map = validator._class_map_at_original_resolution(logits, imgsz, ori_shape, ratio_pad)
    assert class_map.dtype == torch.uint8
    assert torch.equal(class_map.long(), expected), "Prediction is not cropped out of exactly the letterbox content"


def test_semantic_needs_at_most_255_classes() -> None:
    table = _semantic_table(
        "semantic_too_many_classes", [np.zeros((16, 16), dtype=np.int32)], classes={i: f"c{i}" for i in range(256)}
    )
    with pytest.raises(ValueError, match="at most 255"):
        check_tlc_dataset("", {"val": table}, "image", None, task="semantic", splits=("val",))


def test_semantic_and_segment_tables_from_same_yaml() -> None:
    # Ultralytics reads instance segmentation YAMLs as polygon semantic segmentation datasets too. Each task gets tables
    # of its own from the same YAML, in either order.
    from tlc_ultralytics.utils.dataset import create_tables_from_yaml_file

    project_name = "test_semantic_and_segment_tables"
    segment = create_tables_from_yaml_file(
        TASK2DATASET["segment"], task="segment", project_name=project_name, splits=("val",)
    )["val"]
    semantic = create_tables_from_yaml_file(
        TASK2DATASET["segment"], task="semantic", project_name=project_name, splits=("val",)
    )["val"]
    assert segment.url != semantic.url
    assert semantic.dataset_name == "coco8-seg-semantic-val"
    assert semantic.rows_schema.values["mask"].sample_type == "semantic_segmentation"

    # Polygons are rasterized onto a background class, which becomes the column's background.
    data = check_tlc_dataset("", {"val": semantic}, "image", None, task="semantic", splits=("val",))
    assert data["semantic_background"] == 80 and data["names"][80] == "background"
    check_seg_table(segment, "image", None)  # the instance segmentation table is still its own


def test_semantic_all_void_image_losses() -> None:
    # An image with no pixel other than void has nothing to compute a loss over. Its losses are written as 0 rather than
    # the NaN Ultralytics' cross-entropy gives it, while other images' losses are unaffected.
    names = YOLO(TASK2MODEL["semantic"]).names
    rng = np.random.default_rng(0)
    masks = [rng.integers(0, len(names), size=(48, 64)), np.full((48, 64), 255)]
    table = _semantic_table("semantic_all_void", masks, classes={**names, 255: "ignore"}, void=255)

    settings = Settings(
        project_name="test_semantic_all_void_image_losses",
        run_name="test_semantic_all_void_image_losses",
        collect_loss=True,
    )
    TLCYOLO(TASK2MODEL["semantic"]).collect(
        tables={"val": table}, settings=settings, **{**SEMANTIC_OVERRIDES, "imgsz": 64}
    )

    run = _get_run_from_settings(settings)
    (metrics_table,) = get_metrics_tables_from_run(run)["default_stream"]
    rows = {row[EXAMPLE_ID]: row for row in metrics_table.table_rows}
    assert rows[1]["ce_loss"] == rows[1]["dice_loss"] == rows[1]["loss"] == 0.0, "An all-void image has no loss"
    assert np.isfinite(rows[0]["ce_loss"]) and rows[0]["ce_loss"] > 0
    assert rows[0]["loss"] == pytest.approx(rows[0]["ce_loss"] + rows[0]["dice_loss"], rel=1e-5)


def test_semantic_training_from_tables() -> None:
    # Training from tables alone, with plots enabled, collects metrics and reports the tables' classes.
    from tlc_ultralytics.utils.dataset import create_tables_from_yaml_file

    project_name = "test_semantic_training_from_tables"
    tables = create_tables_from_yaml_file(
        TASK2DATASET["semantic"], task="semantic", project_name=project_name, splits=("train", "val")
    )
    settings = Settings(
        project_name=project_name,
        run_name="test_semantic_training_from_tables",
        collection_epoch_start=1,
        collect_loss=True,
    )
    model = TLCYOLO(TASK2MODEL["semantic"])
    with capture_logs() as log_messages:
        results = model.train(tables=tables, settings=settings, epochs=1, **{**SEMANTIC_OVERRIDES, "imgsz": 64})

    run = _get_run_from_settings(settings)
    assert run.status == RUN_STATUS_COMPLETED
    assert "val_mIoU" in run.constants["outputs"][-1]
    # The trainer says why it weights the training loss, once
    reason = "Weighting the cross-entropy loss with Ultralytics' Cityscapes class weights, from the table's recorded"
    assert sum(reason in m for m in log_messages) == 1
    # The label plot is drawn from the table's masks, where Ultralytics would look for mask files and skip it
    assert (Path(model.trainer.save_dir) / "labels.jpg").exists()
    # The tables were created from cityscapes8, so training weights the loss as training through `data=` would, and so
    # does the EMA model's validation loss
    import torch
    from ultralytics.utils.metrics import CITYSCAPES_WEIGHT

    weight = torch.from_numpy(CITYSCAPES_WEIGHT)
    for trained_model in (model.trainer.model, model.trainer.ema.ema):
        assert trained_model.criterion.use_cityscapes_weight
        assert torch.equal(trained_model.criterion.ce.weight.cpu(), weight.to(trained_model.criterion.ce.weight.dtype))
    # The criterion is Ultralytics' own loss, so checkpoints reference no `tlc_ultralytics` class
    assert type(model.trainer.model.criterion).__module__ == "ultralytics.utils.loss"
    assert b"tlc_ultralytics" not in Path(model.trainer.last).read_bytes()

    metrics_tables = get_metrics_tables_from_run(run)
    # Train and val after the epoch and after training
    assert len(metrics_tables["default_stream"]) == len(metrics_tables[PER_CLASS_METRICS_STREAM_NAME]) == 4

    data = check_tlc_dataset("", tables, "image", None, task="semantic", splits=("train", "val"))
    assert results.names == data["names"]


def test_semantic_polygon_background_end_to_end() -> None:
    # A polygon dataset's background is trained as a class of its own, but is the fill of the predicted segmentation,
    # never a layer of it.
    from tlc.constants import INSTANCE_PROPERTIES
    from tlc.constants import LABEL as TLC_LABEL

    from tlc_ultralytics.utils.dataset import create_tables_from_yaml_file

    project_name = "test_semantic_polygon_background"
    tables = create_tables_from_yaml_file(
        TASK2DATASET["segment"], task="semantic", project_name=project_name, splits=("train", "val")
    )
    data = check_tlc_dataset("", tables, "image", None, task="semantic", splits=("train", "val"))
    assert data["semantic_background"] == 80 and data["names"][80] == "background"

    settings = Settings(
        project_name=project_name,
        run_name="test_semantic_polygon_background",
        collection_epoch_start=1,
        collection_val_only=True,
    )
    TLCYOLO(TASK2MODEL["semantic"]).train(
        tables=tables, settings=settings, epochs=1, plots=False, **{**SEMANTIC_OVERRIDES, "imgsz": 64}
    )

    run = _get_run_from_settings(settings)
    metrics_tables = get_metrics_tables_from_run(run)
    assert metrics_tables["default_stream"], "No metrics were collected"
    val_table = tables["val"]
    num_layers = 0
    for metrics_table in metrics_tables["default_stream"]:
        value_map = metrics_table.get_value_map(f"{PREDICTED_SEMANTIC_SEGMENTATION}.{INSTANCE_PROPERTIES}.{TLC_LABEL}")
        for i, row in enumerate(metrics_table.table_rows):
            labels = row[PREDICTED_SEMANTIC_SEGMENTATION][INSTANCE_PROPERTIES][TLC_LABEL]
            num_layers += len(labels)
            assert 80 not in labels, "The background is the fill, not a layer"
            assert all(label in value_map for label in labels)

            image = tlc.Url(val_table.table_rows[row[EXAMPLE_ID]]["image"]).to_absolute().to_str()
            width, height = Image.open(image).size
            assert metrics_table[i][PREDICTED_SEMANTIC_SEGMENTATION].mask.shape == (height, width)
    assert num_layers > 0, "Expected some predicted classes besides the background"

    per_class_table = metrics_tables[PER_CLASS_METRICS_STREAM_NAME][-1]
    assert len(per_class_table) == 82, "One row per class (80 plus the background) plus 'all'"


def test_semantic_tables_record_ultralytics_dataset() -> None:
    # Tables created from an Ultralytics YAML record its stem in the mask column's schema metadata, next to 3LC's own,
    # and the dataset dict reports it. Tables written directly with 3LC record none.
    from tlc_ultralytics.semantic.utils import ultralytics_dataset_from_table
    from tlc_ultralytics.utils.dataset import create_tables_from_yaml_file

    tables = create_tables_from_yaml_file(
        TASK2DATASET["semantic"], task="semantic", project_name="test_semantic_ultralytics_dataset", splits=("val",)
    )
    table = tlc.Table.from_url(tables["val"].url)
    assert table.rows_schema.values["mask"].metadata["tlc_ultralytics"] == {"ultralytics_dataset": "cityscapes8"}
    assert ultralytics_dataset_from_table(table, "mask") == "cityscapes8"
    assert "weight" in table.rows_schema.values, "The table has a sample weight column, like any 3LC table"

    data = check_tlc_dataset("", tables, "image", None, task="semantic", splits=("val",))
    assert data["ultralytics_dataset"] == "cityscapes8"

    other = _semantic_table("semantic_no_dataset", [np.zeros((16, 16), dtype=np.int32)], classes={0: "a", 1: "b"})
    assert ultralytics_dataset_from_table(other, "mask") is None
    data = check_tlc_dataset("", {"val": other}, "image", None, task="semantic", splits=("val",))
    assert data["ultralytics_dataset"] is None


def test_semantic_dataset_class_weights() -> None:
    # Cityscapes tables weight the cross-entropy with Ultralytics' Cityscapes class weights, registered as Ultralytics
    # does, whatever `model.args.data` says (a dict without it on a model loaded from a checkpoint). Other datasets,
    # and models without Cityscapes' 19 classes, stay unweighted.
    import torch
    from ultralytics.nn.tasks import SemanticSegmentationModel
    from ultralytics.utils.loss import SemanticSegmentationLoss
    from ultralytics.utils.metrics import CITYSCAPES_WEIGHT

    from tlc_ultralytics.semantic.utils import apply_dataset_class_weights

    model = YOLO(TASK2MODEL["semantic"]).model
    assert len(model.names) == len(CITYSCAPES_WEIGHT)

    loss = SemanticSegmentationLoss(model)
    assert not loss.use_cityscapes_weight, "Ultralytics does not weight a loaded model's loss"
    assert not apply_dataset_class_weights(loss, None)
    assert not apply_dataset_class_weights(loss, "coco8-seg")
    assert loss.ce.weight is None

    assert apply_dataset_class_weights(loss, "cityscapes8")
    assert loss.use_cityscapes_weight
    assert torch.equal(loss.ce.weight, torch.from_numpy(CITYSCAPES_WEIGHT).to(loss.dtype))
    assert "weight" not in loss.ce.state_dict(), "The weight is not persisted, as in Ultralytics"

    small = SemanticSegmentationModel("yolo26n-sem.yaml", nc=5, verbose=False)
    small.args = {}
    small_loss = SemanticSegmentationLoss(small)
    assert not apply_dataset_class_weights(small_loss, "cityscapes")
    assert small_loss.ce.weight is None

    # The trainer gives its model a criterion weighted from the tables' dataset, which `BaseModel.loss` uses rather than
    # building an unweighted one with `init_criterion`, as it does only while the model has no criterion
    from types import SimpleNamespace

    assert model.init_criterion().ce.weight is None, "Ultralytics' own criterion is left unweighted"
    for ultralytics_dataset, weighted in (("cityscapes8", True), (None, False)):
        trainer = TLCSemanticSegmentationTrainer.__new__(TLCSemanticSegmentationTrainer)
        trainer.model = deepcopy(model)
        trainer.args = SimpleNamespace(max_det=300)
        trainer.data = {"nc": len(model.names), "names": model.names, "ultralytics_dataset": ultralytics_dataset}
        trainer._settings = Settings()
        trainer.set_model_attributes()
        assert trainer.model.criterion.use_cityscapes_weight is weighted
        assert (trainer.model.criterion.ce.weight is not None) is weighted
        assert not hasattr(trainer.model, "tlc_ultralytics_dataset")


def test_semantic_collect_dataset_class_weights(monkeypatch) -> None:
    # Post-hoc collection on a Cityscapes table computes the per-sample `ce_loss` with the class weights Ultralytics
    # trains with, although a loaded model's `args` name no dataset.
    import torch
    from ultralytics.utils.metrics import CITYSCAPES_WEIGHT

    from tlc_ultralytics.semantic.validator import TLCSemanticSegmentationValidator

    loss_fns = []
    prepare_loss_fn = TLCSemanticSegmentationValidator._prepare_loss_fn

    def spy(self, model):
        prepare_loss_fn(self, model)
        loss_fns.append(self.loss_fn)

    monkeypatch.setattr(TLCSemanticSegmentationValidator, "_prepare_loss_fn", spy)

    settings = Settings(
        project_name="test_semantic_collect_dataset_class_weights",
        run_name="test_semantic_collect_dataset_class_weights",
        collect_loss=True,
    )
    with capture_logs() as log_messages:
        TLCYOLO(TASK2MODEL["semantic"]).collect(
            data=TASK2DATASET["semantic"], splits=("val",), settings=settings, **{**SEMANTIC_OVERRIDES, "imgsz": 64}
        )

    (loss_fn,) = loss_fns
    assert torch.equal(loss_fn.ce.weight.cpu(), torch.from_numpy(CITYSCAPES_WEIGHT).to(loss_fn.dtype))
    assert sum("Cityscapes class weights" in m for m in log_messages) == 1

    (metrics_table,) = get_metrics_tables_from_run(_get_run_from_settings(settings))["default_stream"]
    assert all(np.isfinite(row["ce_loss"]) and row["ce_loss"] > 0 for row in metrics_table.table_rows)


def test_semantic_accepts_differing_background_name() -> None:
    # 3LC stores only the background's id, so the tables name it "background" whatever its author called it. A model
    # whose name for it differs is accepted; every other class must match by name and index.
    from tlc_ultralytics.semantic.validator import TLCSemanticSegmentationValidator

    names = YOLO(TASK2MODEL["semantic"]).names
    table = _semantic_table(
        "semantic_background_name",
        [np.zeros((16, 16), dtype=np.int32)],
        classes={**names, 0: "unlabeled"},
        background=0,
    )
    validator = TLCSemanticSegmentationValidator.__new__(TLCSemanticSegmentationValidator)
    validator.data = check_tlc_dataset("", {"val": table}, "image", None, task="semantic", splits=("val",))
    assert validator.data["names"][0] == "background" != names[0]
    validator._verify_model_data_compatibility(names)

    # The per-class metrics table names the classes as the tables do, not as the model does
    from types import SimpleNamespace

    validator.names, validator.nc = names, len(names)
    validator.dataloader = SimpleNamespace(dataset=SimpleNamespace(table=table))
    validator._run = SimpleNamespace(url=tlc.Url(TMP / "semantic_background_name_run"))
    value_map = validator._per_class_base_schemas()[LABEL].value.map
    assert value_map[0].internal_name == "background"
    assert value_map[len(names)].internal_name == "all"
    assert all(value_map[i].internal_name == names[i] for i in range(1, len(names)))

    with pytest.raises(ValueError, match="1: model 'renamed', data"):
        validator._verify_model_data_compatibility({**names, 1: "renamed"})
    with pytest.raises(ValueError, match="trained on 18 classes"):
        validator._verify_model_data_compatibility({i: names[i] for i in range(18)})

    # The background's 3LC id maps through to its training index
    validator.data = {
        "names": {0: "a", 1: "background", 2: "b"},
        "semantic_background": 4,
        "3lc_class_to_range": {1: 0, 4: 1, 6: 2},
    }
    validator._verify_model_data_compatibility({0: "a", 1: "sky", 2: "b"})
    with pytest.raises(ValueError, match="0: model 'sky', data 'a'"):
        validator._verify_model_data_compatibility({0: "sky", 1: "background", 2: "b"})

    # Without a background, every name must match
    validator.data = {"names": {0: "a", 1: "b"}, "semantic_background": None, "3lc_class_to_range": {0: 0, 1: 1}}
    with pytest.raises(ValueError, match="0: model 'sky', data 'a'"):
        validator._verify_model_data_compatibility({0: "sky", 1: "b"})


def test_semantic_per_sample_losses_half_logits() -> None:
    # Ultralytics updates the metrics outside its autocast, so during AMP training the logits are float16 while the EMA
    # model, and with it the loss's Cityscapes class weights, stay float32. A half-precision model (`half=True`) gives
    # the loss float16 weights instead. The per-sample losses are computed in float32 either way.
    import torch
    from ultralytics.utils.loss import SemanticSegmentationLoss

    from tlc_ultralytics.semantic.utils import apply_dataset_class_weights
    from tlc_ultralytics.semantic.validator import TLCSemanticSegmentationValidator

    model = YOLO(TASK2MODEL["semantic"]).model
    loss = SemanticSegmentationLoss(model)
    assert apply_dataset_class_weights(loss, "cityscapes8")

    torch.manual_seed(0)
    logits = torch.randn(2, 19, 8, 8).half()
    batch = {"semantic_mask": torch.randint(0, 19, (2, 8, 8), dtype=torch.int32)}
    with pytest.raises(RuntimeError, match="Half"):
        loss(logits[:1], {"semantic_mask": batch["semantic_mask"][:1]})

    validator = TLCSemanticSegmentationValidator.__new__(TLCSemanticSegmentationValidator)
    validator.loss_fn = loss
    losses = validator._per_sample_losses(logits, batch)
    assert all(np.isfinite(value) and value > 0 for value in losses["ce_loss"] + losses["dice_loss"])
    _, expected = loss(logits[:1].float(), {"semantic_mask": batch["semantic_mask"][:1]})
    assert losses["ce_loss"][0] == pytest.approx(expected[0].item())

    validator._settings = Settings(collect_loss=True)
    validator.data = {"ultralytics_dataset": "cityscapes8"}
    validator.training = False
    validator._prepare_loss_fn(deepcopy(model).half())
    assert validator.loss_fn.ce.weight.dtype == torch.float32
    half_model_losses = validator._per_sample_losses(logits, batch)
    for key, values in losses.items():
        assert half_model_losses[key] == pytest.approx(values, rel=1e-3), key


def test_semantic_cityscapes_class_weights_setting() -> None:
    # `Settings.cityscapes_class_weights` overrides the decision from the table's recorded dataset: None follows it,
    # True forces the weights (for Cityscapes' 19 classes only) and False turns them off, including the ones Ultralytics
    # gives a loss itself when `model.args.data` names Cityscapes.
    from types import SimpleNamespace

    import torch
    from ultralytics.utils.loss import SemanticSegmentationLoss
    from ultralytics.utils.metrics import CITYSCAPES_WEIGHT

    from tlc_ultralytics.semantic.utils import (
        apply_dataset_class_weights,
        dataset_class_weights_message,
        uses_dataset_class_weights,
    )
    from tlc_ultralytics.semantic.validator import TLCSemanticSegmentationValidator

    assert uses_dataset_class_weights("cityscapes8", 19)
    assert not uses_dataset_class_weights("coco8-seg", 19)
    assert not uses_dataset_class_weights("cityscapes", 5)
    assert uses_dataset_class_weights(None, 19, True)
    assert uses_dataset_class_weights("coco8-seg", 19, True)
    assert not uses_dataset_class_weights("cityscapes8", 19, False)
    assert not uses_dataset_class_weights("cityscapes", 5, False)
    with pytest.raises(ValueError, match="19 classes"):
        uses_dataset_class_weights("cityscapes8", 5, True)

    loss_name = "the loss"
    assert dataset_class_weights_message("cityscapes8", 19, None, loss_name).endswith(
        "from the table's recorded dataset 'cityscapes8'"
    )
    assert dataset_class_weights_message(None, 19, True, loss_name).endswith(
        "forced by `Settings.cityscapes_class_weights=True`"
    )
    disabled = dataset_class_weights_message("cityscapes8", 19, False, loss_name)
    assert disabled.startswith("Not weighting") and "`Settings.cityscapes_class_weights=False`" in disabled
    assert dataset_class_weights_message("coco8-seg", 19, False, loss_name) is None
    assert dataset_class_weights_message("coco8-seg", 19, None, loss_name) is None

    model = YOLO(TASK2MODEL["semantic"]).model
    model.args = SimpleNamespace(data="cityscapes8.yaml")  # as while training through `data=`
    loss = SemanticSegmentationLoss(model)
    assert loss.use_cityscapes_weight, "Ultralytics weights the loss itself"
    assert not apply_dataset_class_weights(loss, "cityscapes8", False)
    assert not loss.use_cityscapes_weight and loss.ce.weight is None

    model.args = {}  # as on a model loaded from a checkpoint
    loss = SemanticSegmentationLoss(model)
    assert apply_dataset_class_weights(loss, None, True)
    assert torch.equal(loss.ce.weight, torch.from_numpy(CITYSCAPES_WEIGHT).to(loss.dtype))

    Settings(cityscapes_class_weights=False).verify(training=False)
    with pytest.raises(AssertionError, match="cityscapes_class_weights"):
        Settings(cityscapes_class_weights="yes").verify(training=False)

    # Forcing the weights for tables without 19 classes fails as soon as the dataset is checked
    table = _semantic_table("semantic_forced_weights", [np.zeros((16, 16), dtype=np.int32)], classes={0: "a", 1: "b"})
    validator = TLCSemanticSegmentationValidator.__new__(TLCSemanticSegmentationValidator)
    validator._settings = Settings(cityscapes_class_weights=True)
    with pytest.raises(ValueError, match="cityscapes_class_weights=True"):
        validator.check_dataset("", {"val": table}, "image", None, splits=("val",))


def test_semantic_prediction_row_without_background_matches_tlc_encoding() -> None:
    # Without a declared background, 3LC stores id 0 as a layer like any other class, and the validator's row form must
    # still be exactly what 3LC encodes from the dense map, reading back as the same map.
    import torch
    from tlc.constants import RLES
    from tlc.schemas import SemanticSegmentationRleSchema

    from tlc_ultralytics.semantic.validator import TLCSemanticSegmentationValidator

    ids = [0, 1, 2]
    h, w = 13, 17
    class_map = torch.from_numpy(np.random.default_rng(0).integers(0, len(ids), size=(h, w)))
    class_map[0] = 0  # id 0 fills a row
    dense = np.array(ids)[class_map.numpy()]

    validator = TLCSemanticSegmentationValidator.__new__(TLCSemanticSegmentationValidator)
    validator.data = {"semantic_background": None}
    validator._index_to_3lc_class = ids
    validator._chunk_pixels = 2 * h * w  # two classes per chunk
    row = validator._prediction_row(class_map, h, w)

    schema = SemanticSegmentationRleSchema(classes={0: "a", 1: "b", 2: "c"})
    assert row == schema.to_row(dense), "The row form differs from what 3LC encodes from the dense map"
    assert len(row[RLES]) == 3, "Without a declared background, id 0 is a layer"

    run = tlc.init(project_name="test_semantic_prediction_row_no_bg", run_name="test_semantic_prediction_row_no_bg")
    writer = tlc.MetricsTableWriter(run_url=run.url, foreign_table_url=run.url, schema={"seg": schema})
    writer.add_batch({"example_id": [0, 1], "seg": [row, dense]})
    table = writer.finalize()
    for i in range(2):
        assert np.array_equal(table[i]["seg"].mask, dense), f"Row {i} does not read back as the dense map"


def test_semantic_rejects_instance_segmentation_table() -> None:
    # An instance segmentation table cannot be trained on as semantic segmentation. The error says why, and how to get
    # semantic segmentation tables from the same data.
    from tlc_ultralytics.utils.dataset import create_tables_from_yaml_file

    table = create_tables_from_yaml_file(
        TASK2DATASET["segment"], task="segment", project_name="test_semantic_rejects_instance", splits=("val",)
    )["val"]
    for label_column_name in (None, "segmentations"):
        with pytest.raises(ValueError, match="holds instance segmentation") as error:
            check_tlc_dataset("", {"val": table}, "image", label_column_name, task="semantic", splits=("val",))
        message = str(error.value)
        assert "task='segment'" in message
        assert "create_tables_from_yaml_file(..., task='semantic')" in message
        assert "tlc.Table.from_semantic_segmentation" in message
