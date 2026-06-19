from __future__ import annotations

import json
import logging
import os
import pathlib
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
from tlc._core.objects.tables.from_table.pacmap_table import PacmapTable
from tlc._core.objects.tables.null_overlay import NullOverlay
from tlc.constants._run_status import RUN_STATUS_COMPLETED
from tlc.helpers import KeypointHelper
from ultralytics.cfg import ASSETS
from ultralytics.models.yolo import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.models.yolo.pose import PoseTrainer
from ultralytics.models.yolo.segment import SegmentationTrainer

from tlc_ultralytics import YOLO as TLCYOLO
from tlc_ultralytics import Settings
from tlc_ultralytics.classify.trainer import TLCClassificationTrainer
from tlc_ultralytics.constants import (
    DEFAULT_COLLECT_RUN_DESCRIPTION,
    EPOCH,
    FOREIGN_TABLE_ID,
    LABEL,
    MAP,
    MAP50_95,
    NUM_IMAGES,
    NUM_INSTANCES,
    PER_CLASS_METRICS_STREAM_NAME,
    PRECISION,
    RECALL,
    TRAINING_PHASE,
)
from tlc_ultralytics.detect.dataset import TLCYOLODataset
from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.engine.dataset import TLCDatasetMixin
from tlc_ultralytics.engine.utils import _complete_label_column_name
from tlc_ultralytics.obb.trainer import TLCOBBTrainer
from tlc_ultralytics.pose.trainer import TLCPoseTrainer
from tlc_ultralytics.segment.trainer import TLCSegmentationTrainer
from tlc_ultralytics.segment.utils import check_seg_table
from tlc_ultralytics.utils import check_tlc_dataset

# PaCMAP embedding reduction is known not to work on macOS: the reducer collects
# zero embeddings and silently produces no reduced table. Embedding-specific
# checks are therefore skipped on macOS (everything else still runs there).
PACMAP_BROKEN_ON_MACOS = sys.platform == "darwin"
skip_pacmap_on_macos = pytest.mark.skipif(
    PACMAP_BROKEN_ON_MACOS,
    reason="PaCMAP embedding reduction does not work on macOS",
)

DUMMY_IMAGE_FILE = Path(__file__).parent.parent / "src" / "tlc_ultralytics" / "_static" / "dashboard.png"
TMP = Path(__file__).parent / "tmp"
TMP_PROJECT_ROOT_URL = tlc.Url(TMP / "3LC")
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
    # Note: loss collection is not supported for YOLO26 (end2end) models, so we don't check for loss columns here.
    # Per-sample loss collection for YOLO11 is covered by test_detect_training_with_yolo11_per_sample_loss.
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


def test_detect_yolo26_disables_per_sample_loss() -> None:
    """Test that YOLO26 models correctly disable per-sample loss collection with a warning."""
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
        project_name="test_yolo26_no_loss",
        run_name="test_yolo26_no_loss",
        collect_loss=True,  # Request loss collection, but should be disabled for YOLO26
        collection_epoch_start=1,
    )

    model_3lc = TLCYOLO(model)
    with capture_logs() as log_messages:
        results = model_3lc.train(**overrides, settings=settings)

    assert results, "YOLO26 detection training failed"

    # Check that a warning was logged about disabling loss collection
    loss_warning_found = any("Per-sample loss collection is not supported for YOLO26" in msg for msg in log_messages)
    assert loss_warning_found, "Expected warning about YOLO26 loss collection not being supported"

    run = _get_run_from_settings(settings)
    metrics_tables = get_metrics_tables_from_run(run)

    # Check that loss columns are NOT present (loss was disabled)
    metrics_df = pd.concat(
        [m.to_pandas() for m in metrics_tables["default_stream"]],
        ignore_index=True,
    )

    assert "loss" not in metrics_df.columns, "Expected 'loss' column to NOT be present for YOLO26"
    assert "box_loss" not in metrics_df.columns, "Expected 'box_loss' column to NOT be present for YOLO26"
    assert "cls_loss" not in metrics_df.columns, "Expected 'cls_loss' column to NOT be present for YOLO26"
    assert "dfl_loss" not in metrics_df.columns, "Expected 'dfl_loss' column to NOT be present for YOLO26"


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
    overrides = {"device": "cpu"}
    settings = Settings(project_name=f"test_{task}_collect", run_name=f"test_{task}_collect", collect_loss=True)
    splits = ("train", "val")

    model = TLCYOLO(TASK2MODEL[task])
    results_dict = model.collect(data=TASK2DATASET[task], splits=splits, settings=settings, **overrides)
    assert all(results_dict[split] for split in splits), "Metrics collection failed"

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


def test_collect_with_save_json_disabled() -> None:
    """Check that save_json=True gets disabled with a warning and collection runs to completion."""
    task = "detect"
    overrides = {"device": "cpu", "save_json": True, "imgsz": 320}
    settings = Settings(
        project_name="test_detect_save_json",
        run_name="test_detect_save_json",
    )

    model = TLCYOLO(TASK2MODEL[task])
    with capture_logs(logging.WARNING) as log_messages:
        results_dict = model.collect(data=TASK2DATASET[task], splits=("train",), settings=settings, **overrides)

    # (a) No KeyError: 'path' — collection completed for the split.
    assert results_dict["train"], "Metrics collection failed with save_json=True"

    # (b) A clear warning about save_json being unsupported was emitted.
    save_json_warning_found = any("save_json is not supported with 3LC datasets" in msg for msg in log_messages)
    assert save_json_warning_found, "Expected warning about save_json not being supported with 3LC datasets"


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
    assert any(isinstance(metrics_table, PacmapTable) for metrics_table in run.metrics_tables), "Expected a PaCMAPTable"

    embeddings_table = next(
        metrics_table for metrics_table in run.metrics_tables if isinstance(metrics_table, PacmapTable)
    )
    assert "embeddings_pacmap" in embeddings_table.columns, "Expected embeddings column"

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


def test_train_collection_disabled() -> None:
    task = "classify"
    model_arg = TASK2MODEL[task]
    overrides = {"data": TASK2DATASET[task], "device": "cpu", "epochs": 1, "batch": 4, "imgsz": 224}

    model = TLCYOLO(model_arg)

    settings = Settings(
        collection_disable=True,
        project_name="test_train_collection_disabled",
        run_name="test_train_collection_disabled",
    )
    model.train(**overrides, settings=settings)

    # Ensure that only validation metrics are collected after training
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


def _build_task_dataset(task: str, config: dict, rows: list[dict], table_name: str):
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
    assert cache_data["version"] == 1, "Cache version should be 1"
    assert cache_data["corrupt_example_ids"] == []


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

    Exercises ``_reduce_instance_embeddings`` and ``_transform_instance_embeddings``
    directly on synthetic data so the test doesn't depend on a full model run or
    the size of the YOLO test dataset. This is the scenario that catches pacmap's
    ``save_tree=True`` requirement — without it the fitted reducer can't project
    GT embeddings into the predicted space.
    """
    pytest.importorskip(reducer if reducer != "pca" else "sklearn")

    from tlc_ultralytics.utils._instance_reduce import (
        _reduce_instance_embeddings,
        _transform_instance_embeddings,
    )

    rng = np.random.default_rng(0)
    raw_per_image = [rng.normal(size=(20, 32)).astype(np.float32) for _ in range(10)]

    try:
        # random_state is a raw constructor kwarg for all three reducers; passing it through
        # exercises that instance_embeddings_reducer_kwargs are forwarded to the constructor.
        reduced, fitted = _reduce_instance_embeddings(
            raw_per_image,
            method=reducer,
            n_components=2,
            random_state=42,
        )
    except ValueError as exc:
        # pacmap on macOS ARM currently fails during fit_transform with a
        # broadcast/shape error from its internal KNN. Skip rather than fail —
        # the post-fit .transform() path (the save_tree=True regression guard)
        # can only be checked when fit itself works.
        pytest.skip(f"{reducer} fit failed in this environment: {exc}")

    assert fitted is not None
    assert all(r.shape == (20, 2) for r in reduced)
    # The forwarded kwarg reached the underlying reducer constructor.
    assert fitted.random_state == 42

    # Transform a disjoint batch with the fitted reducer — this crashes on
    # pacmap when save_tree=False, which is the bug the in-process reducer guards.
    new_raw = [rng.normal(size=(5, 32)).astype(np.float32) for _ in range(3)]
    projected = _transform_instance_embeddings(new_raw, fitted, n_components=2)
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
    """pca is only supported by the in-process instance reduction, not the native image reduction."""
    settings = Settings(image_embeddings_dim=2, image_embeddings_reducer="pca", label_column_name="test")
    with pytest.raises(ValueError, match="image_embeddings_reducer"):
        settings.verify(training=False)

    settings = Settings(instance_embeddings_dim=2, instance_embeddings_reducer="pca", label_column_name="test")
    settings.verify(training=False)

    settings = Settings(instance_embeddings_dim=2, instance_embeddings_reducer="illegal", label_column_name="test")
    with pytest.raises(ValueError, match="instance_embeddings_reducer"):
        settings.verify(training=False)


def test_split_reduced_by_rank() -> None:
    """Unit test for the DDP re-split of flattened reduced embeddings back to per-rank lists."""
    from tlc_ultralytics.engine.validator import TLCValidatorMixin

    rng = np.random.default_rng(0)

    def make_payload(n_images, n_instances):
        return [rng.normal(size=(n_instances, 16)).astype(np.float32) for _ in range(n_images)]

    # Rank 0: 3 images, rank 1: 2 images (pred); GT counts differ from pred counts
    gathered = [
        (make_payload(3, 4), make_payload(3, 2)),
        (make_payload(2, 4), make_payload(2, 2)),
    ]
    pred_reduced_all = [rng.normal(size=(4, 2)).astype(np.float32) for _ in range(5)]
    gt_reduced_all = [rng.normal(size=(2, 2)).astype(np.float32) for _ in range(5)]

    per_rank = TLCValidatorMixin._split_reduced_by_rank(gathered, pred_reduced_all, gt_reduced_all)

    assert len(per_rank) == 2
    pred_r0, gt_r0 = per_rank[0]
    pred_r1, gt_r1 = per_rank[1]
    assert len(pred_r0) == 3 and len(gt_r0) == 3
    assert len(pred_r1) == 2 and len(gt_r1) == 2
    # Order is preserved: rank 1's first image is the 4th flattened entry
    np.testing.assert_array_equal(pred_r1[0], pred_reduced_all[3])
    np.testing.assert_array_equal(gt_r1[1], gt_reduced_all[4])

    # Without GT, gt side is None for every rank
    per_rank_no_gt = TLCValidatorMixin._split_reduced_by_rank(gathered, pred_reduced_all, None)
    assert all(gt is None for _, gt in per_rank_no_gt)


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

    # The run's reducer must not leak past collect()
    from tlc_ultralytics.utils._instance_reduce import _get_fitted_reducer

    assert _get_fitted_reducer(run.url.to_str()) is None


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
