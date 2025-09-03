"""Utilities for testing dataset determinism.

This module contains functions for testing that datasets are deterministic
with the same seed across separate processes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _compare_label_equality(label_ultralytics: dict[str, Any], label_3lc: dict[str, Any], task: str, mode: str) -> None:
    keys = {
        "im_file",
        "cls",
        "bboxes",
        "keypoints",
        "segments",
        "normalized",
        "bbox_format",
    }
    if mode == "train":
        keys.add("shape")

    assert set(label_ultralytics.keys()) == keys

    keys = {
        "im_file",
        "cls",
        "bboxes",
        "keypoints",
        "segments",
        "normalized",
        "bbox_format",
        "example_id",
    }
    if mode == "train":
        keys.add("shape")
    assert set(label_3lc.keys()) == keys


def _compare_sample_equality(
    sample_ultralytics: dict[str, Any], sample_3lc: dict[str, Any], task: str, mode: str
) -> None:
    tlc_keys = {"im_file", "ori_shape", "resized_shape", "img", "cls", "bboxes", "batch_idx", "ratio_pad", "example_id"}
    # if mode == "train":
    #     keys.add("example_id")
    assert set(sample_3lc.keys()) == tlc_keys

    ul_keys = {"im_file", "ori_shape", "resized_shape", "img", "cls", "bboxes", "batch_idx", "ratio_pad"}
    # if mode == "train":
    #     keys.add("example_id")
    assert set(sample_ultralytics.keys()) == ul_keys


def _compare_dataset_rows(row_ultralytics: dict[str, Any], row_3lc: dict[str, Any]) -> None:
    """Compare dataset rows from ultralytics and 3lc.

    Args:
        row_ultralytics: Row from ultralytics dataset
        row_3lc: Row from 3lc dataset
    """
    for key, value_ultralytics in row_ultralytics.items():
        if key == "random_tracking_info":
            continue

        assert key in row_3lc, f"Key {key} not found in 3LC row"
        value_3lc = row_3lc[key]
        if key == "im_file":
            assert Path(value_ultralytics) == Path(value_3lc), "Image path not equal in 3LC and Ultralytics"
            continue

        if isinstance(value_ultralytics, (np.ndarray, torch.Tensor)):
            (
                torch.testing.assert_close(value_ultralytics, value_3lc, atol=0.013, rtol=0.125),
                f"Value {key} not equal in 3LC and Ultralytics",
            )
        elif isinstance(value_ultralytics, list):
            for item_ultralytics, item_3lc in zip(value_ultralytics, value_3lc):
                torch.testing.assert_close(np.array(item_ultralytics), np.array(item_3lc), atol=0.02, rtol=0.43)
        else:
            assert value_ultralytics == value_3lc, f"Value {key} not equal in 3LC and Ultralytics"


def create_dataset_samples(mode: str, task: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create dataset samples for both 3lc and ultralytics.

    Args:
        mode: Dataset mode ('train' or 'val')
        task: Task to use ('detect' or 'pose')
    Returns:
        Tuple of (3lc rows, ultralytics rows)
    """
    from test_tlc_ultralytics import TASK2DATASET, TASK2MODEL
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.models.yolo.pose import PoseTrainer

    from tlc_ultralytics import Settings
    from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
    from tlc_ultralytics.pose.trainer import TLCPoseTrainer

    settings = Settings(project_name=f"test_dataset_determinism_mode_{mode}")
    overrides = {
        "data": TASK2DATASET[task],
        "model": TASK2MODEL[task],
        "seed": 42,
        "deterministic": True,
    }

    overrides_3lc = overrides.copy()
    overrides_3lc["settings"] = settings

    trainer_ultralytics = (
        DetectionTrainer(overrides=overrides) if task == "detect" else PoseTrainer(overrides=overrides)
    )
    trainer_ultralytics.model = None
    dataset_ultralytics = trainer_ultralytics.build_dataset(trainer_ultralytics.data["train"], mode=mode, batch=4)
    rows_ultralytics = list(dataset_ultralytics)

    trainer_3lc = (
        TLCDetectionTrainer(overrides=overrides_3lc) if task == "detect" else TLCPoseTrainer(overrides=overrides_3lc)
    )
    trainer_3lc.model = None
    dataset_3lc = trainer_3lc.build_dataset(trainer_3lc.data["train"], mode=mode, batch=4)
    rows_3lc = list(dataset_3lc)

    return rows_3lc, rows_ultralytics


def create_dataset_samples_with_tracking(mode: str, task: str, output_file: str | None = None) -> None:
    """Create dataset samples with tracking and write JSON result to a file or stdout.

    Args:
        mode: Dataset mode ('train' or 'val')
        output_file: Path to the output file. If None, prints to stdout.
    """
    from random_tracker import disable_tracking, enable_tracking, get_tracking_info, reset_tracking
    from test_tlc_ultralytics import TASK2DATASET, TASK2MODEL

    # we want to start the tracking here
    # reset_tracking()
    # enable_tracking()

    try:
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.models.yolo.pose import PoseTrainer

        from tlc_ultralytics import Settings
        from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
        from tlc_ultralytics.pose.trainer import TLCPoseTrainer

        # but we start it here.
        reset_tracking()
        enable_tracking()

        settings = Settings(project_name=f"test_dataset_determinism_mode_{mode}")
        overrides = {
            "data": TASK2DATASET[task],
            "model": TASK2MODEL[task],
            "seed": 42,
            "deterministic": True,
        }

        overrides_3lc = overrides.copy()
        overrides_3lc["settings"] = settings

        trainer_ultralytics = (
            DetectionTrainer(overrides=overrides) if task == "detect" else PoseTrainer(overrides=overrides)
        )
        trainer_ultralytics.model = None
        dataset_ultralytics = trainer_ultralytics.build_dataset(trainer_ultralytics.data["train"], mode=mode, batch=4)
        rows_ultralytics = list(dataset_ultralytics)

        random_info_ultralytics = get_tracking_info()

        reset_tracking()

        trainer_3lc = (
            TLCDetectionTrainer(overrides=overrides_3lc)
            if task == "detect"
            else TLCPoseTrainer(overrides=overrides_3lc)
        )
        trainer_3lc.model = None
        dataset_3lc = trainer_3lc.build_dataset(trainer_3lc.data["train"], mode=mode, batch=4)
        rows_3lc = list(dataset_3lc)

        random_info_3lc = get_tracking_info()

        # Assert row equality here in the sub-process
        for row_ultralytics, row_3lc in zip(rows_ultralytics, rows_3lc):
            _compare_dataset_rows(row_ultralytics, row_3lc)

        result = {
            "random_info_3lc": random_info_3lc,
            "random_info_ultralytics": random_info_ultralytics,
            "rows_count_3lc": len(rows_3lc),
            "rows_count_ultralytics": len(rows_ultralytics),
        }

        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_file.as_posix()
        if output_file:
            with open(output_file, "w") as f:
                json.dump(result, f)
        else:
            print(json.dumps(result))
    except Exception as e:
        error_result = {"error": str(e)}
        if output_file:
            with open(output_file, "w") as f:
                json.dump(error_result, f)
        else:
            print(json.dumps(error_result))
    finally:
        disable_tracking()
