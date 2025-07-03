"""Utilities for testing dataset determinism.

This module contains functions for testing that datasets are deterministic
with the same seed across separate processes.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch


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

        if isinstance(value_ultralytics, (np.ndarray, torch.Tensor)):
            assert (value_ultralytics == value_3lc).all(), f"Value {key} not equal in 3LC and Ultralytics"
        else:
            assert value_ultralytics == value_3lc, f"Value {key} not equal in 3LC and Ultralytics"


def create_dataset_samples(mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create dataset samples for both 3lc and ultralytics.

    Args:
        mode: Dataset mode ('train' or 'val')

    Returns:
        Tuple of (3lc rows, ultralytics rows)
    """
    from test_tlc_ultralytics import TASK2DATASET, TASK2MODEL
    from ultralytics.models.yolo.detect import DetectionTrainer

    from tlc_ultralytics import Settings
    from tlc_ultralytics.detect.trainer import TLCDetectionTrainer

    settings = Settings(project_name=f"test_dataset_determinism_mode_{mode}")
    overrides = {
        "data": TASK2DATASET["detect"],
        "model": TASK2MODEL["detect"],
        "seed": 42,
        "deterministic": True,
    }

    overrides_3lc = overrides.copy()
    overrides_3lc["settings"] = settings

    trainer_ultralytics = DetectionTrainer(overrides=overrides)
    trainer_ultralytics.model = None
    dataset_ultralytics = trainer_ultralytics.build_dataset(trainer_ultralytics.data["train"], mode=mode, batch=4)
    rows_ultralytics = list(dataset_ultralytics)

    trainer_3lc = TLCDetectionTrainer(overrides=overrides_3lc)
    trainer_3lc.model = None
    dataset_3lc = trainer_3lc.build_dataset(trainer_3lc.data["train"], mode=mode, batch=4)
    rows_3lc = list(dataset_3lc)

    return rows_3lc, rows_ultralytics


def create_dataset_samples_with_tracking(mode: str, output_file: str | None = None) -> None:
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

        from tlc_ultralytics import Settings
        from tlc_ultralytics.detect.trainer import TLCDetectionTrainer

        # but we start it here.
        reset_tracking()
        enable_tracking()

        settings = Settings(project_name=f"test_dataset_determinism_mode_{mode}")
        overrides = {
            "data": TASK2DATASET["detect"],
            "model": TASK2MODEL["detect"],
            "seed": 42,
            "deterministic": True,
        }

        overrides_3lc = overrides.copy()
        overrides_3lc["settings"] = settings

        trainer_ultralytics = DetectionTrainer(overrides=overrides)
        trainer_ultralytics.model = None
        dataset_ultralytics = trainer_ultralytics.build_dataset(trainer_ultralytics.data["train"], mode=mode, batch=4)
        rows_ultralytics = list(dataset_ultralytics)

        random_info_ultralytics = get_tracking_info()

        reset_tracking()

        trainer_3lc = TLCDetectionTrainer(overrides=overrides_3lc)
        trainer_3lc.model = None
        dataset_3lc = trainer_3lc.build_dataset(trainer_3lc.data["train"], mode=mode, batch=4)
        rows_3lc = list(dataset_3lc)

        random_info_3lc = get_tracking_info()

        result = {
            "random_info_3lc": random_info_3lc,
            "random_info_ultralytics": random_info_ultralytics,
            "rows_count_3lc": len(rows_3lc),
            "rows_count_ultralytics": len(rows_ultralytics),
        }

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
