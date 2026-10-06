"""Utilities for testing dataset determinism.

This module contains functions for testing that datasets are deterministic
with the same seed across separate processes.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from task_config import TASK2TRAINER, TASK2ULTRALYTICS_TRAINER
from testing_helpers import stub_model_with_stride


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
            assert np.allclose(value_3lc, value_ultralytics), f"Value {key} not equal in 3LC and Ultralytics"
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
    from task_config import TASK2DATASET, TASK2MODEL

    from tlc_ultralytics import Settings

    settings = Settings(project_name=f"test_dataset_determinism_mode_{mode}_{task}")
    overrides = {
        "data": TASK2DATASET[task],
        "model": TASK2MODEL[task],
        "seed": 42,
        "deterministic": True,
    }

    overrides_3lc = overrides.copy()
    overrides_3lc["settings"] = settings

    trainer_ultralytics = TASK2ULTRALYTICS_TRAINER[task](overrides=overrides)
    trainer_ultralytics.model = stub_model_with_stride()
    dataset_ultralytics = trainer_ultralytics.build_dataset(trainer_ultralytics.data["train"], mode=mode, batch=4)
    rows_ultralytics = list(dataset_ultralytics)

    trainer_3lc = TASK2TRAINER[task](overrides=overrides_3lc)
    trainer_3lc.model = stub_model_with_stride()
    dataset_3lc = trainer_3lc.build_dataset(trainer_3lc.data["train"], mode=mode, batch=4)
    rows_3lc = list(dataset_3lc)

    return rows_3lc, rows_ultralytics


def _create_dataset_samples_with_tracking(mode: str, task: str) -> dict[str, Any]:
    """Build both datasets for a mode and task with random-call tracking and compare their rows."""
    from random_tracker import disable_tracking, enable_tracking, get_tracking_info, reset_tracking
    from task_config import TASK2DATASET, TASK2MODEL

    from tlc_ultralytics import Settings

    try:
        # Only count calls made while building the datasets
        reset_tracking()
        enable_tracking()

        # Distinct from the project used by `create_dataset_samples`, which runs in the pytest process: this one runs
        # in a subprocess, so sharing a project name would have two processes create the same tables.
        settings = Settings(project_name=f"test_dataset_determinism_tracking_mode_{mode}_{task}")
        overrides = {
            "data": TASK2DATASET[task],
            "model": TASK2MODEL[task],
            "seed": 42,
            "deterministic": True,
        }

        overrides_3lc = overrides.copy()
        overrides_3lc["settings"] = settings

        trainer_ultralytics = TASK2ULTRALYTICS_TRAINER[task](overrides=overrides)
        trainer_ultralytics.model = stub_model_with_stride()
        dataset_ultralytics = trainer_ultralytics.build_dataset(trainer_ultralytics.data["train"], mode=mode, batch=4)
        rows_ultralytics = list(dataset_ultralytics)

        random_info_ultralytics = get_tracking_info()

        reset_tracking()

        trainer_3lc = TASK2TRAINER[task](overrides=overrides_3lc)
        trainer_3lc.model = stub_model_with_stride()
        dataset_3lc = trainer_3lc.build_dataset(trainer_3lc.data["train"], mode=mode, batch=4)
        rows_3lc = list(dataset_3lc)

        random_info_3lc = get_tracking_info()

        # Assert row equality here, next to the tracking
        for row_ultralytics, row_3lc in zip(rows_ultralytics, rows_3lc, strict=False):
            _compare_dataset_rows(row_ultralytics, row_3lc)

        return {
            "random_info_3lc": random_info_3lc,
            "random_info_ultralytics": random_info_ultralytics,
            "rows_count_3lc": len(rows_3lc),
            "rows_count_ultralytics": len(rows_ultralytics),
        }
    finally:
        disable_tracking()


def create_dataset_samples_with_tracking(combinations: list[tuple[str, str]], output_file: str) -> None:
    """Run each (mode, task) combination and write the results, keyed by `f"{mode}-{task}"`, to `output_file`.

    A failing combination is recorded with its full traceback under `error`, since the caller only sees the file.
    """
    results: dict[str, dict[str, Any]] = {}
    for mode, task in combinations:
        try:
            results[f"{mode}-{task}"] = _create_dataset_samples_with_tracking(mode, task)
        except Exception:
            results[f"{mode}-{task}"] = {"error": traceback.format_exc()}

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results))
