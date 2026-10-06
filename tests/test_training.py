from __future__ import annotations

import json
import os
from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd
import pytest
import tlc
from task_config import (
    COCO_POSE_SETTINGS_OVERRIDES,
    OKS_SIGMAS,
    PACMAP_BROKEN_ON_MACOS,
    TASK2DATASET,
    TASK2LABEL_COLUMN_NAME,
    TASK2MODEL,
    TASK2PREDICTED_LABEL_COLUMN_NAME,
    TASK2TRAINER,
)
from testing_helpers import (
    capture_logs,
    check_pose_table_and_metrics_tables,
    get_metrics_tables_from_run,
    get_run_from_settings,
    override_oks_sigmas,
)
from tlc._core.objects.tables.from_table.edited_table import EditedTable
from tlc.constants._run_status import RUN_STATUS_COMPLETED
from tmp_paths import TMP
from ultralytics.cfg import ASSETS
from ultralytics.models.yolo import YOLO

from tlc_ultralytics import YOLO as TLCYOLO
from tlc_ultralytics import Settings
from tlc_ultralytics.constants import (
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
from tlc_ultralytics.utils import check_tlc_dataset

if TYPE_CHECKING:
    import pathlib


pytestmark = pytest.mark.slow


def _assert_per_sample_loss_collected(task: str, metrics_df: pd.DataFrame, log_messages: list[str]) -> None:
    """Check per-sample loss collection for the YOLO26 (DFL-free) detection model."""
    if task != "detect":
        return

    assert not any("Per-sample loss collection is not supported" in msg for msg in log_messages), (
        "Unexpected warning about loss collection not being supported"
    )
    assert "loss" in metrics_df.columns, "Expected 'loss' column to be present"
    assert "box_loss" in metrics_df.columns, "Expected 'box_loss' column to be present"
    assert "cls_loss" in metrics_df.columns, "Expected 'cls_loss' column to be present"
    assert "dfl_loss" not in metrics_df.columns, "Expected no 'dfl_loss' column for DFL-free YOLO26"
    assert not metrics_df["loss"].isna().all(), "All loss values are NaN"
    assert metrics_df["loss"].sum() > 0, "Total loss should be positive"


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
    run = get_run_from_settings(settings)

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
    _assert_per_sample_loss_collected(task, metrics_df, tlc_messages)
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

    run = get_run_from_settings(settings)
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

    run = get_run_from_settings(settings)
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

    run = get_run_from_settings(settings)
    metrics_tables = get_metrics_tables_from_run(run)
    metrics_df = pd.concat(
        [m.to_pandas() for m in metrics_tables["default_stream"]],
        ignore_index=True,
    )
    assert "loss" not in metrics_df.columns, "Expected no 'loss' column for YOLO26 pose"


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

    run = get_run_from_settings(settings)

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

    # Instance embeddings are not supported for classification: a warning is logged and no embedding columns written
    instance_settings = Settings(
        project_name="test_classify_no_instance_embeddings",
        run_name="test_classify_no_instance_embeddings",
        instance_embeddings_dim=2,
        instance_embeddings_reducer="pca",
    )
    with capture_logs() as log_messages:
        TLCYOLO(best).val(data=data, device="cpu", imgsz=32, batch=4, workers=0, settings=instance_settings)

    assert any("Instance embeddings are not supported for the 'classify' task" in msg for msg in log_messages), (
        "Expected warning about instance embeddings not being supported for classify"
    )
    instance_run = get_run_from_settings(instance_settings)
    instance_df = pd.concat(
        [m.to_pandas() for m in get_metrics_tables_from_run(instance_run)["default_stream"]], ignore_index=True
    )
    assert not any("embedding" in col for col in instance_df.columns), "Expected no instance-embedding columns"

    # model.predict() should work and be the same as vanilla ultralytics
    preds_3lc = model_3lc.predict(imgsz=320)
    preds_ultralytics = model_ultralytics.predict(imgsz=320)

    assert preds_3lc[0].probs.top5 == preds_ultralytics[0].probs.top5, "Predictions mismatch"


def test_train_split_loader_unpinned_and_closed_between_passes(monkeypatch) -> None:
    # The train-split collection loader is a third loader Ultralytics knows nothing about, so it never closes it. It
    # must be built without pinned memory (which PyTorch caches for the rest of the process) and have its workers shut
    # down after every collection pass, instead of holding them and their prefetched batches for the rest of training.
    import torch

    import tlc_ultralytics.overrides

    loaders = []
    build_dataloader = tlc_ultralytics.overrides.build_dataloader_ultralytics

    def recording_build_dataloader(*args, **kwargs):
        loader = build_dataloader(*args, **kwargs)
        loaders.append((loader, kwargs.get("pin_memory", True)))
        return loader

    monkeypatch.setattr(tlc_ultralytics.overrides, "build_dataloader_ultralytics", recording_build_dataloader)

    workers_alive_at_epoch_start = []

    def on_train_epoch_start(trainer):
        if trainer._train_validator is not None:
            workers_alive_at_epoch_start.append(trainer._train_validator.dataloader.iterator is not None)

    model = TLCYOLO(TASK2MODEL["detect"])
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    settings = Settings(
        collection_epoch_start=1,
        project_name="test_train_split_loader_project",
        run_name="test_train_split_loader",
    )
    # batch=1 gives the train-split loader (batch 2) two batches of coco8's four images, so on CUDA it gets workers.
    # On CPU Ultralytics trains without workers, but a live loader still keeps its iterator until it is closed.
    device = "0" if torch.cuda.is_available() else "cpu"
    model.train(
        data=TASK2DATASET["detect"], epochs=2, batch=1, workers=2, device=device, plots=False, settings=settings
    )

    train_validator_loader = model.trainer.train_validator.dataloader
    assert [pinned for loader, pinned in loaders if loader is train_validator_loader] == [False]
    assert all(pinned for loader, pinned in loaders if loader is not train_validator_loader)
    assert workers_alive_at_epoch_start == [False]  # closed after epoch 1's collection, before epoch 2 trains
    assert train_validator_loader.iterator is None  # closed after the final collection pass


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
    run = get_run_from_settings(settings)
    assert len(run.metrics_tables) == 1, "Expected only validation metrics to be collected after training"


@pytest.mark.parametrize("task", ["classify", "detect", "segment", "obb"])
def test_train_collection_disabled(task: str) -> None:
    model_arg = TASK2MODEL[task]
    overrides = {"data": TASK2DATASET[task], "device": "cpu", "epochs": 1, "batch": 4, "imgsz": 32}

    model = TLCYOLO(model_arg)

    settings = Settings(
        collection_disable=True,
        project_name=f"test_train_collection_disabled_{task}",
        run_name=f"test_train_collection_disabled_{task}",
    )
    model.train(**overrides, settings=settings)

    # classify never writes per-class tables, so only detect/segment/obb can catch a gating regression here.
    run = get_run_from_settings(settings)
    assert len(run.metrics_tables) == 0, "Expected no metrics tables to be written"


@pytest.mark.parametrize("task", ["detect", "classify"])
def test_train_no_weight_column_in_table(task) -> None:
    # Test that training with a table that has no weight column works
    settings = Settings(project_name=f"test_train_no_weight_column_in_table_{task}")
    trainer = TASK2TRAINER[task](
        overrides={"data": TASK2DATASET[task], "model": TASK2MODEL[task], "settings": settings},
    )
    table = trainer.data["train"]

    model = TLCYOLO(TASK2MODEL[task])

    no_weight_column_table = table.delete_column(table.weights_column_name)
    tables = {"train": no_weight_column_table, "val": trainer.data.get("val") or trainer.data["test"]}

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


@pytest.mark.parametrize("task", ["detect", "classify"])
def test_train_without_val_split_validates_on_test(task) -> None:
    # Without a val split, the test split is what training validates on
    settings = Settings(project_name=f"test_train_without_val_split_validates_on_test_{task}")
    trainer = TASK2TRAINER[task](
        overrides={"data": TASK2DATASET[task], "model": TASK2MODEL[task], "settings": settings},
    )
    tables = {"train": trainer.data["train"], "test": trainer.data["val"]}

    model = TLCYOLO(TASK2MODEL[task])
    model.train(tables=tables, settings=settings, epochs=1, device="cpu", workers=0)

    assert "val" not in model.trainer.data
    assert model.trainer.test_loader.dataset.table.url == tables["test"].url


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

    run = get_run_from_settings(settings)

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

    run = get_run_from_settings(settings)

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


@pytest.mark.skip(
    reason="Settings.points/lines/point_attributes/line_attributes are not forwarded when tables are created from a "
    "YAML file, so the table metadata checks fail"
)
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
    run = get_run_from_settings(settings)
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
