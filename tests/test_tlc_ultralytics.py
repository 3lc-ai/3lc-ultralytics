import os
import pathlib
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pandas as pd
import pytest
import tlc
from PIL import Image
from ultralytics.models.yolo import YOLO

from tlc_ultralytics import YOLO as TLCYOLO
from tlc_ultralytics import Settings
from tlc_ultralytics.classify.trainer import TLCClassificationTrainer
from tlc_ultralytics.classify.utils import tlc_check_cls_dataset
from tlc_ultralytics.constants import (
    DEFAULT_COLLECT_RUN_DESCRIPTION,
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
from tlc_ultralytics.detect.utils import tlc_check_det_dataset
from tlc_ultralytics.engine.dataset import TLCDatasetMixin
from tlc_ultralytics.engine.utils import _complete_label_column_name
from tlc_ultralytics.segment.trainer import TLCSegmentationTrainer
from tlc_ultralytics.segment.utils import check_seg_table, tlc_check_seg_dataset
from tlc_ultralytics.utils import check_tlc_dataset

DUMMY_IMAGE_FILE = Path(__file__).parent.parent / "src" / "tlc_ultralytics" / "_static" / "dashboard.png"
TMP = Path(__file__).parent / "tmp"
TMP_PROJECT_ROOT_URL = tlc.Url(TMP / "3LC")
tlc.Configuration.instance().project_root_url = TMP_PROJECT_ROOT_URL
tlc.TableIndexingTable.instance().add_scan_url(
    {
        "url": tlc.Url(TMP_PROJECT_ROOT_URL),
        "layout": "project",
        "object_type": "table",
        "static": True,
    }
)

TASK2DATASET = {"detect": "coco8.yaml", "classify": "imagenet10", "segment": "coco8-seg.yaml"}
TASK2MODEL = {"detect": "yolo11n.pt", "classify": "yolo11n-cls.pt", "segment": "yolo11n-seg.pt"}
TASK2LABEL_COLUMN_NAME = {
    "detect": "bbs.bb_list.label",
    "classify": "label",
    "segment": "segmentations.instance_properties.label",
}
TASK2PREDICTED_LABEL_COLUMN_NAME = {
    "detect": "bbs_predicted.bb_list.label",
    "classify": "predicted",
    "segment": "segmentations_predicted.instance_properties.label",
}

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


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_training(task) -> None:
    # End-to-end test of training for detection and segmentation
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

    settings = Settings(
        collection_epoch_start=1,
        collect_loss=True,
        image_embeddings_dim=2,
        image_embeddings_reducer="pacmap",
        project_name=f"test_{task}_project",
        run_name=f"test_{task}",
        run_description=f"Test {task} training",
    )

    model_ultralytics = YOLO(TASK2MODEL[task])
    results_ultralytics = model_ultralytics.train(**overrides)

    model_3lc = TLCYOLO(TASK2MODEL[task])
    results_3lc = model_3lc.train(**overrides, settings=settings)

    assert results_3lc, "Detection training failed"

    # Compare 3LC integration with ultralytics results
    # Segmentation results will be slightly different due to the 3lc mask storage format and conversion
    # back to polygons
    atol = 0.1 if task == "segment" else 0
    for k in results_ultralytics.results_dict.keys():
        assert np.isclose(results_ultralytics.results_dict[k], results_3lc.results_dict[k], atol=atol), (
            f"Results validation metrics 3LC different from Ultralytics for {k}"
        )

    assert results_ultralytics.names == results_3lc.names, "Results validation names"

    # Get 3LC run and inspect the results
    run = _get_run_from_settings(settings)

    assert run.status == tlc.RUN_STATUS_COMPLETED, "Run status not set to completed after training"

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
    metrics_tables = get_metrics_tables_from_run(run)

    # Check that the desired metrics were written
    metrics_df = pd.concat(
        [m.to_pandas() for m in metrics_tables["default_stream"]],
        ignore_index=True,
    )

    embeddings_column_name = f"embeddings_{settings.image_embeddings_reducer}"
    assert embeddings_column_name in metrics_df.columns, "Expected embeddings column missing"
    assert len(metrics_df[embeddings_column_name][0]) == settings.image_embeddings_dim, "Embeddings dimension mismatch"

    if task == "detect":
        assert "loss" in metrics_df.columns, "Expected loss column to be present, but it is missing"
    assert 0 in metrics_df[TRAINING_PHASE], "Expected metrics from during training"
    assert 1 in metrics_df[TRAINING_PHASE], "Expected metrics from after training"

    # model.predict() should work and be the same as vanilla ultralytics
    assert all(model_ultralytics.predict(imgsz=320)[0].boxes.cls == model_3lc.predict(imgsz=320)[0].boxes.cls), (
        "Predictions mismatch"
    )

    per_class_metrics_tables = metrics_tables[PER_CLASS_METRICS_STREAM_NAME]
    # 6 = 2 epochs * 2 splits + 2 splits after training
    assert len(per_class_metrics_tables) == 6, "Expected 6 per-class metrics tables to be written"
    per_class_metrics_df = pd.concat(
        [m.to_pandas() for m in per_class_metrics_tables],
        ignore_index=True,
    )
    assert TRAINING_PHASE in per_class_metrics_df.columns, "Expected training phase column in per-class metrics"
    assert tlc.EPOCH in per_class_metrics_df.columns, "Expected epoch column in per-class metrics"
    assert tlc.FOREIGN_TABLE_ID in per_class_metrics_df.columns, "Expected foreign_table_id column in per-class metrics"
    assert tlc.LABEL in per_class_metrics_df.columns, "Expected label column in per-class metrics"
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
        model_3lc.train(**overrides, settings=Settings(image_embeddings_dim=2))

    # But should run to completion without embeddings collection
    model_3lc.train(**overrides)


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

    assert run.status == tlc.RUN_STATUS_COMPLETED, "Run status not set to completed after training"
    assert not run.description, "Description mismatch, default should be empty string"

    # Imagenet should get special treatment with label display name overrides
    input_table = tlc.Table.from_url(run.url / run.constants["inputs"][0]["input_table_url"])
    value_map = input_table.get_value_map("label")
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

    embeddings_column_name = f"embeddings_{settings.image_embeddings_reducer}"
    assert embeddings_column_name in metrics_df.columns, "Expected embeddings column missing"
    assert len(metrics_df[embeddings_column_name][0]) == settings.image_embeddings_dim, "Embeddings dimension mismatch"

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
    assert run.status == tlc.RUN_STATUS_COMPLETED, "Run status not set to completed after training"
    assert run.description == DEFAULT_COLLECT_RUN_DESCRIPTION, "Description mismatch"
    assert len(metrics_tables[PER_CLASS_METRICS_STREAM_NAME]) == 2, "Expected 2 per-class metrics tables (train, val)"

    per_class_metrics_df = pd.concat(
        [m.to_pandas() for m in metrics_tables[PER_CLASS_METRICS_STREAM_NAME]],
        ignore_index=True,
    )
    assert TRAINING_PHASE not in per_class_metrics_df.columns, "Expected no training phase column"
    assert tlc.EPOCH not in per_class_metrics_df.columns, "Expected no epoch column"


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
    trainer = TLCDetectionTrainer(
        overrides={"data": TASK2DATASET["detect"], "model": TASK2MODEL["detect"], "settings": settings},
    )

    # Create initial tables
    train_table = trainer.data["train"]

    # Create an edited version of the train table
    train_table_edited = tlc.NullOverlay(
        train_table.url.create_sibling("peter").create_unique(),
        input_table_url=train_table,
    )

    # A new trainer should now use the edited table since it gets latest
    new_trainer = TLCDetectionTrainer(
        overrides={"data": TASK2DATASET["detect"], "model": TASK2MODEL["detect"], "settings": settings},
    )
    assert new_trainer.data["train"].url == train_table_edited.url, "Table not resolved correctly"

    # A new trainer should not be able to take the tables directly
    tables = {"train": train_table_edited.url, "val": new_trainer.data.get("val") or new_trainer.data["test"].url}
    trainer_from_tables = TLCDetectionTrainer(
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
    trainer = TLCSegmentationTrainer(
        overrides={"data": TASK2DATASET["segment"], "model": TASK2MODEL["segment"], "settings": settings}
    )

    # A table from a yolo dataset is valid
    check_seg_table(trainer.data["train"], "image", "segmentations")

    # The same data in a new table, but backed by a row cache, is also valid
    overlay_table_url = tlc.NullOverlay(
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
    with pytest.raises(ValueError, match="Schema validation failed"):
        check_seg_table(invalid_schema_seg_table, "image", "segmentations")


def test_sampling_weights() -> None:
    # Test that sampling weights are correctly applied, with worker processes enabled
    settings = Settings(project_name="test_sampling_weights", sampling_weights=True)
    trainer = TLCDetectionTrainer(
        overrides={
            "data": TASK2DATASET["detect"],
            "model": TASK2MODEL["detect"],
            "settings": settings,
            "workers": 4,
        },
    )

    # Needs to be initialized to create dataloader (usually done in train)
    trainer.model = None

    epochs = 1000

    # Create edited table where one sample has weight increased to 2
    train_table = trainer.data["train"]
    edited_table = tlc.EditedTable(
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
    trainer = TLCDetectionTrainer(
        overrides={
            "data": TASK2DATASET["detect"],
            "model": TASK2MODEL["detect"],
            "settings": settings,
            "workers": 4,
        },
    )

    # Needs to be initialized to create dataloader (usually done in train)
    trainer.model = None

    # Create edited table where one sample has weight increased to 2
    train_table = trainer.data["train"]
    edited_table = tlc.EditedTable(
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


@pytest.mark.parametrize(
    "task,trainer_class", [("detect", TLCDetectionTrainer), ("classify", TLCClassificationTrainer)]
)
def test_exclude_zero_weight_collection(task, trainer_class) -> None:
    # Test that sampling weights are correctly applied during metrics collection
    settings = Settings(project_name=f"test_sampling_weights_collection_{task}", exclude_zero_weight_collection=True)
    trainer = trainer_class(
        overrides={
            "model": TASK2MODEL[task],
            "data": TASK2DATASET[task],
            "settings": settings,
            "workers": 2,
        }
    )

    # Trainer needs a model attribute to create dataloader
    trainer.model = Mock() if task == "classify" else None

    # Create edited table where several samples have weight 0
    train_table = trainer.data["train"]
    edited_table = tlc.EditedTable(
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

    settings = Settings(project_name="test_train_no_weight_column_in_table")
    model.train(data=TASK2DATASET[task], settings=settings, epochs=1, device="cpu", workers=0)
    table = model.trainer.data["train"]

    no_weight_column_table = table.delete_column(table.weights_column_name)
    tables = {"train": no_weight_column_table, "val": model.trainer.data.get("val") or model.trainer.data["test"]}

    model.train(tables=tables, settings=settings, epochs=1, device="cpu", workers=0)

    # Should fail to train with weights enabled on table with no weight column
    with pytest.raises(ValueError):
        settings = Settings(project_name="test_train_no_weight_column_in_table", sampling_weights=True)
        model.train(tables=tables, settings=settings, workers=0, epochs=1, device="cpu")

    # Should collect with exclusion enabled and no weight column
    settings = Settings(project_name="test_train_no_weight_column_in_table", exclude_zero_weight_collection=True)
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
        data_dict = tlc_check_det_dataset(
            data=TASK2DATASET["detect"],
            tables=None,
            image_column_name="image",
            label_column_name=label_column_name,
            project_name=settings.project_name,
        )
    elif task == "classify":
        data_dict = tlc_check_cls_dataset(
            data=TASK2DATASET["classify"],
            tables=None,
            image_column_name="image",
            label_column_name=label_column_name,
            project_name=settings.project_name,
        )

    elif task == "segment":
        data_dict = tlc_check_seg_dataset(
            data=TASK2DATASET["segment"],
            tables=None,
            image_column_name="image",
            label_column_name=label_column_name,
            project_name=settings.project_name,
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
                bb_list_override = []
                for bb in row["bbs"]["bb_list"]:
                    bb_list_override.append({**bb, "label": label_map[bb["label"]]})

                bbs_edits.append([i])
                bbs_edits.append({**row["bbs"], "bb_list": bb_list_override})

            edited_tables[split] = tlc.EditedTable(
                url=edited_schema_table.url.create_sibling(f"edited_value_map_and_values_{task}"),
                input_table_url=edited_schema_table,
                edits={"bbs": {"runs_and_values": bbs_edits}},
            )
        elif task == "classify":
            edits = []
            for i, row in enumerate(edited_schema_table.table_rows):
                edits.append([i])
                edits.append(label_map[row[label_column_name]])

            edited_tables[split] = tlc.EditedTable(
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

            edited_tables[split] = tlc.EditedTable(
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
            assert all(bb["label"] <= 0 for bb in metrics_df["bbs_predicted"][i]["bb_list"])

        # Verify that a giraffe is predicted in the second image
        predicted_label = np.sqrt(-metrics_df["bbs_predicted"][1]["bb_list"][0]["label"])
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

    train_structure = {"image": tlc.ImagePath("image"), "label": tlc.CategoricalLabel("label", classes=train_classes)}
    val_structure = {"image": tlc.ImagePath("image"), "label": tlc.CategoricalLabel("label", classes=val_classes)}

    train_table = tlc.Table.from_dict(
        {"image": ["a.jpg", "b.jpg"], "label": [0, 1]},
        structure=train_structure,
        project_name=project_name,
        dataset_name="train",
    )
    val_table = tlc.Table.from_dict(
        {"image": ["c.jpg", "d.jpg"], "label": [0, 1]},
        structure=val_structure,
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
        )


def test_check_tlc_dataset_bad_tables() -> None:
    # Test that an error is raised if tables or urls are not provided properly
    tables = {"train": [1, 2, 3], "val": [4, 5, 6]}

    with pytest.raises(ValueError):
        check_tlc_dataset(data="", tables=tables, image_column_name="a", label_column_name="b")


def test_check_tlc_dataset_bad_url() -> None:
    # Test that an error is raised if a non-valid url is provided
    tables = {"train": "some_url", "val": "some_other_url"}

    with pytest.raises(ValueError):
        check_tlc_dataset(data="", tables=tables, image_column_name="a", label_column_name="b")


def test_small_segmentations() -> None:
    # Test that small segmentations are skipped properly
    structure = {
        "image": tlc.ImagePath("image"),
        "segmentations": tlc.InstanceSegmentationPolygons(
            name="segmentations",
            instance_properties_structure={"label": tlc.CategoricalLabel("label", ["a", "b", "c"])},
            relative=True,
        ),
    }
    zidane_image_path = DUMMY_IMAGE_FILE.as_posix()

    relative_polygons_sample = {
        "image": zidane_image_path,
        "segmentations": {
            "image_width": 10,
            "image_height": 10,
            "instance_properties": {
                "label": [0, 1, 2],
            },
            "polygons": [
                [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],  # Should be fine
                [0.0, 0.0, 0.5, 0.0, 1.0, 0.0],  # A line with no area, should be ignored
                [0.0, 0.0, 0.01, 0.0, 0.01, 0.01, 0.0, 0.01],  # Should become a one pixel mask, which should be ignored
            ],
        },
    }

    table_writer = tlc.TableWriter(
        column_schemas=structure, project_name="test_small_segmentations", dataset_name="test", table_name="initial"
    )
    table_writer.add_row(relative_polygons_sample)
    table = table_writer.finalize()

    assert len(table[0]["segmentations"]["polygons"]) == 3  # All three instances should be present in some way
    assert len(table[0]["segmentations"]["polygons"][0]) == 6  # Expecting a full polygon
    assert len(table[0]["segmentations"]["polygons"][1]) < 6  # Expecting some kind of zero area polygon
    assert len(table[0]["segmentations"]["polygons"][2]) == 0  # Expecting an empty list

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
        "image": tlc.ImagePath("image"),
        "segmentations": tlc.InstanceSegmentationPolygons(
            name="segmentations",
            instance_properties_structure={"label": tlc.CategoricalLabel("label", ["a", "b", "c"])},
            relative=False,
        ),
    }

    table_writer = tlc.TableWriter(
        column_schemas=structure,
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
    url = tlc.Url("<UNEXPANDED_ALIAS>/in/my/url.png")
    with pytest.raises(ValueError):
        TLCDatasetMixin._absolutize_image_url(url, tlc.Url("some_table_url"))


def test_extra_metrics() -> None:
    """Test providing extra metrics callback and schemas work as expected"""

    yolo_dataset_path, (table_train, table_val) = _create_test_image_and_table()

    BATCH_SIZE = 2

    def extra_metrics(preds, batch):
        return {
            "constant_metric": [1] * BATCH_SIZE * 2,
            "metric_with_schema": list(range(BATCH_SIZE * 2)),
        }

    metric_schemas = {
        "metric_with_schema": tlc.Schema(
            value=tlc.Int32Value(value_map={float(i): tlc.MapElement(f"value_{i}") for i in range(BATCH_SIZE * 2)}),
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

        constant_column = metrics_table.get_column("constant_metric").to_numpy()
        assert np.all(constant_column == 1), "Constant metric should be 1"
        metric_with_schema_column = metrics_table.get_column("metric_with_schema").to_numpy()
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
def test_dataset_determinism(mode) -> None:
    """Test that datasets are deterministic with the same seed across separate processes."""
    from dataset_determinism import create_dataset_samples, _compare_dataset_rows

    rows_3lc, rows_ultralytics = create_dataset_samples(mode)

    assert len(rows_3lc) == len(rows_ultralytics), "Number of batches should be the same"

    for row_3lc, row_ultralytics in zip(rows_3lc, rows_ultralytics):
        _compare_dataset_rows(row_ultralytics, row_3lc)


@pytest.mark.parametrize("mode", ["train", "val"])
def test_dataset_determinism_with_random_tracking(mode) -> None:
    """Test that datasets are deterministic and don't make unexpected random calls.

    This test spawns a subprocess, enables random tracking, and checks that no
    unexpected random calls are made.
    """
    import json
    import subprocess
    import sys
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as temp_file:
        output_file = temp_file.name
        cmd = [
            sys.executable,
            "-c",
            "from dataset_determinism import create_dataset_samples_with_tracking;"
            f"create_dataset_samples_with_tracking('{mode}', '{output_file}')",
        ]
        subprocess.run(cmd, check=True)

        with open(output_file) as f:
            tracking_result = json.load(f)

        assert tracking_result["rows_count_3lc"] == tracking_result["rows_count_ultralytics"], (
            "Number of batches should be the same"
        )


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

    table_train = tlc.Table.from_yolo(yolo_dataset_file, "train", if_exists="overwrite")
    table_val = tlc.Table.from_yolo(yolo_dataset_file, "val", if_exists="overwrite")

    return yolo_dataset_file, (table_train, table_val)
