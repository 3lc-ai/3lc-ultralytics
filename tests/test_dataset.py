from __future__ import annotations

import io
import json
import logging
import pickle
import random
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
import tlc
from task_config import (
    DUMMY_IMAGE_FILE,
    TASK2DATASET,
    TASK2LABEL_COLUMN_NAME,
    TASK2MODEL,
    TASK2TRAINER,
    TASK2ULTRALYTICS_TRAINER,
)
from testing_helpers import (
    capture_logs,
    compare_dataset_values,
    plot_ultralytics,
    stub_model_with_stride,
)
from tlc._core.objects.tables.from_table.edited_table import EditedTable
from tmp_paths import TMP
from torch.utils.data import WeightedRandomSampler

from tlc_ultralytics import Settings
from tlc_ultralytics.constants import (
    DETECTION_LABEL_COLUMN_NAME,
)
from tlc_ultralytics.detect.dataset import TLCYOLODataset
from tlc_ultralytics.utils.sampler import create_sampler


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


def _weighted_edited_table(trainer, weight: float) -> EditedTable:
    """Return the trainer's train table with the weight of the first sample set to `weight`."""
    train_table = trainer.data["train"]
    return EditedTable(
        url=train_table.url.create_sibling("jonas"),
        input_table_url=train_table,
        edits={train_table.weights_column_name: {"runs_and_values": [[0], weight]}},
    )


def test_sampling_weights_sampler_distribution() -> None:
    # The weighted sampler draws samples proportionally to their weights (no images are loaded)
    settings = Settings(project_name="test_sampling_weights_distribution", sampling_weights=True)
    trainer = TASK2TRAINER["detect"](
        overrides={"data": TASK2DATASET["detect"], "model": TASK2MODEL["detect"], "settings": settings}
    )
    edited_table = _weighted_edited_table(trainer, 2.0)

    sampler = create_sampler(edited_table, "train", settings)
    assert sampler is not None

    epochs = 2000
    sampled_example_ids = []
    for _epoch in range(epochs):
        sampled_example_ids.extend(sampler)

    # Check other samples are sampled within [0.45, 0.55] of the time of the first
    counts = np.bincount(sampled_example_ids)
    relative_counts = counts[1:] / counts[0]
    assert np.allclose(
        relative_counts,
        np.full_like(relative_counts, 0.5),
        atol=0.05,
    ), f"First sample should be sampled twice as often as others, got {counts}"
    assert len(sampled_example_ids) == len(edited_table) * epochs, "Expected no change in the number of samples"


def test_sampling_weights_dataloader() -> None:
    # The training dataloader uses the weighted sampler and yields one epoch of samples
    settings = Settings(project_name="test_sampling_weights", sampling_weights=True)
    trainer = TASK2TRAINER["detect"](
        overrides={
            "data": TASK2DATASET["detect"],
            "model": TASK2MODEL["detect"],
            "settings": settings,
        },
    )

    # Model is normally set up in train(); build_dataset only needs its stride, so use a stub.
    trainer.model = stub_model_with_stride()

    edited_table = _weighted_edited_table(trainer, 2.0)
    dataloader = trainer.get_dataloader(edited_table, batch_size=2, rank=-1)

    assert isinstance(dataloader.sampler, WeightedRandomSampler)

    sampled_example_ids = []
    for batch in dataloader:
        sampled_example_ids.extend(batch["example_id"])
    assert len(sampled_example_ids) == len(edited_table), "Expected one epoch of samples"


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


@pytest.mark.slow
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


OBB_DETERMINISM_XFAIL = pytest.mark.xfail(
    strict=True, reason="Some boxes are identical but rotated by pi/2 and w-h are swapped"
)


@pytest.mark.parametrize("mode", ["train", "val"])
@pytest.mark.parametrize("task", ["detect", "pose", pytest.param("obb", marks=OBB_DETERMINISM_XFAIL)])
def test_dataset_determinism(mode, task) -> None:
    """Test that datasets are deterministic with the same seed across separate processes."""
    from dataset_determinism import _compare_dataset_rows, create_dataset_samples

    rows_3lc, rows_ultralytics = create_dataset_samples(mode, task)

    assert len(rows_3lc) == len(rows_ultralytics), "Number of batches should be the same"

    for row_3lc, row_ultralytics in zip(rows_3lc, rows_ultralytics, strict=False):
        _compare_dataset_rows(row_ultralytics, row_3lc)


@pytest.mark.slow
def test_dataset_determinism_with_random_tracking(subtests) -> None:
    """Test that datasets are deterministic and don't make unexpected random calls, in one fresh subprocess."""
    import json
    import subprocess
    import tempfile

    combinations = [(mode, task) for task in ("detect", "pose", "obb") for mode in ("train", "val")]
    # FIXME: some obb boxes are identical but rotated by pi/2 with w-h swapped. Strict, so remove once fixed.
    known_failing = {"train-obb", "val-obb"}

    TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(TMP)) as temp_dir:
        output_file = (Path(temp_dir) / "output.json").as_posix()
        cmd = [
            sys.executable,
            "-c",
            "from dataset_determinism import create_dataset_samples_with_tracking;"
            f"create_dataset_samples_with_tracking({combinations!r}, '{output_file}')",
        ]
        subprocess.run(cmd, check=True, cwd=str(Path(__file__).parent))

        with open(output_file) as f:
            results = json.load(f)

    for mode, task in combinations:
        name = f"{mode}-{task}"
        with subtests.test(msg=name):
            tracking_result = results[name]

            if name in known_failing:
                assert "AssertionError" in tracking_result.get("error", ""), (
                    f"Known failing combination {name} now passes or fails differently, update it:\n"
                    f"{tracking_result.get('error', 'no error')}"
                )
                continue

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


# FIXME: known issue with out of order instances in train mode for segment
SEGMENT_TRAIN_XFAIL = pytest.mark.xfail(strict=True, reason="Fails because of out of order instances")


# 3LC stores oriented boxes as rotated rectangles while DOTA labels them as arbitrary quadrilaterals, so corners can
# differ by a few pixels. Mosaic clips instances at the tile borders and drops the ones left without area, which turns
# those few pixels into one more surviving instance on the 3LC side. The raw labels are identical and every other
# instance matches, so only the augmented comparison is affected.
OBB_TRAIN_XFAIL = pytest.mark.xfail(
    strict=True, reason="One sliver-sized instance survives mosaic border clipping on one side only"
)


@pytest.mark.parametrize(
    "task,mode",
    [
        ("pose", "train"),
        ("pose", "val"),
        pytest.param("obb", "train", marks=OBB_TRAIN_XFAIL),
        ("obb", "val"),
        ("detect", "train"),
        ("detect", "val"),
        pytest.param("segment", "train", marks=SEGMENT_TRAIN_XFAIL),
        ("segment", "val"),
    ],
)
def test_single_sample_equality(task: str, mode: str) -> None:
    """Test that a single sample from the dataset is equal between 3LC and Ultralytics."""

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
