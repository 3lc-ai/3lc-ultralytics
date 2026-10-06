from __future__ import annotations

import logging

import numpy as np
import pytest
import tlc
from task_config import (
    DUMMY_IMAGE_FILE,
)
from testing_helpers import (
    capture_logs,
)

from tlc_ultralytics import Settings
from tlc_ultralytics.constants import (
    DETECTION_LABEL_COLUMN_NAME,
    OBB_LABEL_COLUMN_NAME,
    POSE_LABEL_COLUMN_NAME,
    SEGMENTATION_LABEL_COLUMN_NAME,
)
from tlc_ultralytics.segment.utils import check_seg_table
from tlc_ultralytics.utils.dataset import _complete_label_column_name


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


def test_complete_label_column_name() -> None:
    assert _complete_label_column_name("a", "a") == "a"
    assert _complete_label_column_name("a", "a.b.c") == "a.b.c"
    assert _complete_label_column_name("a.b.c", "d.e.f") == "a.b.c"
    assert _complete_label_column_name("", "a.b.c") == "a.b.c"
