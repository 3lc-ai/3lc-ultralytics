from __future__ import annotations

from typing import TYPE_CHECKING

import tlc
from tlc.helpers import AnnotationHelper, AnnotationType

from tlc_ultralytics.constants import IMAGE_COLUMN_NAME
from tlc_ultralytics.detect.dataset import TLCYOLODataset

if TYPE_CHECKING:
    from tlc_ultralytics.settings import Settings


def get_or_create_det_table(
    key: str,
    data_dict: dict[str, object],
    image_column_name: str,
    label_column_name: str,
    project_name: str,
    dataset_name: str,
    table_name: str,
    settings: Settings | None = None,
) -> tlc.Table:
    """Get or create a detection table from a dataset dictionary.

    :param key: The key of the dataset dictionary (the split to use)
    :param data_dict: Dictionary of dataset information
    :param project_name: Name of the project
    :param dataset_name: Name of the dataset
    :param table_name: Name of the table
    :param image_column_name: Name of the column containing image paths
    :param label_column_name: Name of the column containing labels
    :return: A tlc.Table.from_yolo() table
    """
    return tlc.Table.from_yolo(
        dataset_yaml_file=data_dict["yaml_file"],
        split=key,
        override_split_path=data_dict[key],
        task="detect",
        project_name=project_name,
        dataset_name=dataset_name,
        table_name=table_name,
        if_exists="reuse",
        add_weight_column=True,
        description="Created with 3LC YOLO integration",
    )


def build_tlc_yolo_dataset(
    cfg,
    table,
    batch,
    data,
    mode="train",
    rect=False,
    stride=32,
    multi_modal=False,
    exclude_zero=False,
    class_map=None,
    split=None,
    image_column_name=None,
    label_column_name=None,
):
    if multi_modal:
        return ValueError("Multi-modal datasets are not supported in the 3LC Ultralytics integration.")

    return TLCYOLODataset(
        table,
        exclude_zero=exclude_zero,
        class_map=class_map,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=cfg.rect or rect,  # rectangular batches
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=0.0 if mode == "train" else 0.5,
        prefix=split or mode,
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=cfg.fraction if mode == "train" else 1.0,
        image_column_name=image_column_name,
        label_column_name=label_column_name,
    )


def check_det_table(
    table: tlc.Table,
    image_column_name: str = IMAGE_COLUMN_NAME,
    label_column_name: str | None = None,
) -> None:
    """Check that a table is compatible with the detection task in the 3LC YOLO integration.

    Supports both legacy (BoundingBoxListSchema) and new (BoundingBoxes2DSchema) formats.

    :param table: The table to check.
    :param image_column_name: The name of the column containing image paths.
    :param label_column_name: The full label path of the column containing labels.
        If None, auto-detected from the table schema.
    :raises: ValueError if the table is not compatible with the detection task.
    """
    row_schema = table.row_schema.values

    try:
        assert image_column_name in row_schema, f"Image column '{image_column_name}' not found."

        if label_column_name is not None:
            # User-provided label path — validate it exists
            bb_column = label_column_name.split(".")[0]
            assert bb_column in row_schema, f"Bounding box column '{bb_column}' not found."
            assert table.get_value_map(label_column_name) is not None, (
                f"Unable to get value map for label value path {label_column_name}. Ensure that the table is "
                "compatible with the detection task or provide a `label_column_name` that matches the value path."
            )
        else:
            # Auto-detect bounding box column and label path via AnnotationHelper
            ann = AnnotationHelper.find(table, type=AnnotationType.BOUNDING_BOXES)
            assert ann is not None, "No bounding box column found in the table."
            assert ann.label_path is not None, f"Bounding box column '{ann.name}' found but no label field detected."
            assert table.get_value_map(ann.label_path) is not None, (
                f"Unable to get value map for auto-detected label path '{ann.label_path}'."
            )

    except (AssertionError, KeyError) as e:
        raise ValueError(f"Table with url {table.url} is not compatible with YOLO object detection. {e}") from None


def yolo_predicted_bounding_box_schema(
    label_value_map: dict[float, tlc.MapElement],
) -> tlc.Schema:
    """Create a 3LC bounding box schema for YOLO predicted boxes.

    :param label_value_map: Mapping of class indices to label metadata.
    :returns: A BoundingBoxes2DSchema for predicted boxes.
    """
    return tlc.BoundingBoxes2DSchema(
        classes=label_value_map,
        include_per_instance_confidence=True,
        description="Predicted Bounding Boxes",
        writable=False,
    )


def yolo_loss_schemas(training: bool = False) -> dict[str, tlc.Schema]:
    """Create a 3LC schema for YOLO per-sample loss metrics.

    :param training: Whether metrics are collected during training.
    :returns: The YOLO loss schemas for each of the three components.
    """
    schemas = {}
    schemas["box_loss"] = tlc.schemas.Float32Schema(
        description="Box Loss",
        writable=False,
        display_importance=3004,
    )
    schemas["dfl_loss"] = tlc.Float32Schema(
        description="Distribution Focal Loss",
        writable=False,
        display_importance=3005,
    )
    schemas["cls_loss"] = tlc.Float32Schema(
        description="Classification Loss",
        writable=False,
        display_importance=3006,
    )
    if training:
        schemas["loss"] = tlc.Float32Schema(
            description="Weighted sum of box, DFL, and classification losses used in training",
            writable=False,
            display_importance=3007,
        )
    return schemas


def construct_bbox_struct(
    predicted_annotations: list[dict[str, int | float | dict[str, float]]],
    image_width: int,
    image_height: int,
    inverse_label_mapping: dict[int, int] | None = None,
) -> dict:
    """Construct a BoundingBoxes2D prediction and serialize to wire format.

    :param predicted_annotations: A list of predicted bounding boxes, each with
        "category_id", "score", and "bbox" (normalized center-xywh) keys.
    :param image_width: The width of the image.
    :param image_height: The height of the image.
    :param inverse_label_mapping: A mapping from predicted label to category id.
    :returns: A serialized dict suitable for writing to a 3LC Table.
    """
    import numpy as np
    from tlc.core.data_formats.bb_conversions import denormalize_bbs_2d
    from tlc.core.data_formats.bounding_boxes import BoundingBoxes2D

    if not predicted_annotations:
        bb2d = BoundingBoxes2D.create_empty(
            image_width=image_width,
            image_height=image_height,
        )
    else:
        # Predictions arrive in normalized cxywh (YOLO format); denormalize layout-preserving,
        # then let the constructor convert cxywh → xyxy via bbox_format.
        cxywh_norm = np.array([pred["bbox"] for pred in predicted_annotations], dtype=np.float32)
        cxywh_abs = denormalize_bbs_2d(cxywh_norm, image_width, image_height)

        labels = []
        confidences = []
        for pred in predicted_annotations:
            label = pred["category_id"]
            if inverse_label_mapping is not None:
                label = inverse_label_mapping[label]
            labels.append(int(label))
            confidences.append(float(pred["score"]))

        bb2d = BoundingBoxes2D(
            bboxes=cxywh_abs,
            bbox_format="cxywh",
            labels=labels,
            confidences=confidences,
            x_max=float(image_width),
            y_max=float(image_height),
        )

    return bb2d.to_row()
