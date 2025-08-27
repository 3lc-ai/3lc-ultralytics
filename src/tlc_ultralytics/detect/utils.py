from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import tlc
from tlc.client.torch.metrics.metrics_collectors.bounding_box_metrics_collector import (
    _TLCPredictedBoundingBox,
    _TLCPredictedBoundingBoxes,
)
from ultralytics.data.utils import check_det_dataset

from tlc_ultralytics.detect.dataset import TLCYOLODataset
from tlc_ultralytics.utils import check_tlc_dataset


def tlc_check_det_dataset(
    data: str,
    tables: dict[str, tlc.Table | tlc.Url | Path | str] | None,
    image_column_name: str,
    label_column_name: str,
    project_name: str | None = None,
    splits: Iterable[str] | None = None,
) -> dict[str, tlc.Table | dict[float, str] | int]:
    return check_tlc_dataset(
        data,
        tables,
        image_column_name,
        label_column_name,
        dataset_checker=check_det_dataset,
        table_creator=get_or_create_det_table,
        table_checker=check_det_table,
        project_name=project_name,
        splits=splits,
    )


def get_or_create_det_table(
    key: str,
    data_dict: dict[str, object],
    image_column_name: str,
    label_column_name: str,
    project_name: str,
    dataset_name: str,
    table_name: str,
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
    image_column_name: str = tlc.IMAGE,
    label_column_name: str = f"{tlc.BOUNDING_BOXES}.{tlc.BOUNDING_BOX_LIST}.{tlc.LABEL}",
) -> None:
    """Check that a table is compatible with the detection task in the 3LC YOLO integration.

    :param table: The table to check.
    :param image_column_name: The name of the column containing image paths.
    :param label_column_name: The full label path of the column containing labels.
    :raises: ValueError if the table is not compatible with the detection task.
    """
    row_schema = table.row_schema.values

    bounding_boxes_column_key, bounding_boxes_list_key, label_key = label_column_name.split(".")

    try:
        assert image_column_name in row_schema, f"Image column '{image_column_name}' not found."
        assert bounding_boxes_column_key in row_schema, f"Bounding box column '{bounding_boxes_column_key}' not found."
        assert bounding_boxes_list_key in row_schema[bounding_boxes_column_key].values, (
            f"Bounding box list '{bounding_boxes_list_key}' not found in column '{bounding_boxes_column_key}'."
        )

        assert tlc.IMAGE_HEIGHT in row_schema[bounding_boxes_column_key].values, (
            f"Bounding box column '{bounding_boxes_column_key}' does not contain a key '{tlc.IMAGE_HEIGHT}'."
        )
        assert tlc.IMAGE_WIDTH in row_schema[bounding_boxes_column_key].values, (
            f"Bounding box column '{bounding_boxes_column_key}' does not contain a key '{tlc.IMAGE_WIDTH}'."
        )

        for coordinate in [tlc.X0, tlc.Y0, tlc.X1, tlc.Y1]:
            assert coordinate in row_schema[bounding_boxes_column_key].values[bounding_boxes_list_key].values, (
                f"Bounding box list '{bounding_boxes_list_key}' in column '{bounding_boxes_column_key}' "
                f"does not contain a key '{coordinate}'."
            )
        assert label_key in row_schema[bounding_boxes_column_key].values[bounding_boxes_list_key].values, (
            f"Bounding box list '{bounding_boxes_list_key}' in column '{bounding_boxes_column_key}' "
            f"does not contain a key '{label_key}'."
        )
        assert table.get_value_map(label_column_name) is not None, (
            f"Unable to get value map for label value path {label_column_name}. Ensure that the table is compatible "
            "with the detection task or provide a `label_column_name` that matches the value path to the labels."
        )

    except (AssertionError, KeyError) as e:
        raise ValueError(f"Table with url {table.url} is not compatible with YOLO object detection. {e}") from None


def yolo_predicted_bounding_box_schema(
    label_value_map: dict[float, tlc.MapElement],
) -> tlc.Schema:
    """Create a 3LC bounding box schema for YOLO.

    :param categories: Categories for the current dataset.
    :returns: The YOLO bounding box schema for predicted boxes.
    """

    bounding_box_schema = tlc.BoundingBoxListSchema(
        label_value_map=label_value_map,
        x0_number_role=tlc.NUMBER_ROLE_BB_CENTER_X,
        x1_number_role=tlc.NUMBER_ROLE_BB_SIZE_X,
        y0_number_role=tlc.NUMBER_ROLE_BB_CENTER_Y,
        y1_number_role=tlc.NUMBER_ROLE_BB_SIZE_Y,
        x0_unit=tlc.UNIT_RELATIVE,
        y0_unit=tlc.UNIT_RELATIVE,
        x1_unit=tlc.UNIT_RELATIVE,
        y1_unit=tlc.UNIT_RELATIVE,
        description="Predicted Bounding Boxes",
        writable=False,
        is_prediction=True,
        include_segmentation=False,
    )

    return bounding_box_schema


def yolo_loss_schemas(training: bool = False) -> dict[str, tlc.Schema]:
    """Create a 3LC schema for YOLO per-sample loss metrics.

    :param training: Whether metrics are collected during training.
    :returns: The YOLO loss schemas for each of the three components.
    """
    schemas = {}
    schemas["box_loss"] = tlc.Schema(
        description="Box Loss",
        writable=False,
        value=tlc.Float32Value(),
        display_importance=3004,
    )
    schemas["dfl_loss"] = tlc.Schema(
        description="Distribution Focal Loss",
        writable=False,
        value=tlc.Float32Value(),
        display_importance=3005,
    )
    schemas["cls_loss"] = tlc.Schema(
        description="Classification Loss",
        writable=False,
        value=tlc.Float32Value(),
        display_importance=3006,
    )
    if training:
        schemas["loss"] = tlc.Schema(
            description="Weighted sum of box, DFL, and classification losses used in training",
            writable=False,
            value=tlc.Float32Value(),
            display_importance=3007,
        )
    return schemas


def construct_bbox_struct(
    predicted_annotations: list[dict[str, int | float | dict[str, float]]],
    image_width: int,
    image_height: int,
    inverse_label_mapping: dict[int, int] | None = None,
) -> _TLCPredictedBoundingBoxes:
    """Construct a 3LC bounding box struct from a list of bounding boxes.

    :param predicted_annotations: A list of predicted bounding boxes.
    :param image_width: The width of the image.
    :param image_height: The height of the image.
    :param inverse_label_mapping: A mapping from predicted label to category id.
    """

    bbox_struct = _TLCPredictedBoundingBoxes(
        bb_list=[],
        image_width=image_width,
        image_height=image_height,
    )

    for pred in predicted_annotations:
        bbox, label, score, iou = (
            pred["bbox"],
            pred["category_id"],
            pred["score"],
            pred["iou"],
        )
        label_val = inverse_label_mapping[label] if inverse_label_mapping is not None else label
        bbox_struct["bb_list"].append(
            _TLCPredictedBoundingBox(
                label=label_val,
                confidence=score,
                iou=iou,
                x0=bbox[0],
                y0=bbox[1],
                x1=bbox[2],
                y1=bbox[3],
            )
        )

    return bbox_struct
