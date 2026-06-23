from __future__ import annotations

import tlc

from tlc_ultralytics.constants import IMAGE_COLUMN_NAME
from tlc_ultralytics.detect.dataset import TLCYOLODataset
from tlc_ultralytics.utils.dataset import resolve_annotation_label_path, resolve_label_value_path


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

    Supports both legacy (BoundingBoxListSchema) and new (BoundingBoxes2D) formats. The bounding-box column is resolved
    via `resolve_annotation_label_path`: when the root column of `label_column_name` is absent (or it is None), the
    column is inferred so differently-named bounding-box tables are accepted.

    :param table: The table to check.
    :param image_column_name: The name of the column containing image paths.
    :param label_column_name: The full label path of the column containing labels. If None, the default detection label
        path is used (and inference is applied as needed).
    :raises: ValueError if the table is not compatible with the detection task.
    """
    # Resolve (and, if needed, infer) the bounding-box column. Raises a precise ValueError when the
    # table genuinely has no bounding boxes.
    label_column_name = resolve_annotation_label_path(table, label_column_name, "detect")

    try:
        assert image_column_name in table.rows_schema.values, f"Image column '{image_column_name}' not found."

        # Validate the label path resolves to a value map, falling back to the label path resolved
        # by AnnotationHelper (e.g. `bbs.bb_list.label` for legacy tables).
        label_path = resolve_label_value_path(table, label_column_name)
        assert table.get_value_map(label_path) is not None, (
            f"Unable to get value map for label value path {label_column_name}. Ensure that the table is "
            "compatible with the detection task or provide a `label_column_name` that matches the value path."
        )

    except (AssertionError, KeyError) as e:
        raise ValueError(f"Table with url {table.url} is not compatible with YOLO object detection. {e}") from None


def yolo_predicted_bounding_box_schema(
    label_value_map: dict[float, tlc.schemas.MapElement],
) -> tlc.Schema:
    """Create a 3LC bounding box schema for YOLO predicted boxes.

    :param label_value_map: Mapping of class indices to label metadata.
    :returns: A schema for predicted bounding boxes.
    """
    return tlc.data_types.BoundingBoxes2D.schema(
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
    )
    schemas["dfl_loss"] = tlc.schemas.Float32Schema(
        description="Distribution Focal Loss",
        writable=False,
    )
    schemas["cls_loss"] = tlc.schemas.Float32Schema(
        description="Classification Loss",
        writable=False,
    )
    if training:
        schemas["loss"] = tlc.schemas.Float32Schema(
            description="Weighted sum of box, DFL, and classification losses used in training",
            writable=False,
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
    from tlc.data_types import BoundingBoxes2D

    if not predicted_annotations:
        bb2d = BoundingBoxes2D.create_empty(
            image_width=image_width,
            image_height=image_height,
        )
    else:
        # Predictions arrive in normalized cxywh (YOLO format); the constructor handles
        # both denormalization (via image_width/height + normalized=True) and cxywh → xyxy.
        cxywh_norm = np.array([pred["bbox"] for pred in predicted_annotations], dtype=np.float32)

        labels = []
        confidences = []
        for pred in predicted_annotations:
            label = pred["category_id"]
            if inverse_label_mapping is not None:
                label = inverse_label_mapping[label]
            labels.append(int(label))
            confidences.append(float(pred["score"]))

        bb2d = BoundingBoxes2D(
            bounding_boxes=cxywh_norm,
            bounding_box_format="cxywh",
            normalized=True,
            image_width=image_width,
            image_height=image_height,
            labels=labels,
            confidences=confidences,
        )

    return bb2d.to_row()
