from __future__ import annotations

import tlc
from tlc.helpers import AnnotationHelper, AnnotationType

from tlc_ultralytics.constants import DETECTION_LABEL_COLUMN_NAME, IMAGE_COLUMN_NAME
from tlc_ultralytics.detect.dataset import TLCYOLODataset
from tlc_ultralytics.utils.dataset import resolve_label_value_path


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


def infer_detection_label_column_name(table: tlc.Table, label_column_name: str) -> str:
    """Resolve the bounding-box label column path for a detection table.

    If the root column of ``label_column_name`` (e.g. ``"bbs"``) exists in the table, the path is
    returned unchanged. Otherwise the bounding-box column is inferred via ``AnnotationHelper.find``,
    and the inferred column's label path is returned instead. This lets detection tables whose
    bounding-box column is not named ``"bbs"`` work without the user specifying ``label_column_name``.

    :param table: The table to resolve the label path against.
    :param label_column_name: The configured (possibly default) full label value path.
    :returns: A label value path whose root column exists in the table.
    :raises ValueError: If the root column is absent and no bounding-box column can be inferred (the
        message names the annotation type the table does have, and the columns present), or if
        ``AnnotationHelper.find`` matches more than one bounding-box column (it raises its own
        ``ValueError`` listing the candidates).
    """
    row_schema = table.rows_schema.values

    if label_column_name.split(".")[0] in row_schema:
        return label_column_name

    # The configured/default bounding-box column is not present — infer it. Asking for
    # BOUNDING_BOXES matches both new and legacy (`bbs.bb_list.label`) bounding-box columns.
    ann = AnnotationHelper.find(table, type=AnnotationType.BOUNDING_BOXES)
    if ann is not None and ann.label_path is not None:
        return ann.label_path

    # No bounding boxes — produce a precise, actionable message naming what the table does have.
    columns = ", ".join(f"'{name}'" for name in row_schema)
    other = AnnotationHelper.find(table, type=None)
    if other is not None:
        detail = (
            f"this table has {other.type.name} annotations in column '{other.name}', not bounding boxes. "
            f"Use the task matching {other.type.name} annotations instead."
        )
    else:
        detail = "no annotation columns were found in the table."
    raise ValueError(
        f"Table with url {table.url} is not compatible with YOLO object detection: {detail} "
        f"Columns present: {columns}."
    )


def check_det_table(
    table: tlc.Table,
    image_column_name: str = IMAGE_COLUMN_NAME,
    label_column_name: str | None = None,
) -> None:
    """Check that a table is compatible with the detection task in the 3LC YOLO integration.

    Supports both legacy (BoundingBoxListSchema) and new (BoundingBoxes2D) formats. When the root
    column of ``label_column_name`` is absent, the bounding-box column is inferred via
    ``infer_detection_label_column_name`` so differently-named bounding-box tables are accepted.

    :param table: The table to check.
    :param image_column_name: The name of the column containing image paths.
    :param label_column_name: The full label path of the column containing labels.
        If None, the default detection label path is used.
    :raises: ValueError if the table is not compatible with the detection task.
    """
    if label_column_name is None:
        label_column_name = DETECTION_LABEL_COLUMN_NAME

    row_schema = table.rows_schema.values

    # Infer the bounding-box column if the configured one is absent. Raises a precise ValueError
    # when the table genuinely has no bounding boxes.
    label_column_name = infer_detection_label_column_name(table, label_column_name)

    try:
        assert image_column_name in row_schema, f"Image column '{image_column_name}' not found."

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
