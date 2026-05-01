from __future__ import annotations

from typing import TYPE_CHECKING

import tlc
from tlc.helpers import AnnotationHelper, AnnotationType

if TYPE_CHECKING:
    from tlc_ultralytics.settings import Settings


def get_or_create_pose_table(
    key: str,
    data_dict: dict[str, object],
    image_column_name: str,
    label_column_name: str,
    project_name: str,
    dataset_name: str,
    table_name: str,
    settings: Settings | None = None,
) -> tlc.Table:
    return tlc.Table.from_yolo(
        dataset_yaml_file=data_dict["yaml_file"],
        split=key,
        override_split_path=data_dict[key],
        task="pose",
        project_name=project_name,
        dataset_name=dataset_name,
        table_name=table_name,
        if_exists="reuse",
        add_weight_column=True,
        description="Created with 3LC YOLO integration",
        points=settings.points,
        lines=settings.lines,
        triangles=settings.triangles,
        point_attributes=settings.point_attributes,
        line_attributes=settings.line_attributes,
        triangle_attributes=settings.triangle_attributes,
        flip_indices=settings.flip_indices,
    )


def check_pose_table(table: tlc.Table, image_column_name: str, label_column_name: str) -> None:
    """Verify that the table is compatible with pose keypoints.

    :param table: The table to check.
    :param image_column_name: The name of the image column.
    :param label_column_name: The name of the pose label root column (e.g., 'pose').
    :raises ValueError: If the table is not compatible with pose.
    """
    label_root = label_column_name.split(".")[0]

    try:
        assert image_column_name in table.row_schema.values, f"Image column '{image_column_name}' not found."
        ann = AnnotationHelper.get(table, label_root)
        assert ann.type is AnnotationType.KEYPOINTS, (
            f"Label column '{label_root}' is not a keypoints column (got {ann.type})."
        )
        assert ann.label_path is not None, f"Pose column '{label_root}' missing label."
    except (AssertionError, KeyError, ValueError) as e:
        raise ValueError(f"Table with url {table.url} is not compatible with YOLO pose. {e}") from None


def yolo_pose_loss_schemas(training: bool = False) -> dict[str, tlc.Schema]:
    """Create 3LC schemas for YOLO pose per-sample loss metrics.

    :param training: Whether metrics are collected during training.
    :returns: The YOLO pose loss schemas for each component.
    """
    schemas: dict[str, tlc.Schema] = {}
    schemas["box_loss"] = tlc.schemas.Float32Schema(
        description="Box Loss",
        writable=False,
        display_importance=3004,
    )
    schemas["dfl_loss"] = tlc.schemas.Float32Schema(
        description="Distribution Focal Loss",
        writable=False,
        display_importance=3005,
    )
    schemas["cls_loss"] = tlc.schemas.Float32Schema(
        description="Classification Loss",
        writable=False,
        display_importance=3006,
    )
    schemas["pose_loss"] = tlc.schemas.Float32Schema(
        description="Keypoint location loss",
        writable=False,
        display_importance=3008,
    )
    schemas["kobj_loss"] = tlc.schemas.Float32Schema(
        description="Keypoint visibility/objectness loss",
        writable=False,
        display_importance=3009,
    )
    if training:
        schemas["loss"] = tlc.schemas.Float32Schema(
            description=(
                "Weighted sum of box, DFL, classification, keypoint location and visibility losses used in training"
            ),
            writable=False,
            display_importance=3010,
        )
    return schemas
