from __future__ import annotations

import tlc
from tlc.core.builtins.constants import (
    BBS_2D,
    INSTANCES,
    INSTANCES_ADDITIONAL_DATA,
    VERTICES_2D,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
)

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
    row_schema = table.row_schema.values

    label_root = label_column_name.split(".")[0]

    try:
        assert image_column_name in row_schema, f"Image column '{image_column_name}' not found."
        assert label_root in row_schema, f"Pose column '{label_root}' not found."

        schema = row_schema[label_root]
        assert hasattr(schema, "values"), f"Pose column '{label_root}' has no values schema."
        for key in (X_MIN, Y_MIN, X_MAX, Y_MAX, INSTANCES, INSTANCES_ADDITIONAL_DATA):
            assert key in schema.values, f"Pose column '{label_root}' missing key '{key}'."

        instances_schema = schema.values[INSTANCES]
        assert hasattr(instances_schema, "values"), "Instances schema must be composite."
        for k in (VERTICES_2D, BBS_2D):
            assert k in instances_schema.values, f"Instances missing '{k}'."

    except (AssertionError, KeyError) as e:
        raise ValueError(f"Table with url {table.url} is not compatible with YOLO pose. {e}") from None


def yolo_pose_loss_schemas(training: bool = False) -> dict[str, tlc.Schema]:
    """Create 3LC schemas for YOLO pose per-sample loss metrics.

    :param training: Whether metrics are collected during training.
    :returns: The YOLO pose loss schemas for each component.
    """
    schemas: dict[str, tlc.Schema] = {}
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
    schemas["pose_loss"] = tlc.Schema(
        description="Keypoint location loss",
        writable=False,
        value=tlc.Float32Value(),
        display_importance=3008,
    )
    schemas["kobj_loss"] = tlc.Schema(
        description="Keypoint visibility/objectness loss",
        writable=False,
        value=tlc.Float32Value(),
        display_importance=3009,
    )
    if training:
        schemas["loss"] = tlc.Schema(
            description=(
                "Weighted sum of box, DFL, classification, keypoint location and visibility losses used in training"
            ),
            writable=False,
            value=tlc.Float32Value(),
            display_importance=3010,
        )
    return schemas
