from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import tlc
from ultralytics.data.utils import check_det_dataset

from tlc_ultralytics.utils import check_tlc_dataset


def tlc_check_pose_dataset(
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
        table_creator=get_or_create_pose_table,
        table_checker=check_pose_table,
        project_name=project_name,
        splits=splits,
    )


def get_or_create_pose_table(
    key: str,
    data_dict: dict[str, object],
    image_column_name: str,
    label_column_name: str,
    project_name: str,
    dataset_name: str,
    table_name: str,
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
        for key in (tlc.IMAGE_WIDTH, tlc.IMAGE_HEIGHT, "instances"):
            assert key in schema.values, f"Pose column '{label_root}' missing key '{key}'."

        instances_schema = schema.values["instances"]
        assert hasattr(instances_schema, "values"), "Instances schema must be composite."
        for k in ("xys", "lines"):
            assert k in instances_schema.values, f"Instances missing '{k}'."

    except (AssertionError, KeyError) as e:
        raise ValueError(f"Table with url {table.url} is not compatible with YOLO pose. {e}") from None
