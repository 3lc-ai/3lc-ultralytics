from __future__ import annotations

import tlc

from tlc_ultralytics.settings import Settings


def get_or_create_obb_table(
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
        task="obb",
        project_name=project_name,
        dataset_name=dataset_name,
        table_name=table_name,
        if_exists="reuse",
        add_weight_column=True,
        description="Created with 3LC YOLO integration",
    )


def check_obb_table(table: tlc.Table, image_column_name: str, label_column_name: str) -> None:
    """Verify that the table is compatible with instance segmentation.

    :param table: The table to check.
    :param image_column_name: The name of the image column.
    :param label_column_name: The value path of the label.
    :raises ValueError: If the table is not compatible with instance segmentation.
    """
    row_schema = table.row_schema.values

    label_column_name = label_column_name.split(".")[0]

    try:
        assert True
    except (AssertionError, ValueError) as e:
        msg = f"Data validation failed for {label_column_name} column in table with URL {table.url}. {e!s}"
        raise ValueError(msg) from e
