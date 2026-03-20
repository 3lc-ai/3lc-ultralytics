from __future__ import annotations

from typing import TYPE_CHECKING

import tlc
from tlc.core.sample_types.registry import SampleTypeRegistry

if TYPE_CHECKING:
    from tlc_ultralytics.settings import Settings


def get_or_create_seg_table(
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
        task="segment",
        project_name=project_name,
        dataset_name=dataset_name,
        table_name=table_name,
        if_exists="reuse",
        add_weight_column=True,
        description="Created with 3LC YOLO integration",
    )


def check_seg_table(table: tlc.Table, image_column_name: str, label_column_name: str) -> None:
    """Verify that the table is compatible with instance segmentation.

    :param table: The table to check.
    :param image_column_name: The name of the image column.
    :param label_column_name: The value path of the label.
    :raises ValueError: If the table is not compatible with instance segmentation.
    """
    row_schema = table.row_schema.values

    label_column_name = label_column_name.split(".")[0]

    # Check that the schema and data are compatible with instance segmentation
    try:
        # Schema checks
        assert image_column_name in row_schema, f"Image column '{image_column_name}' not found."
        assert label_column_name in row_schema, f"Label column '{label_column_name}' not found."

        # Data checks
        first_row = table.table_rows[0]
        assert isinstance(first_row[label_column_name], dict), f"Label column '{label_column_name}' must be a dictionary."
        assert "rles" in first_row[label_column_name], f"Label column '{label_column_name}' missing rles."
        assert "image_width" in first_row[label_column_name], f"Label column '{label_column_name}' missing image_width."
        assert "image_height" in first_row[label_column_name], f"Label column '{label_column_name}' missing image_height."
        assert "instance_properties" in first_row[label_column_name], f"Label column '{label_column_name}' missing instance_properties."
        assert "label" in first_row[label_column_name]["instance_properties"], f"Label column '{label_column_name}' missing label."
        assert image_column_name in first_row, (
            f"Image column {image_column_name} not found in table with URL {table.url}"
        )

    except (AssertionError, ValueError) as e:
        msg = f"Validation failed for {label_column_name} column in table with URL {table.url}. {e!s}"
        raise ValueError(msg) from e
