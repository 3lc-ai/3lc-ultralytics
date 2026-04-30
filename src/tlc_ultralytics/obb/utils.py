from __future__ import annotations

from typing import TYPE_CHECKING

import tlc
from tlc.helpers import AnnotationHelper, AnnotationType

if TYPE_CHECKING:
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
    """Verify that the table is compatible with oriented bounding boxes.

    :param table: The table to check.
    :param image_column_name: The name of the image column.
    :param label_column_name: The value path of the label.
    :raises ValueError: If the table is not compatible with oriented bounding boxes.
    """
    label_column_name = label_column_name.split(".")[0]
    try:
        assert image_column_name in table.row_schema, f"Image column '{image_column_name}' not found."
        ann = AnnotationHelper.get(table, label_column_name)
        assert ann.type is AnnotationType.ORIENTED_BOUNDING_BOXES, (
            f"Label column '{label_column_name}' is not an oriented bounding box column (got {ann.type})."
        )
        assert ann.label_path is not None, f"Label column '{label_column_name}' missing label."
    except (AssertionError, KeyError, ValueError) as e:
        msg = f"Data validation failed for {label_column_name} column in table with URL {table.url}. {e!s}"
        raise ValueError(msg) from e
