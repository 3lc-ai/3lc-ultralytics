from __future__ import annotations

from typing import TYPE_CHECKING

from tlc.helpers import AnnotationHelper, AnnotationType

if TYPE_CHECKING:
    import tlc


def check_seg_table(table: tlc.Table, image_column_name: str, label_column_name: str) -> None:
    """Verify that the table is compatible with instance segmentation.

    :param table: The table to check.
    :param image_column_name: The name of the image column.
    :param label_column_name: The value path of the label.
    :raises ValueError: If the table is not compatible with instance segmentation.
    """
    label_column_name = label_column_name.split(".")[0]
    try:
        assert image_column_name in table.rows_schema.values, f"Image column '{image_column_name}' not found."
        ann = AnnotationHelper.get(table, label_column_name)
        assert ann.type is AnnotationType.SEGMENTATION, (
            f"Label column '{label_column_name}' is not a segmentation column (got {ann.type})."
        )
        assert ann.label_path is not None, f"Label column '{label_column_name}' missing label."
    except (AssertionError, KeyError, ValueError) as e:
        msg = f"Validation failed for {label_column_name} column in table with URL {table.url}. {e!s}"
        raise ValueError(msg) from e
