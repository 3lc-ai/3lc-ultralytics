from __future__ import annotations

from typing import TYPE_CHECKING

from tlc.helpers import AnnotationHelper, AnnotationType

from tlc_ultralytics.constants import IMAGE_COLUMN_NAME
from tlc_ultralytics.utils.dataset import resolve_annotation_label_path

if TYPE_CHECKING:
    import tlc


def check_obb_table(
    table: tlc.Table,
    image_column_name: str = IMAGE_COLUMN_NAME,
    label_column_name: str | None = None,
) -> None:
    """Verify that the table is compatible with oriented bounding boxes.

    The oriented-bounding-box column is resolved via `resolve_annotation_label_path`: when the root column of
    `label_column_name` is absent (or it is None), the column is inferred so differently-named tables are accepted.

    :param table: The table to check.
    :param image_column_name: The name of the image column.
    :param label_column_name: The value path of the label. If None, the default oriented-bounding-box column is used
        (and inference is applied as needed).
    :raises ValueError: If the table is not compatible with oriented bounding boxes.
    """
    label_root = resolve_annotation_label_path(table, label_column_name, "obb").split(".")[0]
    try:
        assert image_column_name in table.rows_schema.values, f"Image column '{image_column_name}' not found."
        ann = AnnotationHelper.get(table, label_root)
        assert ann.type is AnnotationType.ORIENTED_BOUNDING_BOXES, (
            f"Label column '{label_root}' is not an oriented bounding box column (got {ann.type})."
        )
        assert ann.label_path is not None, f"Label column '{label_root}' missing label."
    except (AssertionError, KeyError, ValueError) as e:
        msg = f"Data validation failed for {label_root} column in table with URL {table.url}. {e!s}"
        raise ValueError(msg) from e
