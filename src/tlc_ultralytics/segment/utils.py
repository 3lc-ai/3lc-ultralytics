from __future__ import annotations

from typing import TYPE_CHECKING

from tlc.helpers import AnnotationHelper, AnnotationType

from tlc_ultralytics.constants import IMAGE_COLUMN_NAME
from tlc_ultralytics.utils.dataset import is_semantic_segmentation_column, resolve_annotation_label_path

if TYPE_CHECKING:
    import tlc


def check_seg_table(
    table: tlc.Table,
    image_column_name: str = IMAGE_COLUMN_NAME,
    label_column_name: str | None = None,
) -> None:
    """Verify that the table is compatible with instance segmentation.

    The segmentation column is resolved via `resolve_annotation_label_path`: when the root column of
    `label_column_name` is absent (or it is None), the column is inferred so differently-named segmentation tables are
    accepted.

    :param table: The table to check.
    :param image_column_name: The name of the image column.
    :param label_column_name: The value path of the label. If None, the default segmentation label path is used (and
        inference is applied as needed).
    :raises ValueError: If the table is not compatible with instance segmentation.
    """
    label_column_name = resolve_annotation_label_path(table, label_column_name, "segment")
    label_root = label_column_name.split(".")[0]
    try:
        assert image_column_name in table.rows_schema.values, f"Image column '{image_column_name}' not found."
        # TEMP(annotation-helper-semseg): a semantic segmentation column is stored in the instance segmentation RLE
        # layout, so it passes the structural checks below and AnnotationHelper classifies it as instance
        # segmentation too. Training it as instance segmentation would treat every class region as an instance.
        # Remove once tlc's AnnotationHelper distinguishes semantic segmentation.
        assert not is_semantic_segmentation_column(table, label_root), (
            f"Label column '{label_root}' holds semantic segmentation label maps, not instance segmentation. Use "
            "task='semantic' (a `-sem` model) for it."
        )
        ann = AnnotationHelper.get(table, label_root)
        assert ann.type is AnnotationType.SEGMENTATION, (
            f"Label column '{label_root}' is not a segmentation column (got {ann.type})."
        )
        assert ann.label_path is not None, f"Label column '{label_root}' missing label."
    except (AssertionError, KeyError, ValueError) as e:
        msg = f"Validation failed for {label_root} column in table with URL {table.url}. {e!s}"
        raise ValueError(msg) from e
