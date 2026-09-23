from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from tlc.helpers import AnnotationHelper, AnnotationType

from tlc_ultralytics.constants import IMAGE_COLUMN_NAME
from tlc_ultralytics.utils.dataset import resolve_annotation_label_path

if TYPE_CHECKING:
    import tlc
    import torch


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
        ann = AnnotationHelper.get(table, label_root)
        assert ann.type is AnnotationType.SEGMENTATION, (
            f"Label column '{label_root}' is not a segmentation column (got {ann.type})."
        )
        assert ann.label_path is not None, f"Label column '{label_root}' missing label."
    except (AssertionError, KeyError, ValueError) as e:
        msg = f"Validation failed for {label_root} column in table with URL {table.url}. {e!s}"
        raise ValueError(msg) from e


def rles_from_column_major_masks(masks: torch.Tensor, height: int, width: int) -> list[dict[str, Any]]:
    """COCO RLE-encode binary masks laid out column-major, finding the runs where the masks live.

    `masks` is `(N, W, H)` uint8, so each mask flattened is its pixels in column-major order, which is the order a
    COCO RLE counts runs in. The run boundaries are found with one comparison and `nonzero` on the masks' device,
    so from a GPU only the boundary positions are transferred, never the dense masks. pycocotools then compresses
    the counts into the same `counts` bytes `pycocotools.mask.encode` produces from the dense masks.
    """
    import pycocotools.mask as mask_utils

    num_masks = masks.shape[0]
    length = height * width
    if num_masks == 0:
        return []

    flat = masks.reshape(num_masks, length)
    rows, cols = (flat[:, 1:] != flat[:, :-1]).nonzero(as_tuple=True)  # sorted by mask, then by position
    starts_with_one = flat[:, 0].bool().cpu().numpy()
    rows = rows.cpu().numpy()
    run_starts = (cols + 1).cpu().numpy().astype(np.int64)

    # Lay every mask's run boundaries out as [0, run starts..., length], one mask after another, so a single diff
    # gives all run lengths. Mask i's boundaries start at the number of run starts before it plus 2 * i.
    runs_per_mask = np.bincount(rows, minlength=num_masks)
    first_boundary = np.concatenate(([0], np.cumsum(runs_per_mask + 2)[:-1]))
    boundaries = np.empty(len(run_starts) + 2 * num_masks, dtype=np.int64)
    boundaries[first_boundary] = 0
    boundaries[first_boundary + runs_per_mask + 1] = length
    boundaries[np.arange(len(run_starts)) + 2 * rows + 1] = run_starts
    run_lengths = np.diff(boundaries).astype(np.uint32)

    uncompressed = []
    for i in range(num_masks):
        counts = run_lengths[first_boundary[i] : first_boundary[i] + runs_per_mask[i] + 1]
        if starts_with_one[i]:
            counts = np.concatenate((np.zeros(1, dtype=np.uint32), counts))  # COCO RLE starts with a run of zeros
        uncompressed.append({"size": [height, width], "counts": counts})
    return mask_utils.frPyObjects(uncompressed, height, width)
