"""COCO RLE encoding of binary mask chunks, shared by the instance and semantic segmentation validators.

Both validators build their masks on the model's device a chunk at a time, laid out `(n, W, H)`: each mask flattened
is then its pixels in column-major order, which is the order a COCO RLE counts runs in. `rles_from_column_major_chunk`
encodes such a chunk wherever it lives, so neither validator ever holds more than one chunk of dense masks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from tlc.helpers import SegmentationHelper

if TYPE_CHECKING:
    import torch
    from tlc.data_types.segmentation import CocoRle


def rles_from_column_major_chunk(masks: torch.Tensor, height: int, width: int) -> list[CocoRle]:
    """COCO RLE-encode a chunk of binary masks laid out `(N, W, H)` uint8, on whichever device it lives.

    On CUDA the runs are found on the GPU (`rles_from_column_major_masks`), so only run boundaries leave the device.
    Elsewhere the chunk is brought to the host, where its `(H, W, N)` view is the Fortran-ordered layout pycocotools
    encodes from, so `SegmentationHelper.rles_from_masks` reads it without a copy. Both give the same RLEs.
    """
    if masks.is_cuda:
        return rles_from_column_major_masks(masks, height, width)
    return SegmentationHelper.rles_from_masks(masks.cpu().numpy().transpose(2, 1, 0))


def rles_from_column_major_masks(masks: torch.Tensor, height: int, width: int) -> list[CocoRle]:
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
