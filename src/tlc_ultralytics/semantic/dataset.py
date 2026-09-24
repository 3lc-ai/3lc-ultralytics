from __future__ import annotations

from typing import Any

import numpy as np
from tlc.constants import IMAGE_HEIGHT, IMAGE_WIDTH, INSTANCE_PROPERTIES, LABEL, RLES
from ultralytics.data.dataset import SemanticDataset

from tlc_ultralytics.detect.dataset import BaseTLCYOLODataset
from tlc_ultralytics.semantic.utils import IGNORE_INDEX, decode_semantic_mask
from tlc_ultralytics.utils.dataset import map_label

SEMANTIC_SHAPE = "tlc_semantic_shape"
SEMANTIC_FILL = "tlc_semantic_fill"
SEMANTIC_VALUES = "tlc_semantic_values"
SEMANTIC_RLES = "tlc_semantic_rles"
"""Label keys holding an image's semantic segmentation in the table's compact row form: its `(H, W)`, the training
class index of pixels no layer covers, and the training class index and COCO RLE counts of each stored layer. The shape
has a key of its own because Ultralytics pops `shape` from the labels in rect mode."""


RESIZED_SHAPE = "tlc_resized_shape"
"""Sample key holding the `(h, w)` an image was resized to on load, before `LetterBox` padded it to the model input.

Ultralytics' own `resized_shape` is overwritten with the padded shape by `LetterBox` and dropped by `SemanticFormat`,
and the load-time resize rounds up where `LetterBox` rounds, so the padding cannot be recomputed from the original
shape."""


class TLCSemanticDataset(BaseTLCYOLODataset, SemanticDataset):
    """3LC YOLO dataset for semantic segmentation.

    Each label keeps the image's segmentation as the table stores it, one RLE layer per class present, with the layers'
    3LC class ids already mapped to training class indices. `load_mask` decodes it into the dense label map Ultralytics'
    `SemanticDataset` would otherwise read from a mask PNG, so the rest of Ultralytics' semantic pipeline (resizing,
    augmentation, class weights and label plots in newer versions) runs on it unchanged.
    """

    def __init__(
        self,
        table,
        data=None,
        exclude_zero=False,
        class_map=None,
        image_column_name=None,
        label_column_name=None,
        **kwargs,
    ):
        """Initialize the semantic segmentation dataset.

        :param table: The 3LC table containing the dataset
        :param data: The dataset dict, with the column's background and void ids from `check_tlc_dataset`
        :param exclude_zero: Whether to exclude zero-weight samples
        :param class_map: Mapping from 3LC class ids to training class indices
        :param image_column_name: Name of the image column in the table
        :param label_column_name: Name of the semantic segmentation column in the table
        """
        self._annotation_column = label_column_name.split(".")[0]
        self._void = data.get("semantic_void")

        # Pixels no layer covers are the background when the column declares one. Otherwise 3LC reads them as id 0,
        # which is either a class of its own or leaves them unlabeled, so they are ignored.
        background = data.get("semantic_background")
        fill_id = background if background is not None else 0
        self._fill = (class_map or {}).get(fill_id, IGNORE_INDEX)

        super().__init__(
            table,
            data=data,
            task="semantic",
            exclude_zero=exclude_zero,
            class_map=class_map,
            image_column_name=image_column_name,
            label_column_name=label_column_name,
            **kwargs,
        )

    def _get_label_from_row(self, im_file: str, row: Any, example_id: int) -> dict[str, Any]:
        """Get the semantic segmentation label for a row, keeping the segmentation in its compact row form."""
        raw = row[self._annotation_column]
        height, width = self._resolve_image_dimensions(im_file, raw[IMAGE_HEIGHT], raw[IMAGE_WIDTH])

        layer_ids = (raw.get(INSTANCE_PROPERTIES) or {}).get(LABEL) or []
        values = np.array(
            [
                IGNORE_INDEX
                if layer_id == self._void
                else map_label(self._class_map, layer_id, self.table, self._label_column_name, "semantic", example_id)
                for layer_id in layer_ids
            ],
            dtype=np.uint8,
        )

        return {
            "im_file": im_file,
            "shape": (int(height), int(width)),
            "cls": np.zeros((0, 1), dtype=np.float32),
            "bboxes": np.zeros((0, 4), dtype=np.float32),
            "segments": [],
            "keypoints": None,
            "normalized": True,
            "bbox_format": "xywh",
            "example_id": example_id,
            SEMANTIC_SHAPE: (int(height), int(width)),
            SEMANTIC_FILL: self._fill,
            SEMANTIC_VALUES: values,
            SEMANTIC_RLES: tuple(raw[RLES]),
        }

    def load_mask(self, index: int, image_shape: tuple[int, int] | None = None) -> np.ndarray:
        """Decode image `index`'s segmentation into a `(H, W)` uint8 map of training class indices at its original
        resolution, with void pixels at `IGNORE_INDEX`.

        Overrides `SemanticDataset.load_mask`, which reads a mask PNG. `image_shape` is unused, as it is there: the mask
        is returned at its own resolution and resized by the caller.
        """
        label = self.labels[index]
        height, width = label[SEMANTIC_SHAPE]
        return decode_semantic_mask(height, width, label[SEMANTIC_FILL], label[SEMANTIC_VALUES], label[SEMANTIC_RLES])

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """Drop the compact segmentation from the per-sample label, which only `load_mask` reads, from `self.labels`."""
        for key in (SEMANTIC_SHAPE, SEMANTIC_FILL, SEMANTIC_VALUES, SEMANTIC_RLES):
            label.pop(key, None)
        return super().update_labels_info(label)

    def get_image_and_label(self, index):
        """Get the image and label, keeping the load-time resized shape the validator inverts the letterbox with.

        Only without augmentation, where the transforms are `LetterBox` and `SemanticFormat` alone, so every sample of a
        batch carries the key.
        """
        label = super().get_image_and_label(index)
        if not self.augment:
            label[RESIZED_SHAPE] = label["resized_shape"]
        return label
