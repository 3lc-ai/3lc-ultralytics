from __future__ import annotations

import tlc
from pathlib import Path

from ultralytics.data.dataset import ClassificationDataset
from ultralytics.utils import colorstr, LOGGER
from ultralytics.data.dataset import classify_augmentations, classify_transforms

from tlc_ultralytics.engine.dataset import TLCDatasetMixin

from typing import Any


class TLCClassificationDataset(TLCDatasetMixin, ClassificationDataset):
    """
    Initialize 3LC classification dataset for use in YOLO classification.

    Args:
        table (tlc.Table): The 3LC table with classification data. Needs columns 'image' and 'label'.
        args (Namespace): See parent.
        augment (bool): See parent.
        prefix (str): See parent.

    """

    def __init__(
        self,
        table,
        args,
        augment=False,
        prefix="",
        image_column_name=tlc.IMAGE,
        label_column_name=tlc.LABEL,
        exclude_zero=False,
        class_map=None,
    ):
        # Populate self.samples with image paths and labels
        # Each is a tuple of (image_path, label)
        assert isinstance(table, tlc.Table)
        self.table = table
        self.root = table.url
        self.prefix = prefix
        self._image_column_name = image_column_name
        self._label_column_name = label_column_name
        self._exclude_zero = exclude_zero
        self._class_map = class_map
        self._example_ids = []

        self.verify_schema()

        im_files, labels = self._get_rows_from_table()

        self.samples = list(zip(im_files, labels))

        # Initialize attributes (e.g. transforms)
        self._init_attributes(args, augment, prefix)

        # Call mixin
        self._post_init()

    def verify_schema(self):
        """Verify that the provided Table has the desired entries"""

        # Check for data in columns
        assert len(self.table) > 0, f"Table {self.root.to_str()} has no rows."
        first_row = self.table.table_rows[0]
        assert isinstance(first_row[self._image_column_name], str), (
            f"First value in image column '{self._image_column_name}' in table {self.root.to_str()} is not a string."
        )
        assert isinstance(first_row[self._label_column_name], int), (
            f"First value in label column '{self._label_column_name}' in table {self.root.to_str()} is not an integer."
        )

    def verify_images(self):
        """Called by parent init_attributes, but this is handled by the 3LC mixin."""
        return self.samples

    def _get_label_from_row(self, im_file: str, row: Any, example_id: int) -> Any:
        label = row[self._label_column_name]

        if self._class_map:
            label = self._class_map[label]

        self._example_ids.append(example_id)

        return label

    def _index_to_example_id(self, index: int) -> int:
        """Get the example id for the given index."""
        return self._example_ids[index]

    def _init_attributes(self, args, augment, prefix):
        """Copied from ultralytics.data.dataset.ClassificationDataset.__init__."""

        # Initialize attributes
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[: round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f"{prefix}: ") if prefix else ""
        self.cache_ram = args.cache is True or str(args.cache).lower() == "ram"  # cache images into RAM
        if self.cache_ram:
            LOGGER.warning(
                "Classification `cache_ram` training has known memory leak in "
                "https://github.com/ultralytics/ultralytics/issues/9824, setting `cache_ram=False`."
            )
            self.cache_ram = False
        self.cache_disk = str(args.cache).lower() == "disk"  # cache images on hard drive as uncompressed *.npy files
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im
        scale = (1.0 - args.scale, 1.0)  # (0.08, 1.0)
        self.torch_transforms = (
            classify_augmentations(
                size=args.imgsz,
                scale=scale,
                hflip=args.fliplr,
                vflip=args.flipud,
                erasing=args.erasing,
                auto_augment=args.auto_augment,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
            )
            if augment
            else classify_transforms(size=args.imgsz)
        )
