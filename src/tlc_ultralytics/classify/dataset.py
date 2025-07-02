from __future__ import annotations

from functools import partial
from typing import Any

import tlc
from ultralytics.data.dataset import ClassificationDataset

from tlc_ultralytics.engine.dataset import TLCDatasetMixin


class _DummyImageFolder:
    def __init__(self, root: tlc.Table, allow_empty: bool = False, samples: list[tuple[str, int]] | None = None):
        self._root = root
        self._samples = samples

    @property
    def samples(self):
        return self._samples

    @property
    def root(self):
        return self._root.url


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

        # Override torchvision ImageFolder when called by parent __init__
        import torchvision

        OriginalImageFolder = torchvision.datasets.ImageFolder
        torchvision.datasets.ImageFolder = partial(_DummyImageFolder, samples=self.samples)

        ClassificationDataset.__init__(self, table, args, augment=augment, prefix=prefix)

        # Restore torchvision ImageFolder
        torchvision.datasets.ImageFolder = OriginalImageFolder

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
