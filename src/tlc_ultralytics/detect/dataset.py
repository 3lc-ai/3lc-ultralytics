from __future__ import annotations

from typing import Any

import numpy as np
from tlc.constants import IMAGE_HEIGHT, IMAGE_WIDTH
from tlc.data_types import BoundingBoxes2D, SegmentationPolygons
from tlc.helpers import AnnotationHelper, AnnotationType
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_file_speeds, segments2boxes
from ultralytics.utils import LOGGER, colorstr

from tlc_ultralytics.engine.dataset import TLCDatasetMixin
from tlc_ultralytics.utils.dataset import IdentityDict, map_label


class TLCYOLODataset:
    """Factory class for creating task-specific 3LC YOLO datasets."""

    def __new__(
        cls,
        table,
        data=None,
        task="detect",
        exclude_zero=False,
        class_map=None,
        image_column_name=None,
        label_column_name=None,
        **kwargs,
    ):
        """Create a new dataset instance of the appropriate type.

        :param table: The 3LC table containing the dataset
        :param data: Optional data parameter for YOLODataset
        :param task: Either "segment" or "detect"
        :param exclude_zero: Whether to exclude zero-class annotations
        :param class_map: Optional mapping from original class indices to new ones
        :param image_column_name: Name of the image column in the table
        :param label_column_name: Name of the label column in the table
        :param **kwargs: Additional arguments passed to the dataset constructor
        """
        from tlc_ultralytics.obb.dataset import TLCOBBDataset
        from tlc_ultralytics.pose.dataset import TLCYOLOPoseDataset

        if task == "detect":
            return TLCYOLODetectionDataset(
                table=table,
                data=data,
                exclude_zero=exclude_zero,
                class_map=class_map,
                image_column_name=image_column_name,
                label_column_name=label_column_name,
                **kwargs,
            )
        elif task == "segment":
            return TLCYOLOSegmentationDataset(
                table=table,
                data=data,
                exclude_zero=exclude_zero,
                class_map=class_map,
                image_column_name=image_column_name,
                label_column_name=label_column_name,
                **kwargs,
            )
        elif task == "pose":
            return TLCYOLOPoseDataset(
                table=table,
                data=data,
                exclude_zero=exclude_zero,
                class_map=class_map,
                image_column_name=image_column_name,
                label_column_name=label_column_name,
                task="pose",
                **kwargs,
            )
        elif task == "obb":
            return TLCOBBDataset(
                table=table,
                data=data,
                exclude_zero=exclude_zero,
                class_map=class_map,
                image_column_name=image_column_name,
                label_column_name=label_column_name,
                task="obb",
                **kwargs,
            )
        else:
            msg = (
                f"Unsupported task: {task} for TLCYOLODataset. "
                "Only 'segment', 'detect', 'pose', and 'obb' are supported."
            )
            raise ValueError(msg)


class BaseTLCYOLODataset(TLCDatasetMixin, YOLODataset):
    """Base class for 3LC YOLO datasets.

    This class provides common functionality for any detection task.
    Task-specific functionality should be implemented in subclasses.
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
        """Initialize the base dataset.

        :param table: The 3LC table containing the dataset
        :param data: Optional data parameter for YOLODataset
        :param exclude_zero: Whether to exclude zero-class annotations
        :param class_map: Optional mapping from original class indices to new ones
        :param image_column_name: Name of the image column in the table
        :param label_column_name: Name of the label column in the table
        """
        self.table = table
        self._exclude_zero = exclude_zero
        self._class_map = class_map if class_map is not None else IdentityDict()
        self._image_column_name = image_column_name
        self._label_column_name = label_column_name

        super().__init__(table, data=data, **kwargs)
        self._post_init()

    def get_img_files(self, _):
        """Images are read in `get_labels` to avoid two loops, return empty list here."""
        im_files, labels = self._get_rows_from_table()
        check_file_speeds(im_files, prefix=colorstr(self.prefix + ":") + " ")
        self.labels = labels
        self.im_files = im_files
        return self.im_files

    def get_labels(self):
        """Get the labels from the table."""
        return self.labels

    def _index_to_example_id(self, index: int) -> int:
        """Get the example id for the given index."""
        return self.labels[index]["example_id"]

    def _get_label_from_row(self, im_file: str, row: Any, example_id: int) -> dict[str, Any]:
        """Get the label for a row in the appropriate format.

        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _get_label_from_row")


class TLCYOLODetectionDataset(BaseTLCYOLODataset):
    """3LC YOLO dataset for object detection."""

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
        """Initialize the detection dataset.

        :param table: The 3LC table containing the dataset
        :param data: Optional data parameter for YOLODataset
        :param exclude_zero: Whether to exclude zero-class annotations
        :param class_map: Optional mapping from original class indices to new ones
        :param image_column_name: Name of the image column in the table
        :param label_column_name: Name of the label column in the table
        """
        # Determine the annotation column name and whether this is a legacy-format table
        column_name = label_column_name.split(".")[0]
        ann_col = AnnotationHelper.get(table, column_name)
        self._annotation_column = column_name
        self._is_legacy_bb = ann_col.type is AnnotationType.LEGACY_BOUNDING_BOXES
        self._bb_schema = table.rows_schema.values[column_name] if self._is_legacy_bb else None

        super().__init__(
            table,
            data=data,
            task="detect",
            exclude_zero=exclude_zero,
            class_map=class_map,
            image_column_name=image_column_name,
            label_column_name=label_column_name,
            **kwargs,
        )

    def _get_label_from_row(self, im_file: str, row: Any, example_id: int) -> dict[str, Any]:
        """Get the detection label for a row using BoundingBoxes2D."""
        raw = row[self._annotation_column]

        # Get BoundingBoxes2D — handles both legacy and new format
        if self._is_legacy_bb:
            bb2d = BoundingBoxes2D.from_legacy_row(raw, self._bb_schema)
        else:
            bb2d = BoundingBoxes2D.from_row(raw)

        # The coordinate-space bounds carry the image dimensions (boxes are stored as absolute
        # xyxy pixels). Fall back to the real image size if they are missing or non-positive.
        height = bb2d.y_max - (bb2d.y_min or 0) if bb2d.y_max else 0
        width = bb2d.x_max - (bb2d.x_min or 0) if bb2d.x_max else 0
        height, width = self._resolve_image_dimensions(im_file, height, width)

        if bb2d.num_instances == 0 or bb2d.labels is None:
            return {
                "im_file": im_file,
                "shape": (height, width),
                "cls": np.zeros((0, 1), dtype=np.float32),
                "bboxes": np.zeros((0, 4), dtype=np.float32),
                "segments": [],
                "keypoints": None,
                "normalized": True,
                "bbox_format": "xywh",
                "example_id": example_id,
            }

        # Normalize to [0,1] and convert to centered XYWH (what YOLO expects)
        cxywh = bb2d.bounding_boxes_cxywh / np.array([width, height, width, height], dtype=np.float32)

        # Filter boxes with non-positive width or height and apply class map
        widths = cxywh[:, 2]
        heights = cxywh[:, 3]
        valid = (widths > 0) & (heights > 0)

        valid_boxes = cxywh[valid]
        valid_labels = bb2d.labels[valid]
        classes = np.array(
            [
                map_label(self._class_map, lbl, self.table, self._label_column_name, "detect", example_id)
                for lbl in valid_labels
            ],
            dtype=np.float32,
        ).reshape(-1, 1)

        return {
            "im_file": im_file,
            "shape": (height, width),
            "cls": classes,
            "bboxes": valid_boxes,
            "segments": [],
            "keypoints": None,
            "normalized": True,
            "bbox_format": "xywh",
            "example_id": example_id,
        }


class TLCYOLOSegmentationDataset(BaseTLCYOLODataset):
    """3LC YOLO dataset for instance segmentation."""

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
        """Initialize the segmentation dataset.

        :param table: The 3LC table containing the dataset
        :param data: Optional data parameter for YOLODataset
        :param exclude_zero: Whether to exclude zero-class annotations
        :param class_map: Optional mapping from original class indices to new ones
        :param image_column_name: Name of the image column in the table
        :param label_column_name: Name of the label column in the table
        """
        super().__init__(
            table,
            data=data,
            task="segment",
            exclude_zero=exclude_zero,
            class_map=class_map,
            image_column_name=image_column_name,
            label_column_name=label_column_name,
            **kwargs,
        )

    def _get_label_from_row(self, im_file: str, row: Any, example_id: int) -> dict[str, Any]:
        """Get the segmentation label for a row.

        Polygons are always requested in relative ([0, 1]) coordinates. The dataclass
        helper ``.to_relative()`` does the conversion from the absolute pixel coords
        produced by ``SegmentationPolygons.from_row``, regardless of how the source
        table was authored. YOLO consumes relative segments directly, so no further
        normalization is performed here.
        """
        column_name, _, _ = self._label_column_name.split(".")

        # The row view holds the raw, serialized segmentation dict.
        row = self.table.table_rows[example_id]
        raw_segmentations = row[column_name]

        # Masks are RLE-encoded and the size is reconstructed solely from the stored dimensions,
        # so they must be valid before decoding - inject the resolved dimensions into the row.
        height, width = self._resolve_image_dimensions(
            im_file, raw_segmentations[IMAGE_HEIGHT], raw_segmentations[IMAGE_WIDTH]
        )
        raw_segmentations = {**raw_segmentations, IMAGE_HEIGHT: height, IMAGE_WIDTH: width}

        segmentations = SegmentationPolygons.from_row(raw_segmentations).to_relative()
        height, width = segmentations.image_height, segmentations.image_width
        classes = []
        segments = []

        # Unlabeled rows come back with `labels=None`
        labels = segmentations.labels if segmentations.labels is not None else []

        for i, (category, polygon) in enumerate(
            zip(
                labels,
                segmentations.polygons,
                strict=False,
            )
        ):
            # Handle polygons with zero area
            if len(polygon) < 6:
                LOGGER.warning(f"Polygon {i} in row {example_id} has fewer than 3 points and will be ignored.")
                continue

            classes.append(
                map_label(self._class_map, category, self.table, self._label_column_name, "segment", example_id)
            )
            row_segments = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            segments.append(row_segments)

        # Sanity check: ``.to_relative()`` is contracted to return coordinates in [0, 1].
        # If it doesn't, training silently produces zero gradients — fail loudly with
        # an actionable message instead.
        if segments:
            max_coord = float(max(np.max(s) for s in segments))
            if max_coord > 1.0 + 1e-3:
                raise ValueError(
                    f"Segmentation polygons for example_id={example_id} have coordinates outside [0, 1] "
                    f"(max={max_coord:.4f}). SegmentationPolygons.from_row(...).to_relative() should "
                    f"return normalized polygons; this indicates a 3LC / segmentation-schema mismatch. "
                    f"Check the 3LC version or the table's segmentation column schema."
                )
            bboxes = segments2boxes(segments)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)

        return {
            "im_file": im_file,
            "shape": (height, width),  # format: (height, width)
            "cls": np.array(classes).astype(np.float32).reshape(-1, 1),
            "bboxes": bboxes,
            "segments": segments,
            "keypoints": None,
            "normalized": True,
            "bbox_format": "xywh",
            "example_id": example_id,
        }
