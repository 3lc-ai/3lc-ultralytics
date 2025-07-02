from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
import tlc
from tlc.core.builtins.types.bounding_box import CenteredXYWHBoundingBox
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_file_speeds, segments2boxes
from ultralytics.utils import LOGGER, colorstr

from tlc_ultralytics.engine.dataset import TLCDatasetMixin

SegmentType = Literal["absolute", "relative"]


class IdentityDict(dict):
    def __missing__(self, key):
        return key


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
        else:
            raise ValueError(f"Unsupported task: {task} for TLCYOLODataset. Only 'segment' and 'detect' are supported.")


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
        self._detection_factory: Callable[[list[float]], tlc.BoundingBox] = self._get_detection_factory(
            table, label_column_name
        )

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

    def _get_detection_factory(
        self, table: tlc.Table, label_column_name: str
    ) -> Callable[[list[float]], tlc.BoundingBox]:
        """Infer the bounding box factory from the table schema.

        :param table: The 3LC table containing the dataset
        :param label_column_name: The name of the label column in the table
        :returns: A factory function that creates bounding boxes from coordinates
        """
        column_name, instances_name, _ = label_column_name.split(".")

        try:
            factory = tlc.BoundingBox.from_schema(table.rows_schema.values[column_name].values[instances_name])
        except Exception as e:
            raise ValueError(f"Table {table.url} is not a detection table: {e}") from None

        return factory

    def _get_label_from_row(self, im_file: str, row: Any, example_id: int) -> dict[str, Any]:
        """Get the detection label for a row."""
        return tlc_table_row_to_yolo_label(
            row,
            self._detection_factory,
            self._class_map,
            im_file,
            label_column_name=self._label_column_name,
            example_id=example_id,
        )


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
        self._segment_type: SegmentType = self._get_segment_type(table, label_column_name)
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

    def _get_segment_type(self, table: tlc.Table, label_column_name: str) -> SegmentType:
        """Verify the table format and check if the polygons are relative.

        :param table: The 3LC table containing the dataset
        :param label_column_name: The name of the label column in the table
        :returns: The segment type ("absolute" or "relative")
        """
        column_name, instances_name, _ = label_column_name.split(".")

        try:
            rles_schema_value = table.rows_schema.values[column_name].values["rles"].value
            segment_type = "relative" if getattr(rles_schema_value, "polygons_are_relative", False) else "absolute"
        except Exception as e:
            raise ValueError(f"Table {table.url} is not a segmentation table: {e}") from None

        return segment_type

    def _normalize_segments(self, segments: list[np.ndarray], width: int, height: int) -> list[np.ndarray]:
        """Normalize segments to relative coordinates if they are absolute.

        :param segments: List of segment coordinates
        :param width: Image width
        :param height: Image height
        :return: Normalized segments
        """
        if self._segment_type == "absolute":
            return segments / np.array([width, height])
        return segments

    def _get_label_from_row(self, im_file: str, row: Any, example_id: int) -> dict[str, Any]:
        """Get the segmentation label for a row."""
        column_name, instances_name, label_key = self._label_column_name.split(".")

        # Use sample view to get polygons, row is row view
        sample = self.table[example_id]

        segmentations = sample[column_name]
        height, width = segmentations[tlc.IMAGE_HEIGHT], segmentations[tlc.IMAGE_WIDTH]
        classes = []
        segments = []

        for i, (category, polygon) in enumerate(
            zip(
                segmentations[instances_name][label_key],
                segmentations[tlc.POLYGONS],
            )
        ):
            # Handle polygons with zero area
            if len(polygon) < 6:
                LOGGER.warning(f"Polygon {i} in row {example_id} has fewer than 6 points and will be ignored.")
                continue

            classes.append(self._class_map[category])
            row_segments = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            segments.append(self._normalize_segments(row_segments, width, height))

        # Compute bounding boxes from segments
        if segments:
            bboxes = segments2boxes(segments)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)

        return {
            "im_file": im_file,
            "shape": (height, width),  # format: (height, width)
            "cls": np.array(classes).reshape(-1, 1),
            "bboxes": bboxes,
            "segments": segments,
            "keypoints": None,
            "normalized": True,
            "bbox_format": "xywh",
            "example_id": example_id,
        }


def convert_to_xywh(bbox: tlc.BoundingBox, image_width: int, image_height: int) -> CenteredXYWHBoundingBox:
    if isinstance(bbox, CenteredXYWHBoundingBox):
        return bbox
    else:
        return CenteredXYWHBoundingBox.from_top_left_xywh(bbox.to_top_left_xywh().normalize(image_width, image_height))


def unpack_box(
    bbox: dict[str, int | float],
    table_format: Callable[[list[float]], tlc.BoundingBox],
    image_width: int,
    image_height: int,
    label_key: str,
) -> tuple[int, list[float]]:
    coordinates = [bbox[tlc.X0], bbox[tlc.Y0], bbox[tlc.X1], bbox[tlc.Y1]]
    return bbox[label_key], convert_to_xywh(table_format(coordinates), image_width, image_height)


def unpack_boxes(
    bboxes: list[dict[str, int | float]],
    class_map: dict[int, int],
    table_format: Callable[[list[float]], tlc.BoundingBox],
    image_width: int,
    image_height: int,
    label_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    classes_list, boxes_list = [], []
    for bbox in bboxes:
        _class, box = unpack_box(bbox, table_format, image_width, image_height, label_key)

        # Ignore boxes with non-positive width or height
        if box[2] > 0 and box[3] > 0:
            classes_list.append(class_map[_class])
            boxes_list.append(box)

    # Convert to np array
    boxes = np.array(boxes_list, ndmin=2, dtype=np.float32)
    if len(boxes_list) == 0:
        boxes = boxes.reshape(0, 4)

    classes = np.array(classes_list, dtype=np.float32).reshape((-1, 1))
    assert classes.shape == (boxes.shape[0], 1)
    return classes, boxes


def tlc_table_row_to_yolo_label(
    row,
    detection_factory: Callable[[list[float]], tlc.BoundingBox],
    class_map: dict[int, int],
    im_file: str,
    label_column_name: str,
    example_id: int,
) -> dict[str, Any]:
    """Convert a table row from a 3lc Table to a Ultralytics YOLO label dict.

    :param row: The table row to convert
    :param detection_factory: Factory function to create bounding boxes
    :param class_map: A dictionary mapping 3lc class labels to contiguous class labels
    :param im_file: The path to the image file of the row
    :param label_column_name: The name of the label column in the table
    :returns: A dictionary containing the Ultralytics YOLO label information
    """
    bounding_boxes_column_key, bounding_boxes_list_key, label_key = label_column_name.split(".")

    classes, bboxes = unpack_boxes(
        row[bounding_boxes_column_key][bounding_boxes_list_key],
        class_map,
        detection_factory,
        row[bounding_boxes_column_key][tlc.IMAGE_WIDTH],
        row[bounding_boxes_column_key][tlc.IMAGE_HEIGHT],
        label_key,
    )

    return {
        "im_file": im_file,
        "shape": (
            row[bounding_boxes_column_key][tlc.IMAGE_HEIGHT],
            row[bounding_boxes_column_key][tlc.IMAGE_WIDTH],
        ),  # format: (height, width)
        "cls": classes,
        "bboxes": bboxes,
        "segments": [],
        "keypoints": None,
        "normalized": True,
        "bbox_format": "xywh",
        "example_id": example_id,
    }
