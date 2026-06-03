from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from tlc.data_types import OrientedBoundingBoxes2D

from tlc_ultralytics.constants import IMAGE_COLUMN_NAME, OBB_LABEL_COLUMN_NAME
from tlc_ultralytics.detect.dataset import BaseTLCYOLODataset


def xcyxwhr_to_corner_points(xc, yc, w, h, r):
    # Ensure r is in the first quadrant [0, pi/2] by adding or subtracting pi/2 as needed
    if r < 0 or r > np.pi / 2:
        # Bring r into [0, pi/2] by adding or subtracting multiples of pi/2
        r = r % np.pi
        if r > np.pi / 2:
            r -= np.pi / 2
        elif r < 0:
            r += np.pi / 2
    assert 0 <= r <= np.pi / 2, f"Rotation r (in radians) must be in the first quadrant [0, pi/2]. Got {r}"
    dx, dy = w / 2.0, h / 2.0
    cos_r, sin_r = np.cos(r), np.sin(r)

    # local corners (relative to center): (±dx, ±dy)
    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]], dtype=np.float64)

    R = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float64)

    pts = corners @ R.T
    pts[:, 0] += xc
    pts[:, 1] += yc
    return pts


def corner_points_to_xywh(corner_points):
    pts = np.asarray(corner_points, dtype=np.float64).reshape(-1, 2)
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    w, h = xmax - xmin, ymax - ymin
    xc, yc = xmin + w / 2.0, ymin + h / 2.0
    return np.array([xc, yc, w, h], dtype=np.float32)


def regularize_rboxes(rboxes):
    """
    Regularize rotated bounding boxes to range [0, pi/2].

    Args:
        rboxes (np.array): Input rotated boxes with shape (N, 5) in xywhr format.

    Returns:
        (np.array): Regularized rotated boxes.
    """
    x, y, w, h, t = rboxes.tolist()
    # Swap edge if t >= pi/2 while not being symmetrically opposite
    swap = t % np.pi >= np.pi / 2
    w_ = np.where(swap, h, w)
    h_ = np.where(swap, w, h)
    t = t % (np.pi / 2)
    return np.array([x, y, w_, h_, t], dtype=np.float32).reshape(1, 5)


class TLCOBBDataset(BaseTLCYOLODataset):
    """3LC YOLO dataset for OBB (oriented bounding boxes) models.

    Builds YOLO-compatible per-image labels dict with keys: im_file, shape, cls, bboxes.
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
        super().__init__(
            table,
            data=data,
            exclude_zero=exclude_zero,
            class_map=class_map,
            image_column_name=image_column_name or IMAGE_COLUMN_NAME,
            label_column_name=label_column_name or OBB_LABEL_COLUMN_NAME,
            **kwargs,
        )
        self._post_init()

    def _get_label_from_row(self, im_file: str, row: Any, example_id: int) -> dict[str, Any]:
        label_root = self._label_column_name.split(".")[0]
        instances = OrientedBoundingBoxes2D.from_row(row[label_root])

        image_width = float(instances.x_max - (instances.x_min or 0))
        image_height = float(instances.y_max - (instances.y_min or 0))

        cls_arr = instances.labels.astype(np.float32).reshape(-1, 1)

        boxes = []
        segments = []
        # obbs has shape (N, 5): [center_x, center_y, size_x, size_y, rotation_radians]
        for cx, cy, sx, sy, r in instances.obbs:
            xc_n = cx / image_width
            yc_n = cy / image_height
            w_n = sx / image_width
            h_n = sy / image_height
            r_deg = r * 180 / np.pi  # cv2.boxPoints expects degrees
            corner_points = cv2.boxPoints(((xc_n, yc_n), (w_n, h_n), r_deg))
            segments.append(corner_points)
            boxes.append(corner_points_to_xywh(corner_points))

        bboxes_arr = np.array(boxes, ndmin=2, dtype=np.float32)
        if len(boxes) == 0:
            bboxes_arr = bboxes_arr.reshape(0, 4)

        return {
            "im_file": im_file,
            "shape": (round(image_height), round(image_width)),
            "cls": cls_arr,
            "bboxes": bboxes_arr,
            "keypoints": None,
            "segments": segments,
            "normalized": True,
            "bbox_format": "xywh",
            "example_id": example_id,
        }
