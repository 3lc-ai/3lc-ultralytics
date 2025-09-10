from __future__ import annotations

from typing import Any

import numpy as np
from tlc.core import GeometryHelper
from tlc.core.builtins.constants import (
    BBS_2D,
    IMAGE,
    INSTANCES,
    KEYPOINTS_2D,
    LABEL,
    LINES,
    VERTICES_2D,
    VERTICES_2D_ADDITIONAL_DATA,
    VISIBILITIES,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
)

from tlc_ultralytics.detect.dataset import BaseTLCYOLODataset


class TLCYOLOPoseDataset(BaseTLCYOLODataset):
    """3LC YOLO dataset for pose (keypoints) models.

    Builds YOLO-compatible per-image labels dict with keys: im_file, shape, cls, bboxes, keypoints.
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
            image_column_name=image_column_name or IMAGE,
            label_column_name=label_column_name or KEYPOINTS_2D,
            **kwargs,
        )
        self._post_init()

    def _get_label_from_row(self, im_file: str, row: Any, example_id: int) -> dict[str, Any]:
        pose_root = self._label_column_name.split(".")[0]

        label_column_value = row[pose_root]

        x_min = label_column_value[X_MIN]
        y_min = label_column_value[Y_MIN]
        x_max = label_column_value[X_MAX]
        y_max = label_column_value[Y_MAX]

        image_width = x_max - x_min
        image_height = y_max - y_min

        # Desired fixed K from dataset config (default 17)
        kpt_shape = self.data.get("kpt_shape")

        instances = label_column_value[INSTANCES]

        classes_list: list[int] = []
        keypoints_list: list[np.ndarray] = []
        bb_list = []

        for instance in instances:
            # Class (dummy 0 if missing)
            label_val = instance.get(LABEL, 0)
            mapped = self._class_map.get(label_val, label_val)
            classes_list.append(int(mapped))

            # Bounding boxes
            bb = instance[BBS_2D][0]  # Only one bounding box per instance
            bb_width = bb[X_MAX] - bb[X_MIN]
            bb_height = bb[Y_MAX] - bb[Y_MIN]
            bb_xywhn = [bb[X_MIN] + bb_width / 2, bb[Y_MIN] + bb_height / 2, bb_width, bb_height]
            bb_xywhn = [
                bb_xywhn[0] / image_width,
                bb_xywhn[1] / image_height,
                bb_xywhn[2] / image_width,
                bb_xywhn[3] / image_height,
            ]
            bb_list.append(bb_xywhn)

            # Keypoints xys
            xys = instance[VERTICES_2D]
            if len(xys) >= 2:
                xys_arr = np.array(xys, dtype=np.float32).reshape(-1, 2)
            else:
                xys_arr = np.zeros((0, 2), dtype=np.float32)

            if xys_arr.size:
                norm_xy = np.empty_like(xys_arr)
                norm_xy[:, 0] = xys_arr[:, 0] / image_width
                norm_xy[:, 1] = xys_arr[:, 1] / image_height
                norm_xy = np.clip(norm_xy, 0.0, 1.0)
            else:
                norm_xy = xys_arr

            # Visibilities
            if VERTICES_2D_ADDITIONAL_DATA in instance:
                add = instance[VERTICES_2D_ADDITIONAL_DATA]
                vis = np.array(add[VISIBILITIES], dtype=np.float32).reshape(-1, 1)
            else:
                vis = np.ones((xys_arr.shape[0], 1), dtype=np.float32)

            kp = np.concatenate([norm_xy, vis], axis=1)

            keypoints_list.append(kp)

        # Convert to arrays with expected shapes
        cls_arr = np.array(classes_list, dtype=np.float32).reshape(-1, 1)
        bboxes_arr = np.array(bb_list, dtype=np.float32).reshape(-1, 4)

        # Stack to (N, K, 3)
        if keypoints_list:
            kp_stack = np.stack(keypoints_list, axis=0)
        else:
            kp_stack = np.zeros((0, kpt_shape[0], 3), dtype=np.float32)

        return {
            "im_file": im_file,
            "shape": (round(image_height), round(image_width)),
            "cls": cls_arr,
            "bboxes": bboxes_arr,
            "segments": [],
            "keypoints": kp_stack,  # (N, K, 3)
            "normalized": True,
            "bbox_format": "xywh",
            "example_id": example_id,
        }
