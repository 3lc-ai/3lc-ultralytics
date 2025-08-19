from __future__ import annotations

from typing import Any

import numpy as np
import tlc
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import colorstr

from tlc_ultralytics.engine.dataset import TLCDatasetMixin


class TLCYOLOPoseDataset(TLCDatasetMixin, YOLODataset):
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
        self.table = table
        self._exclude_zero = exclude_zero
        self._class_map = class_map or {}
        self._image_column_name = image_column_name or tlc.IMAGE
        self._label_column_name = label_column_name or "pose"

        super().__init__(table, data=data, **kwargs)
        self._post_init()

    def get_img_files(self, _):
        im_files, labels = self._get_rows_from_table()
        self.labels = labels
        self.im_files = im_files
        return self.im_files

    def get_labels(self):
        return self.labels

    def _index_to_example_id(self, index: int) -> int:
        return self.labels[index]["example_id"]

    def _get_label_from_row(self, im_file: str, row: Any, example_id: int) -> dict[str, Any]:
        pose_root = self._label_column_name.split(".")[0]
        pose = row.get(pose_root, {}) or {}

        height = int(pose.get(tlc.IMAGE_HEIGHT, row.get(tlc.IMAGE_HEIGHT, 0)))
        width = int(pose.get(tlc.IMAGE_WIDTH, row.get(tlc.IMAGE_WIDTH, 0)))
        if height == 0 or width == 0:
            _ = f"{colorstr(self.prefix + ':')} Missing image bounds in pose data; defaulting to zeros."
            height, width = 0, 0

        instances = pose.get("instances") or []
        classes, boxes, keypoints = [], [], []

        for inst in instances:
            label_val = inst.get(tlc.LABEL)
            if label_val is None:
                label_val = 0
            mapped = self._class_map.get(label_val, label_val)
            classes.append(mapped)

            bbox = inst.get("bbox", [0.0, 0.0, 0.0, 0.0])
            boxes.append(bbox)

            xys = inst.get("xys", [])
            if isinstance(xys, list):
                xys_arr = (
                    np.array(xys, dtype=np.float32).reshape(-1, 2) if len(xys) else np.zeros((0, 2), dtype=np.float32)
                )
            else:
                xys_arr = np.zeros((0, 2), dtype=np.float32)

            add = inst.get("xys_additional_data") or {}
            if "conf" in add:
                conf = np.array(add["conf"], dtype=np.float32).reshape(-1, 1)
                kpts = np.concatenate([xys_arr, conf], axis=1)
            else:
                kpts = xys_arr

            keypoints.append(kpts)

        cls_arr = np.array(classes, dtype=np.float32).reshape(-1, 1)
        bboxes_arr = (
            np.array(boxes, dtype=np.float32).reshape(-1, 4) if len(boxes) else np.zeros((0, 4), dtype=np.float32)
        )

        if len(keypoints):
            max_k = max(k.shape[0] for k in keypoints)
            dim = keypoints[0].shape[1] if keypoints[0].ndim == 2 and keypoints[0].shape[0] > 0 else 2
            kp_stack = np.zeros((len(keypoints), max_k, dim), dtype=np.float32)
            for i, k in enumerate(keypoints):
                kp_stack[i, : k.shape[0], : k.shape[1]] = k
        else:
            kp_stack = np.zeros((0, 0, 2), dtype=np.float32)

        return {
            "im_file": im_file,
            "shape": (height, width),
            "cls": cls_arr,
            "bboxes": bboxes_arr,
            "segments": [],
            "keypoints": kp_stack,  # (N, K, D) D in {2,3}
            "normalized": True,
            "bbox_format": "xywh",
            "example_id": example_id,
        }
