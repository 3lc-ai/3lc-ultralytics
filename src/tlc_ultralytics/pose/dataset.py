from __future__ import annotations

from typing import Any

import numpy as np
from tlc.core.builtins.constants import IMAGE, KEYPOINTS_2D
from tlc.core.data_formats import Keypoints2DInstances

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

        # Desired fixed K from dataset config (default 17) for empty-case shapes
        kpt_shape = self.data.get("kpt_shape")
        instances = Keypoints2DInstances.from_row(label_column_value)

        # Image dimensions and raw arrays
        H = float(instances.image_height)
        W = float(instances.image_width)
        labels = instances.instance_labels.astype(np.int32, copy=False)  # (N,)
        bboxes_xyxy = instances.instance_bbs.astype(np.float32, copy=False)  # (N,4) [x_min, y_min, x_max, y_max]
        kxy = instances.keypoints.astype(np.float32, copy=False)  # (N,K,2)
        vis = instances.keypoint_visibilities  # (N,K) or None

        # Normalize keypoints to [0,1]
        kxy[..., 0] /= W
        kxy[..., 1] /= H

        # Vectorized: xyxy (abs) -> center-xywh (normalized)
        xy_min = bboxes_xyxy[:, 0:2]
        xy_max = bboxes_xyxy[:, 2:4]
        wh_abs = xy_max - xy_min
        ctr_abs = xy_min + 0.5 * wh_abs
        scale = np.array([W, H], dtype=np.float32)
        bboxes_arr = np.concatenate([(ctr_abs / scale), (wh_abs / scale)], axis=1).astype(np.float32)

        # Handle empty vs non-empty
        if labels.size == 0:
            K_cfg = int(kpt_shape[0]) if kpt_shape else (int(kxy.shape[1]) if kxy.shape[1] > 0 else 17)
            cls_arr = np.zeros((0, 1), dtype=np.float32)
            bboxes_arr = np.zeros((0, 4), dtype=np.float32)
            kp_stack = np.zeros((0, K_cfg, 3), dtype=np.float32)
        else:
            mapped_list = [self._class_map.get(int(v), int(v)) for v in labels.tolist()]
            cls_arr = np.asarray(mapped_list, dtype=np.float32).reshape(-1, 1)

            # Build (N,K,3) with visibilities
            if vis is None:
                vis_arr = np.ones((kxy.shape[0], kxy.shape[1], 1), dtype=np.float32)
            else:
                vis_arr = vis.astype(np.float32, copy=False).reshape(kxy.shape[0], kxy.shape[1], 1)
            kp_stack = np.concatenate([kxy, vis_arr], axis=2)

        return {
            "im_file": im_file,
            "shape": (round(instances.image_height), round(instances.image_width)),
            "cls": cls_arr,
            "bboxes": bboxes_arr,
            "segments": [],
            "keypoints": kp_stack,  # (N, K, 3)
            "normalized": True,
            "bbox_format": "xywh",
            "example_id": example_id,
        }
