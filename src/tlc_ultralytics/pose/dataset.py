from __future__ import annotations

from typing import Any

import numpy as np
from tlc.core.builtins.constants import IMAGE, KEYPOINTS_2D
from tlc.client.data_format import Keypoints2DInstances

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

        # Parse using the shared dataclass, export normalized arrays
        instances = Keypoints2DInstances.from_row(label_column_value)
        arrays = instances.as_numpy(normalized=True, bbox_format="xywh")

        labels = arrays.get("labels")
        bboxes_xywh = arrays.get("bboxes")  # top-left x,y,w,h normalized
        kxy = arrays.get("keypoints")  # (N,K,2) normalized
        vis = arrays.get("visibilities")

        if labels is None or bboxes_xywh is None or kxy is None:
            # Fallback empty outputs
            cls_arr = np.zeros((0, 1), dtype=np.float32)
            bboxes_arr = np.zeros((0, 4), dtype=np.float32)
            kp_stack = np.zeros((0, kpt_shape[0], 3), dtype=np.float32)
        else:
            # Map labels
            mapped_labels = np.vectorize(lambda v: self._class_map.get(int(v), int(v)))(labels.astype(np.int32))
            cls_arr = mapped_labels.astype(np.float32).reshape(-1, 1)

            # Convert xywh (top-left) -> xywh (center)
            # bboxes_xywh is normalized already
            cx = bboxes_xywh[:, 0] + bboxes_xywh[:, 2] / 2.0
            cy = bboxes_xywh[:, 1] + bboxes_xywh[:, 3] / 2.0
            bboxes_arr = np.stack([cx, cy, bboxes_xywh[:, 2], bboxes_xywh[:, 3]], axis=1).astype(np.float32)

            # Build (N,K,3) with visibilities
            if vis is None:
                vis_arr = np.ones((kxy.shape[0], kxy.shape[1], 1), dtype=np.float32)
            else:
                vis_arr = vis.astype(np.float32, copy=False).reshape(kxy.shape[0], kxy.shape[1], 1)
            kp_stack = np.concatenate([kxy.astype(np.float32, copy=False), vis_arr], axis=2)

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
