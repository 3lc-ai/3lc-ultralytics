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
        self._label_column_name = label_column_name or tlc.KEYPOINTS_2D

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

        # Read bounds (prefer explicit x_min/x_max/y_min/y_max)
        x_min = float(pose.get("x_min", 0.0))
        y_min = float(pose.get("y_min", 0.0))
        x_max = float(pose.get("x_max", 0.0))
        y_max = float(pose.get("y_max", 0.0))

        width = max(x_max - x_min, 1.0)
        height = max(y_max - y_min, 1.0)

        if width <= 0 or height <= 0:
            _ = f"{colorstr(self.prefix + ':')} Invalid image bounds in pose data; defaulting to 1x1."
            width, height = 1.0, 1.0

        # Desired fixed K from dataset config (default 17)
        desired_k = 17
        if isinstance(getattr(self, "data", None), dict):
            kps = self.data.get("kpt_shape")
            if isinstance(kps, (list, tuple)) and len(kps) >= 1:
                try:
                    desired_k = int(kps[0])
                except Exception:
                    desired_k = 17

        instances = pose.get("instances") or []

        classes_list: list[int] = []
        boxes_list: list[list[float]] = []
        keypoints_list: list[np.ndarray] = []

        for inst in instances:
            # Class (dummy 0 if missing)
            label_val = inst.get(tlc.LABEL, 0)
            mapped = self._class_map.get(label_val, label_val)
            classes_list.append(int(mapped))

            # Keypoints xys
            xys = inst.get("xys", [])
            if isinstance(xys, list) and len(xys) >= 2:
                xys_arr = np.array(xys, dtype=np.float32).reshape(-1, 2)
            else:
                xys_arr = np.zeros((0, 2), dtype=np.float32)

            # Visibilities: prefer xys_additional_data['visibilities'] else ones
            add = inst.get("xys_additional_data") or {}
            if isinstance(add.get("visibilities"), list):
                vis = np.array(add["visibilities"], dtype=np.float32).reshape(-1, 1)
            else:
                # fallback to conf if provided
                if isinstance(add.get("conf"), list):
                    vis = np.array(add["conf"], dtype=np.float32).reshape(-1, 1)
                else:
                    vis = np.ones((xys_arr.shape[0], 1), dtype=np.float32)

            # Normalize x,y to [0,1] using full-image bounds
            if xys_arr.size:
                norm_xy = np.empty_like(xys_arr)
                norm_xy[:, 0] = xys_arr[:, 0] / width
                norm_xy[:, 1] = xys_arr[:, 1] / height
                norm_xy = np.clip(norm_xy, 0.0, 1.0)
            else:
                norm_xy = xys_arr

            # Pad/truncate to desired_k, then stack to (K,3)
            k = norm_xy.shape[0]
            kp = np.zeros((desired_k, 3), dtype=np.float32)
            if k:
                copy_k = min(desired_k, k)
                kp[:copy_k, :2] = norm_xy[:copy_k]
                if vis.shape[0] >= copy_k:
                    kp[:copy_k, 2:3] = vis[:copy_k]
                else:
                    # pad missing vis with ones
                    pad_vis = np.ones((copy_k, 1), dtype=np.float32)
                    pad_vis[: vis.shape[0]] = vis
                    kp[:copy_k, 2:3] = pad_vis

            keypoints_list.append(kp)

            # Derive bbox from visible keypoints
            if kp.shape[0]:
                vis_mask = kp[:, 2] > 0
                pts = kp[vis_mask, :2] if np.any(vis_mask) else kp[:, :2]
                if pts.size:
                    x0 = float(pts[:, 0].min())
                    y0 = float(pts[:, 1].min())
                    x1 = float(pts[:, 0].max())
                    y1 = float(pts[:, 1].max())
                    cx = (x0 + x1) / 2.0
                    cy = (y0 + y1) / 2.0
                    bw = max(x1 - x0, 1e-6)
                    bh = max(y1 - y0, 1e-6)
                    box = [cx, cy, bw, bh]
                else:
                    box = [0.5, 0.5, 1e-6, 1e-6]
            else:
                box = [0.5, 0.5, 1e-6, 1e-6]

            boxes_list.append(box)

        # Convert to arrays with expected shapes
        cls_arr = np.array(classes_list, dtype=np.float32).reshape(-1, 1)
        bboxes_arr = (
            np.array(boxes_list, dtype=np.float32).reshape(-1, 4) if boxes_list else np.zeros((0, 4), dtype=np.float32)
        )

        # Stack to (N, K, 3) (N may be 0, keep K fixed)
        if keypoints_list:
            kp_stack = np.stack(keypoints_list, axis=0)
        else:
            kp_stack = np.zeros((0, desired_k, 3), dtype=np.float32)

        # Height/width integers for metadata (H,W)
        shape_hw = (round(height), round(width))

        return {
            "im_file": im_file,
            "shape": shape_hw,
            "cls": cls_arr,
            "bboxes": bboxes_arr,
            "segments": [],
            "keypoints": kp_stack,  # (N, K, 3)
            "lines": pose["instances"][0]["lines"] if len(pose["instances"]) > 0 else [],
            "normalized": True,
            "bbox_format": "xywh",
            "example_id": example_id,
        }
