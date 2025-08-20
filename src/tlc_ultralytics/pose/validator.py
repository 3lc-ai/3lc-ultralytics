from __future__ import annotations

from typing import Any

import numpy as np
import tlc
import torch
from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.utils.metrics import OKS_SIGMA

from tlc_ultralytics.constants import IMAGE_COLUMN_NAME, POSE_LABEL_COLUMN_NAME
from tlc_ultralytics.engine.validator import TLCValidatorMixin
from tlc_ultralytics.pose.dataset import TLCYOLOPoseDataset
from tlc_ultralytics.pose.utils import tlc_check_pose_dataset


class TLCPoseValidator(TLCValidatorMixin, PoseValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = POSE_LABEL_COLUMN_NAME

    # def init_metrics(self, model: torch.nn.Module) -> None:
    #     """
    #     Initialize evaluation metrics for YOLO pose validation.

    #     Args:
    #         model (torch.nn.Module): Model to validate.
    #     """
    #     super(PoseValidator, self).init_metrics(model)
    #     # Then override these
    #     # self.kpt_shape = self.data["kpt_shape"]
    #     # self.sigma = OKS_SIGMA

    def check_dataset(self, *args, **kwargs):
        # return tlc_check_pose_dataset(*args, **kwargs)
        # TODO: FIXME
        tables = args[1]
        random_table = next(iter(tables.values()))
        return {
            **tables,
            "names": {0: "person"},
            "names_3lc": {"person": 0},
            "nc": 1,
            "range_to_3lc_class": {0: 0},
            "3lc_class_to_range": {0: 0},
            "channels": 3,  # TODO(Frederik): Read out channels from appropriate place and populate here
            "kpt_shape": random_table.kpt_shape if hasattr(random_table, "kpt_shape") else (17, 3),
            "flip_idx": random_table.flip_idx if hasattr(random_table, "flip_idx") else None,
        }

    def build_dataset(self, table, mode: str = "val", batch=None):
        return TLCYOLOPoseDataset(
            table,
            data=self.data,
            exclude_zero=self._settings.exclude_zero_weight_collection,
            class_map=self.data["3lc_class_to_range"],
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            rect=self.args.rect,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(self.stride),
            pad=0.5,
            prefix=self.args.split or mode,
            task=self.args.task,
            classes=self.args.classes,
            fraction=1.0,
            image_column_name=self._image_column_name,
            label_column_name=self._label_column_name,
        )

    def _get_metrics_schemas(self) -> dict[str, tlc.Schema]:
        # self._table.rows_schema["keypoints_2d"]["instances"]["lines"].default_value
        predicted_pose_schema = tlc.Keypoints2DSchema(
            keypoint_shape=self.kpt_shape,
            # keypoint_names=self.keypoint_names,
            # lines_default_value=self.lines,
            per_point_schemas={tlc.CONFIDENCE: tlc.Schema(value=tlc.Float32Value(), size0=tlc.DimensionNumericValue())},
            writable=False,
        )

        return {"pose_predicted": predicted_pose_schema}

    def _compute_3lc_metrics(self, preds, batch) -> dict[str, Any]:
        predicted = []
        for i, pred in enumerate(preds):
            h, w = batch["ori_shape"][i]

            if len(pred["keypoints"]) == 0:
                predicted.append(
                    {
                        "x_max": w,
                        "y_max": h,
                        "x_min": 0,
                        "y_min": 0,
                        "instances": [],
                    }
                )
                continue

            # Parse ratio_pad robustly: ((gain_w, gain_h), (padw, padh)) or (gain, (padw, padh))
            gw = gh = 1.0
            padw = padh = 0.0
            ratio_pad = batch.get("ratio_pad", None)
            if ratio_pad is not None:
                rp = ratio_pad[i]
                if isinstance(rp, (tuple, list)) and len(rp) == 2:
                    gain, pad = rp
                    if isinstance(gain, (tuple, list)):
                        if len(gain) >= 2:
                            gw, gh = float(gain[0]), float(gain[1])
                        elif len(gain) == 1:
                            gw = gh = float(gain[0])
                    else:
                        gw = gh = float(gain)
                    if isinstance(pad, (tuple, list)) and len(pad) >= 2:
                        padw, padh = float(pad[0]), float(pad[1])

            kpts = pred["keypoints"].detach().cpu().numpy()  # (N, K, D)
            num_instances = kpts.shape[0]
            instances = []
            for j in range(num_instances):
                xy = kpts[j, :, :2].copy()
                # undo letterbox: (xy - pad) / gain (per-axis)
                xy[:, 0] = (xy[:, 0] - padw) / (gw + 1e-9)
                xy[:, 1] = (xy[:, 1] - padh) / (gh + 1e-9)
                # clamp to image
                xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
                xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)

                conf = kpts[j, :, 2] if kpts.shape[2] >= 3 else np.ones(kpts.shape[1], dtype=np.float32)
                inst = {
                    "xys": xy.reshape(-1).astype(np.float32).tolist(),
                    "lines": batch["lines"][0],
                    "xys_additional_data": {tlc.CONFIDENCE: conf.astype(np.float32).tolist()},
                }
                instances.append(inst)

            predicted.append(
                {
                    "x_max": w,
                    "y_max": h,
                    "x_min": 0,
                    "y_min": 0,
                    "instances": instances,
                }
            )

        return {"pose_predicted": predicted}

    def _add_embeddings_hook(self, model) -> int:
        if hasattr(model.model, "model"):
            model = model.model

        sppf_index = next((i for i, m in enumerate(model.model) if "SPPF" in m.type), -1)
        if sppf_index == -1:
            return 0

        weak_self = self

        def hook_fn(_module, _input, output):
            flat = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1)).squeeze(-1).squeeze(-1)
            weak_self.embeddings = flat.detach().cpu().numpy()

        self._hook_handles.append(model.model[sppf_index].register_forward_hook(hook_fn))
        return model.model[sppf_index]._modules["cv2"]._modules["conv"].out_channels

    def _infer_batch_size(self, preds, batch) -> int:
        return len(batch["im_file"])
