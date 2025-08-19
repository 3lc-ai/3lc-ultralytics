from __future__ import annotations

from typing import Any

import numpy as np
import tlc
import torch
from ultralytics.models.yolo.pose.val import PoseValidator

from tlc_ultralytics.constants import IMAGE_COLUMN_NAME, POSE_LABEL_COLUMN_NAME
from tlc_ultralytics.engine.validator import TLCValidatorMixin
from tlc_ultralytics.pose.dataset import TLCYOLOPoseDataset
from tlc_ultralytics.pose.utils import tlc_check_pose_dataset


class TLCPoseValidator(TLCValidatorMixin, PoseValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = POSE_LABEL_COLUMN_NAME

    def check_dataset(self, *args, **kwargs):
        # return tlc_check_pose_dataset(*args, **kwargs)
        tables = args[1]
        return {
            **tables,
            "names": {0: "person"},
            "names_3lc": {"person": 0},
            "nc": 1,
            "range_to_3lc_class": {0: 0},
            "3lc_class_to_range": {0: 0},
            "channels": 3,  # TODO(Frederik): Read out channels from appropriate place and populate here
            "kpt_shape": (17, 3),
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
        values_dict = {
            "xys": tlc.Schema(value=tlc.Float32Value(), size0=tlc.DimensionNumericValue()),
            "lines": tlc.Schema(value=tlc.Int32Value(), size0=tlc.DimensionNumericValue()),
        }
        values_dict["xys_additional_data"] = tlc.Schema(
            values={
                "conf": tlc.Schema(value=tlc.Float32Value(), size0=tlc.DimensionNumericValue()),
            }
        )

        schema = tlc.Schema(
            values={
                "instances": tlc.Schema(values=values_dict, size0=tlc.DimensionNumericValue()),
                "x_min": tlc.Schema(value=tlc.Float32Value()),
                "y_min": tlc.Schema(value=tlc.Float32Value()),
                "x_max": tlc.Schema(value=tlc.Float32Value()),
                "y_max": tlc.Schema(value=tlc.Float32Value()),
            }
        )
        return {"pose_predicted": schema}

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

            kpts = pred["keypoints"].detach().cpu().numpy()  # (N, K, D)
            num_instances = kpts.shape[0]
            instances = []
            for j in range(num_instances):
                xy = kpts[j, :, :2].reshape(-1)
                conf = kpts[j, :, 2] if kpts.shape[2] >= 3 else np.ones(kpts.shape[1], dtype=np.float32)
                inst = {
                    "xys": xy.tolist(),
                    "lines": batch["lines"][0],
                    "xys_additional_data": {"conf": conf.astype(np.float32).tolist()},
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
