from __future__ import annotations

from typing import Any

import numpy as np
import tlc
import torch
from ultralytics.models.yolo.pose.val import PoseValidator

from tlc_ultralytics.constants import IMAGE_COLUMN_NAME, POSE_LABEL_COLUMN_NAME
from tlc_ultralytics.engine.validator import TLCValidatorMixin
from tlc_ultralytics.pose.dataset import TLCYOLOPoseDataset
from tlc_ultralytics.pose.loss import v8UnreducedPoseLoss
from tlc_ultralytics.pose.utils import yolo_pose_loss_schemas
from tlc_ultralytics.utils.dataset import check_tlc_dataset


class TLCPoseValidator(TLCValidatorMixin, PoseValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = POSE_LABEL_COLUMN_NAME

    def check_dataset(self, *args, **kwargs):
        return check_tlc_dataset(*args, task="pose", **kwargs)

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

    def postprocess(self, preds):
        self._curr_raw_preds = preds if self._settings.collect_loss else None
        return super().postprocess(preds)

    def _get_metrics_schemas(self) -> dict[str, tlc.Schema]:
        try:
            lines = self._table.rows_schema["keypoints_2d"]["instances"]["lines"].default_value
        except KeyError:
            lines = None
        predicted_pose_schema = tlc.Keypoints2DSchema(
            keypoint_shape=self.kpt_shape,
            keypoint_names=self.data["kpt_names"],
            lines_default_value=None,
            per_point_schemas={tlc.CONFIDENCE: tlc.Schema(value=tlc.Float32Value())},
            per_instance_schemas={
                "label": tlc.Schema(
                    value=tlc.Int32Value(value_map=tlc.MapElement._construct_value_map(self.data["names"]))
                )
            },
            writable=False,
        )

        loss_schemas = yolo_pose_loss_schemas(training=self._training) if self._settings.collect_loss else {}
        return {tlc.KEYPOINTS_2D_PREDICTED: predicted_pose_schema, **loss_schemas}

    def _compute_3lc_metrics(self, preds, batch) -> dict[str, Any]:
        predicted = []
        for i, pred in enumerate(preds):
            h, w = batch["ori_shape"][i]

            if len(pred["keypoints"]) == 0:
                predicted.append(
                    {
                        tlc.X_MIN: 0,
                        tlc.Y_MIN: 0,
                        tlc.X_MAX: w,
                        tlc.Y_MAX: h,
                        tlc.INSTANCES: [],
                        tlc.INSTANCES_ADDITIONAL_DATA: {"label": []},
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
            bbs = pred["bboxes"].cpu().numpy()

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
                bb = bbs[j, :]
                # undo letterbox: (xy - pad) / gain (per-axis)
                bb[0] = (bb[0] - padw) / (gw + 1e-9)
                bb[1] = (bb[1] - padh) / (gh + 1e-9)
                bb[2] = (bb[2] - padw) / (gw + 1e-9)
                bb[3] = (bb[3] - padh) / (gh + 1e-9)
                # clamp to image
                bb[0] = np.clip(bb[0], 0, w - 1)
                bb[1] = np.clip(bb[1], 0, h - 1)
                bb[2] = np.clip(bb[2], 0, w - 1)
                bb[3] = np.clip(bb[3], 0, h - 1)

                inst = {
                    tlc.XYS: xy.reshape(-1).astype(np.float32).tolist(),
                    tlc.LINES: batch["lines"][0],
                    tlc.XYS_ADDITIONAL_DATA: {tlc.CONFIDENCE: conf.astype(np.float32).tolist()},
                    "bbs_2d": [
                        {
                            tlc.X_MIN: bb[0],
                            tlc.Y_MIN: bb[1],
                            tlc.X_MAX: bb[2],
                            tlc.Y_MAX: bb[3],
                        }
                    ],
                }
                instances.append(inst)

            predicted.append(
                {
                    tlc.X_MAX: w,
                    tlc.Y_MAX: h,
                    tlc.X_MIN: 0,
                    tlc.Y_MIN: 0,
                    tlc.INSTANCES: instances,
                    tlc.INSTANCES_ADDITIONAL_DATA: {"label": [0] * num_instances},
                }
            )

        losses = self.loss_fn(self._curr_raw_preds, batch) if self._settings.collect_loss else {}
        return {
            tlc.KEYPOINTS_2D_PREDICTED: predicted,
            **{k: tensor.mean(dim=1).cpu().numpy() for k, tensor in losses.items()},
        }

    def _prepare_loss_fn(self, model):
        self.loss_fn = v8UnreducedPoseLoss(
            model.model if hasattr(model.model, "model") else model,
            training=self._training,
        )

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
