from __future__ import annotations

from typing import Any

import numpy as np
import tlc
import torch
from tlc.core.builtins.constants import (
    BBS_2D,
    CONFIDENCE,
    INSTANCES,
    INSTANCES_ADDITIONAL_DATA,
    KEYPOINTS_2D,
    KEYPOINTS_2D_PREDICTED,
    LABEL,
    LINES,
    VERTICES_2D,
    VERTICES_2D_ADDITIONAL_DATA,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
)
from tlc.core.builtins.schemas import CategoricalLabelListSchema, Float32ListSchema, Geometry2DSchema
from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.utils import ops

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
            lines = self._table.rows_schema[KEYPOINTS_2D][INSTANCES][LINES].default_value
        except KeyError:
            lines = None

        predicted_pose_schema = Geometry2DSchema(
            writable=False,
            add_2d_points=True,
            add_2d_bounding_boxes=True,
            num_vertices=self.kpt_shape[0],
            vertex_labels=self.data["kpt_names"],
            add_lines=True,
            per_point_schemas={CONFIDENCE: Float32ListSchema()},
            per_instance_schemas={
                LABEL: CategoricalLabelListSchema(classes=self.data["names"]),
                CONFIDENCE: Float32ListSchema(),
            },
        )

        loss_schemas = yolo_pose_loss_schemas(training=self._training) if self._settings.collect_loss else {}
        return {KEYPOINTS_2D_PREDICTED: predicted_pose_schema, **loss_schemas}

    def _compute_3lc_metrics(self, preds, batch) -> dict[str, Any]:
        predicted = []

        for i, pred in enumerate(preds):
            predicted_keypoints, predicted_confidences, predicted_classes, predicted_bboxes = (
                pred["keypoints"].clone(),
                pred["conf"].clone(),
                pred["cls"].clone(),
                pred["bboxes"].clone(),
            )
            h, w = batch["ori_shape"][i]

            if len(pred) == 0:
                predicted.append(
                    {
                        X_MIN: 0,
                        Y_MIN: 0,
                        X_MAX: w,
                        Y_MAX: h,
                        INSTANCES: [],
                        INSTANCES_ADDITIONAL_DATA: {LABEL: [], CONFIDENCE: []},
                    }
                )
                continue

            # Filter out low confidence predictions
            mask = predicted_confidences > self._settings.conf_thres
            predicted_keypoints = predicted_keypoints[mask]
            predicted_confidences = predicted_confidences[mask].tolist()
            predicted_classes = predicted_classes[mask].tolist()
            predicted_bboxes = predicted_bboxes[mask]

            resized_shape = batch["resized_shape"][i]
            ori_shape = batch["ori_shape"][i]
            ratio_pad = batch["ratio_pad"][i]
            scaled_bboxes = ops.scale_boxes(resized_shape, predicted_bboxes, ori_shape, ratio_pad)
            scaled_keypoints = ops.scale_coords(resized_shape, predicted_keypoints, ori_shape, ratio_pad)

            instances = []
            for j in range(len(predicted_keypoints)):
                predicted_kpts = scaled_keypoints[j]
                predicted_bbox = scaled_bboxes[j].cpu().numpy().astype(np.float32).tolist()
                keypoints = predicted_kpts[:, 0:2].reshape(-1).cpu().numpy().astype(np.float32).tolist()
                confidences = predicted_kpts[:, 2].cpu().numpy().astype(np.float32).tolist()

                instances.append(
                    {
                        VERTICES_2D: keypoints,
                        LINES: batch[LINES][0],
                        VERTICES_2D_ADDITIONAL_DATA: {
                            CONFIDENCE: confidences,
                        },
                        BBS_2D: [
                            {
                                X_MIN: predicted_bbox[0],
                                Y_MIN: predicted_bbox[1],
                                X_MAX: predicted_bbox[2],
                                Y_MAX: predicted_bbox[3],
                            }
                        ],
                    }
                )

            predicted.append(
                {
                    X_MAX: w,
                    Y_MAX: h,
                    X_MIN: 0,
                    Y_MIN: 0,
                    INSTANCES: instances,
                    INSTANCES_ADDITIONAL_DATA: {
                        LABEL: predicted_classes,
                        CONFIDENCE: predicted_confidences,
                    },
                }
            )

        losses = self.loss_fn(self._curr_raw_preds, batch) if self._settings.collect_loss else {}
        return {
            KEYPOINTS_2D_PREDICTED: predicted,
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
