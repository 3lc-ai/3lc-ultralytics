from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tlc.core.builtins.constants import KEYPOINTS_2D_PREDICTED
from tlc.core.builtins.schemas import Keypoints2DSchema
from tlc.core.data_formats import Keypoints2DInstances
from ultralytics.models.yolo.pose.val import PoseValidator
from ultralytics.utils import LOGGER

from tlc_ultralytics.constants import IMAGE_COLUMN_NAME, POSE_LABEL_COLUMN_NAME, TLC_COLORSTR
from tlc_ultralytics.engine.validator import TLCValidatorMixin
from tlc_ultralytics.pose.dataset import TLCYOLOPoseDataset
from tlc_ultralytics.pose.loss import v8UnreducedPoseLoss
from tlc_ultralytics.pose.utils import yolo_pose_loss_schemas
from tlc_ultralytics.utils.dataset import check_tlc_dataset

if TYPE_CHECKING:
    import tlc


class TLCPoseValidator(TLCValidatorMixin, PoseValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = POSE_LABEL_COLUMN_NAME

    def check_dataset(self, *args, **kwargs):
        return check_tlc_dataset(*args, task="pose", settings=self._settings, **kwargs)

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
            image_column_name=self._settings.image_column_name,
            label_column_name=self._settings.label_column_name,
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)

        if self.data.get("oks_sigmas"):
            self.sigma = np.array(self.data.get("oks_sigmas"))

    def postprocess(self, preds):
        self._curr_raw_preds = preds if self._settings.collect_loss else None
        return super().postprocess(preds)

    def _get_metrics_schemas(self) -> dict[str, tlc.Schema]:
        predicted_pose_schema = Keypoints2DSchema(
            classes=self.data["names"],
            num_keypoints=self.kpt_shape[0],
            points=self.data.get("points"),
            point_attributes=self.data.get("keypoint_attributes"),
            lines=self.data.get("lines"),
            line_attributes=self.data.get("line_attributes"),
            triangles=self.data.get("triangles"),
            triangle_attributes=self.data.get("triangle_attributes"),
            include_per_instance_confidence=True,
            include_per_point_confidence=self.kpt_shape[1] == 3,
            writable=False,
        )

        loss_schemas = yolo_pose_loss_schemas(training=self._training) if self._settings.collect_loss else {}
        return {KEYPOINTS_2D_PREDICTED: predicted_pose_schema, **loss_schemas}

    def _compute_3lc_metrics(self, preds, batch) -> dict[str, Any]:
        losses = self.loss_fn(self._curr_raw_preds, batch) if self._settings.collect_loss else {}
        return {
            KEYPOINTS_2D_PREDICTED: self._process_predictions(preds, batch),
            **{k: tensor.mean(dim=1).cpu().numpy() for k, tensor in losses.items()},
        }

    def _build_annotation(self, scaled, mapped_classes, h, w):
        builder = Keypoints2DInstances.create_empty(
            image_height=int(h),
            image_width=int(w),
            include_instance_confidences=True,
        )
        kpts = scaled["kpts"]  # scale_preds outputs scaled keypoints under "kpts"
        for j in range(len(mapped_classes)):
            kxy = kpts[j, :, 0:2].cpu().numpy().astype(np.float32)
            kconf = (
                kpts[j, :, 2].cpu().numpy().astype(np.float32).tolist() if kpts.shape[2] == 3 else None
            )
            builder.add_instance(
                keypoints=kxy,
                bbox=scaled["bboxes"][j].cpu().numpy().astype(np.float32).tolist(),
                label=int(mapped_classes[j]),
                confidence=kconf,
                normalized=False,
                bbox_format="xyxy",
                instance_confidence=float(scaled["conf"][j]),
            )
        return builder.to_row()

    def _empty_annotation(self, h, w):
        return Keypoints2DInstances.create_empty(
            image_height=int(h),
            image_width=int(w),
            include_instance_confidences=True,
        ).to_row()

    def _prepare_loss_fn(self, model):
        loss_model = model.model if hasattr(model.model, "model") else model

        # Check if this is a YOLO26 (end2end) model - per-sample loss is not supported for these
        is_end2end = getattr(loss_model.model[-1], "end2end", False) if hasattr(loss_model, "model") else False

        if is_end2end and self._settings.collect_loss:
            LOGGER.warning(
                f"{TLC_COLORSTR}Per-sample loss collection is not supported for YOLO26 (end2end) models. "
                "Disabling loss collection for this run."
            )
            self._settings.collect_loss = False
            return

        if self._settings.collect_loss:
            # Pass through dataset-provided OKS sigmas to the loss via model attribute for consistency
            oks_sigmas = self._settings.oks_sigmas or self.data.get("oks_sigmas")
            if oks_sigmas is not None:
                loss_model.oks_sigmas = oks_sigmas
            self.loss_fn = v8UnreducedPoseLoss(loss_model, training=self._training)

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
