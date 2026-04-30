from __future__ import annotations

import weakref

import torch
from tlc.constants import PREDICTED_BOUNDING_BOXES
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops

from tlc_ultralytics.constants import (
    DETECTION_LABEL_COLUMN_NAME,
    IMAGE_COLUMN_NAME,
    TLC_COLORSTR,
)
from tlc_ultralytics.detect.loss import v8UnreducedDetectionLoss
from tlc_ultralytics.detect.utils import (
    build_tlc_yolo_dataset,
    construct_bbox_struct,
    yolo_loss_schemas,
    yolo_predicted_bounding_box_schema,
)
from tlc_ultralytics.engine.validator import TLCValidatorMixin
from tlc_ultralytics.utils.dataset import check_tlc_dataset


class TLCDetectionValidator(TLCValidatorMixin, DetectionValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = DETECTION_LABEL_COLUMN_NAME

    def check_dataset(self, *args, **kwargs):
        return check_tlc_dataset(*args, task="detect", settings=self._settings, **kwargs)

    def build_dataset(self, table, mode="val", batch=None):
        return build_tlc_yolo_dataset(
            self.args,
            table,
            batch,
            self.data,
            mode=mode,
            stride=self.stride,
            exclude_zero=self._settings.exclude_zero_weight_collection,
            class_map=self.data["3lc_class_to_range"],
            split=self.args.split,
            image_column_name=self._settings.image_column_name,
            label_column_name=self._settings.label_column_name,
        )

    def postprocess(self, preds):
        self._curr_raw_preds = preds if self._settings.collect_loss else None
        return super().postprocess(preds)

    def _get_metrics_schemas(self):
        loss_schemas = yolo_loss_schemas(training=self._training) if self._settings.collect_loss else {}

        return {
            PREDICTED_BOUNDING_BOXES: yolo_predicted_bounding_box_schema(self.data["names_3lc"]),
            **loss_schemas,
        }

    def _compute_3lc_metrics(self, preds, batch):
        losses = self.loss_fn(self._curr_raw_preds, batch) if self._settings.collect_loss else {}

        return {
            PREDICTED_BOUNDING_BOXES: self._process_predictions(preds, batch),
            **{k: tensor.mean(dim=1).cpu().numpy() for k, tensor in losses.items()},
        }

    def _build_annotation(self, scaled, mapped_classes, h, w):
        bboxes_xywhn = ops.xyxy2xywhn(scaled["bboxes"], w=w, h=h)
        annotations = [
            {"score": conf, "category_id": label, "bbox": box.cpu().tolist()}
            for conf, label, box in zip(scaled["conf"].tolist(), mapped_classes, bboxes_xywhn, strict=True)
        ]
        return construct_bbox_struct(annotations, image_width=w, image_height=h)

    def _empty_annotation(self, h, w):
        return construct_bbox_struct([], image_width=w, image_height=h)

    def _prepare_loss_fn(self, model):
        # Get the inner model for checking end2end attribute
        inner_model = model.model if hasattr(model.model, "model") else model

        # Check if this is a YOLO26 (end2end) model - per-sample loss is not supported for these
        is_end2end = getattr(inner_model.model[-1], "end2end", False) if hasattr(inner_model, "model") else False

        if is_end2end and self._settings.collect_loss:
            LOGGER.warning(
                f"{TLC_COLORSTR}Per-sample loss collection is not supported for YOLO26 (end2end) models. "
                "Disabling loss collection for this run."
            )
            self._settings.collect_loss = False
            return

        if self._settings.collect_loss:
            self.loss_fn = v8UnreducedDetectionLoss(
                inner_model,
                training=self._training,
            )

    def _add_embeddings_hook(self, model) -> int:
        if hasattr(model.model, "model"):
            model = model.model

        # Find index of the SPPF layer
        sppf_index = next((i for i, m in enumerate(model.model) if "SPPF" in m.type), -1)

        if sppf_index == -1:
            raise ValueError(
                "Image level embeddings can only be collected for detection models with a SPPF layer, "
                "but this model does not have one."
            )

        weak_self = weakref.ref(self)  # Avoid circular reference (self <-> hook_fn)

        def hook_fn(_module, _input, output):
            # Store embeddings
            self_ref = weak_self()
            flattened_output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1)).squeeze(-1).squeeze(-1)
            embeddings = flattened_output.detach().cpu().numpy()
            self_ref.embeddings = embeddings

        # Add forward hook to collect embeddings
        for i, module in enumerate(model.model):
            if i == sppf_index:
                self._hook_handles.append(module.register_forward_hook(hook_fn))

        activation_size = model.model[sppf_index]._modules["cv2"]._modules["conv"].out_channels
        return activation_size

    def _infer_batch_size(self, preds, batch) -> int:
        return len(batch["im_file"])
