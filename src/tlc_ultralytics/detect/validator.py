from __future__ import annotations

import weakref

import numpy as np
import tlc
import torch
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
        bbox_schema = yolo_predicted_bounding_box_schema(self.data["names_3lc"])

        emb_schemas = {}
        if self._settings.instance_embeddings_dim > 0:
            from tlc_ultralytics.utils.schemas import instance_embeddings_list_schema

            dim = self._settings.instance_embeddings_dim
            emb_schemas["predicted_instance_embedding"] = instance_embeddings_list_schema(
                dim, display_name=f"Predicted Instance Embedding ({dim}D)"
            )

            if self._settings.ground_truth_instance_embeddings:
                emb_schemas["ground_truth_instance_embedding"] = instance_embeddings_list_schema(
                    dim, display_name=f"Ground Truth Instance Embedding ({dim}D)"
                )

        return {
            tlc.PREDICTED_BOUNDING_BOXES: bbox_schema,
            **loss_schemas,
            **emb_schemas,
        }

    def _compute_3lc_metrics(self, preds, batch):
        losses = self.loss_fn(self._curr_raw_preds, batch) if self._settings.collect_loss else {}

        return {
            tlc.PREDICTED_BOUNDING_BOXES: self._process_predictions(preds, batch),
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

    def _add_instance_embeddings_hook(self, model) -> int:
        """Add a hook to capture class-discriminative feature maps for instance embeddings.

        By default, hooks into the classification branch (cv3) of the detection head,
        which produces features optimized for class discrimination rather than localization.
        Falls back to a neck layer if instance_embeddings_layer is explicitly set.

        Returns the channel dimension size of the hooked layer(s).
        """
        from tlc_ultralytics.utils.embeddings import _infer_layer_channels

        if hasattr(model.model, "model"):
            model = model.model

        # If user explicitly set a layer index, use the neck-layer approach
        if self._settings.instance_embeddings_layer is not None:
            layer_index = self._settings.instance_embeddings_layer
            LOGGER.info(
                f"{TLC_COLORSTR}Using layer {layer_index} ({model.model[layer_index].type}) "
                "for instance embeddings extraction."
            )

            weak_self = weakref.ref(self)

            def hook_fn(_module, _input, output):
                weak_self()._instance_feature_map = output

            self._hook_handles.append(model.model[layer_index].register_forward_hook(hook_fn))
            return _infer_layer_channels(model.model[layer_index], layer_index)

        # Default: hook the cls branch (cv3) of the detection head for class-discriminative features
        return self._add_cls_head_hooks(model)

    @staticmethod
    def _find_cls_head(model) -> torch.nn.ModuleList | None:
        """Find the cls head ModuleList from the detection head."""
        detect_head = model.model[-1]
        cv3 = detect_head.cv3
        if cv3 is not None:
            return cv3
        if hasattr(detect_head, "one2one"):
            return detect_head.one2one.get("cls_head")
        return None

    def _add_cls_head_hooks(self, model) -> int:
        """Hook the cls branch of the detection head at all FPN levels.

        Captures the penultimate layer output (before the final 1x1 conv to class logits)
        from each FPN level. These are resized to P3 resolution and concatenated into a
        single feature map stored in _instance_feature_map.

        Returns the total channel dimension across all levels.
        """
        import torch.nn.functional as F

        detect_head = model.model[-1]
        cv3 = self._find_cls_head(model)

        if cv3 is None:
            # Fallback to neck layer approach
            from tlc_ultralytics.utils.embeddings import _auto_detect_p3_layer, _infer_layer_channels

            layer_index = _auto_detect_p3_layer(model.model)
            LOGGER.info(
                f"{TLC_COLORSTR}No cls head found, falling back to neck layer {layer_index} "
                "for instance embeddings extraction."
            )
            weak_self = weakref.ref(self)

            def hook_fn(_module, _input, output):
                weak_self()._instance_feature_map = output

            self._hook_handles.append(model.model[layer_index].register_forward_hook(hook_fn))
            return _infer_layer_channels(model.model[layer_index], layer_index)

        # Hook penultimate sub-layer of each FPN level's cls branch
        # cv3[level] = Sequential([DWConv+Conv, DWConv+Conv, Conv2d])
        # We want [-2] (second DWConv+Conv block) — class-discriminative features
        hook_sub_index = len(cv3[0]) - 2
        level_features: list[torch.Tensor | None] = [None] * len(cv3)
        weak_self = weakref.ref(self)

        total_channels = 0
        for level_idx in range(len(cv3)):
            target = cv3[level_idx][hook_sub_index]
            try:
                total_channels += target[-1].conv.out_channels
            except (AttributeError, IndexError):
                total_channels += detect_head.nc

            def make_hook(idx):
                def hook_fn(_module, _input, output):
                    level_features[idx] = output
                return hook_fn

            self._hook_handles.append(target.register_forward_hook(make_hook(level_idx)))

        def combine_hook(_module, _input, _output):
            self_ref = weak_self()
            if self_ref is None or level_features[0] is None:
                return
            target_size = level_features[0].shape[2:]
            resized = [
                F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
                if f.shape[2:] != target_size else f
                for f in level_features
            ]
            self_ref._instance_feature_map = torch.cat(resized, dim=1)

        self._hook_handles.append(detect_head.register_forward_hook(combine_hook))

        LOGGER.info(
            f"{TLC_COLORSTR}Using detection head cls branch (cv3) for instance embeddings "
            f"({len(cv3)} levels, {total_channels} total channels)."
        )
        return total_channels

    def _extract_instance_embeddings(self, preds, batch) -> list[np.ndarray]:
        """Extract per-instance embeddings from the captured feature map using bboxes."""
        from tlc_ultralytics.utils.embeddings import extract_instance_embeddings_bbox

        feature_map = self._instance_feature_map

        # Use model-input (letterboxed) coords — these align with the feature map spatial domain.
        # pred["bboxes"] are already in model-input coords, and imgsz is the letterboxed input size.
        bboxes_list = []
        image_sizes = []
        for i, pred in enumerate(preds):
            pbatch = self._prepare_batch(i, batch)
            imgsz = pbatch["imgsz"]
            image_sizes.append((int(imgsz[0]), int(imgsz[1])))

            mask = pred["conf"] >= self._settings.conf_thres
            if not mask.any():
                bboxes_list.append(torch.empty((0, 4), device=feature_map.device))
                continue

            filtered_bboxes = pred["bboxes"][mask]

            # Keep only top max_det
            filtered_conf = pred["conf"][mask]
            max_det = self._settings.max_det
            if len(filtered_conf) > max_det:
                topk = filtered_conf.topk(max_det).indices
                filtered_bboxes = filtered_bboxes[topk]

            bboxes_list.append(filtered_bboxes)

        return extract_instance_embeddings_bbox(feature_map, bboxes_list, image_sizes)

    def _inject_instance_embeddings(self, batch_metrics, reduced_embeddings):
        """Inject reduced predicted instance embeddings as a top-level metric column."""
        pred_embeddings = []
        for emb_array in reduced_embeddings:
            if emb_array.shape[0] > 0:
                pred_embeddings.append([emb_array[i].astype(np.float32).tolist() for i in range(len(emb_array))])
            else:
                pred_embeddings.append([])
        batch_metrics["predicted_instance_embedding"] = pred_embeddings

    def _extract_gt_instance_embeddings(self, preds, batch) -> list[np.ndarray]:
        """Extract per-instance embeddings for ground-truth bboxes."""
        from tlc_ultralytics.utils.embeddings import extract_instance_embeddings_bbox

        feature_map = self._instance_feature_map

        # GT bboxes from _prepare_batch are already in model-input (letterboxed) coords
        bboxes_list = []
        image_sizes = []
        for i in range(len(preds)):
            pbatch = self._prepare_batch(i, batch)
            imgsz = pbatch["imgsz"]
            image_sizes.append((int(imgsz[0]), int(imgsz[1])))

            gt_bboxes = pbatch["bboxes"]
            if gt_bboxes.numel() == 0:
                bboxes_list.append(torch.empty((0, 4), device=feature_map.device))
            else:
                bboxes_list.append(gt_bboxes.to(feature_map.device))

        return extract_instance_embeddings_bbox(feature_map, bboxes_list, image_sizes)

    def _inject_gt_instance_embeddings(self, batch_metrics, reduced_embeddings):
        """Inject reduced GT instance embeddings as a top-level metric column."""
        gt_embeddings = []
        for emb_array in reduced_embeddings:
            if emb_array.shape[0] > 0:
                gt_embeddings.append([emb_array[i].astype(np.float32).tolist() for i in range(len(emb_array))])
            else:
                gt_embeddings.append([])
        batch_metrics["ground_truth_instance_embedding"] = gt_embeddings

    def _infer_batch_size(self, preds, batch) -> int:
        return len(batch["im_file"])
