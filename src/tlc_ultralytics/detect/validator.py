from __future__ import annotations

import weakref

import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops

from tlc_ultralytics.constants import (
    DETECTION_LABEL_COLUMN_NAME,
    IMAGE_COLUMN_NAME,
    PREDICTED_BOUNDING_BOXES,
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
            PREDICTED_BOUNDING_BOXES: bbox_schema,
            **loss_schemas,
            **emb_schemas,
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

    def _add_instance_embeddings_hook(self, model) -> int:
        """Add a hook to capture high-resolution feature maps for instance embeddings.

        Returns the channel dimension size of the hooked layer.
        """
        if hasattr(model.model, "model"):
            model = model.model

        layer_index = self._settings.instance_embeddings_layer

        if layer_index is None:
            # Auto-detect: find the highest-resolution C3k2/C2f layer in the neck.
            # In the FPN neck, features are upsampled progressively: SPPF (20x20) -> P4 (40x40) -> P3 (80x80).
            # The LAST C3k2/C2f layer before any downsampling Conv is the P3 output with highest spatial detail.
            # In YOLO11, this is layer 16 (C3k2, 80x80). In YOLOv8, it's also the last C2f before
            # the bottom-up path starts with a strided Conv.
            sppf_index = next((i for i, m in enumerate(model.model) if "SPPF" in m.type), -1)
            candidates = []
            for i, m in enumerate(model.model):
                if i > sppf_index and any(t in m.type for t in ("C3k2", "C2f")):
                    candidates.append(i)

            if candidates:
                # Find the last candidate before a downsampling Conv (stride=2) appears.
                # This is the P3 neck output — highest resolution feature map in the neck.
                p3_index = candidates[0]
                for idx in candidates:
                    # Check if the next layer is a strided Conv (start of bottom-up path)
                    next_idx = idx + 1
                    if next_idx < len(model.model):
                        next_layer = model.model[next_idx]
                        if "Conv" in next_layer.type and hasattr(next_layer, "conv"):
                            stride = next_layer.conv.stride
                            if isinstance(stride, tuple):
                                stride = stride[0]
                            if stride >= 2:
                                # This candidate is the last one before downsampling
                                p3_index = idx
                                break
                    else:
                        p3_index = idx
                layer_index = p3_index
            else:
                raise ValueError(
                    "Could not auto-detect a suitable layer for instance embeddings. "
                    "Please set instance_embeddings_layer manually in settings."
                )

        LOGGER.info(
            f"{TLC_COLORSTR}Using layer {layer_index} ({model.model[layer_index].type}) "
            "for instance embeddings extraction."
        )

        weak_self = weakref.ref(self)

        def hook_fn(_module, _input, output):
            self_ref = weak_self()
            self_ref._instance_feature_map = output

        self._hook_handles.append(model.model[layer_index].register_forward_hook(hook_fn))

        # Infer channel size from the layer's output convolution
        layer = model.model[layer_index]
        # Try common patterns for getting output channels
        if hasattr(layer, "cv2") and hasattr(layer.cv2, "conv"):
            return layer.cv2.conv.out_channels
        elif hasattr(layer, "cv2") and hasattr(layer.cv2, "out_channels"):
            return layer.cv2.out_channels
        elif hasattr(layer, "c"):
            return layer.c
        else:
            # Fallback: run a test or just use a sensible default
            LOGGER.warning(
                f"{TLC_COLORSTR}Could not infer channel size for layer {layer_index}. "
                "Instance embedding dimension will be determined at runtime."
            )
            return 256  # Common default for P3

    def _extract_instance_embeddings(self, preds, batch) -> list[np.ndarray]:
        """Extract per-instance embeddings from the captured feature map using bboxes."""
        from tlc_ultralytics.utils.embeddings import extract_instance_embeddings_bbox

        feature_map = self._instance_feature_map

        # Get predicted bboxes (xyxy in pixel coords) and image sizes per image
        bboxes_list = []
        image_sizes = []
        for i, pred in enumerate(preds):
            pbatch = self._prepare_batch(i, batch)
            h, w = pbatch["ori_shape"]
            image_sizes.append((h, w))

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

            # Scale bboxes to original image size
            scaled = self.scale_preds(
                {"bboxes": filtered_bboxes, "conf": filtered_conf[:len(filtered_bboxes)]}, pbatch
            )
            bboxes_list.append(scaled["bboxes"])

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

        bboxes_list = []
        image_sizes = []
        for i in range(len(preds)):
            pbatch = self._prepare_batch(i, batch)
            h, w = pbatch["ori_shape"]
            image_sizes.append((h, w))

            gt_bboxes = pbatch["bboxes"]  # GT bboxes in resized image coords (xyxy)
            if gt_bboxes.numel() == 0:
                bboxes_list.append(torch.empty((0, 4), device=feature_map.device))
            else:
                # Scale GT bboxes from resized image coords to original image coords
                scaled_bboxes = ops.scale_boxes(
                    pbatch["imgsz"], gt_bboxes.clone(), pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
                )
                bboxes_list.append(scaled_bboxes.to(feature_map.device))

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
