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
        emb_schemas = {}
        if self._settings.instance_embeddings_dim > 0:
            from tlc_ultralytics.utils.schemas import _instance_embeddings_list_schema

            dim = self._settings.instance_embeddings_dim
            emb_schemas["predicted_instance_embedding"] = _instance_embeddings_list_schema(
                dim, display_name=f"Predicted Instance Embedding ({dim}D)"
            )

            if self._settings.ground_truth_instance_embeddings:
                emb_schemas["ground_truth_instance_embedding"] = _instance_embeddings_list_schema(
                    dim, display_name=f"Ground Truth Instance Embedding ({dim}D)"
                )

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
        return {KEYPOINTS_2D_PREDICTED: predicted_pose_schema, **loss_schemas, **emb_schemas}

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
            kconf = kpts[j, :, 2].cpu().numpy().astype(np.float32).tolist() if kpts.shape[2] == 3 else None
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

    def _add_instance_embeddings_hook(self, model) -> int:
        """Add a hook to capture class-discriminative feature maps for instance embeddings.

        Delegates to the detection validator's cls-head hook implementation.
        """
        from tlc_ultralytics.detect.validator import TLCDetectionValidator

        return TLCDetectionValidator._add_instance_embeddings_hook(self, model)

    @staticmethod
    def _find_cls_head(model):
        from tlc_ultralytics.detect.validator import TLCDetectionValidator

        return TLCDetectionValidator._find_cls_head(model)

    def _add_cls_head_hooks(self, model) -> int:
        from tlc_ultralytics.detect.validator import TLCDetectionValidator

        return TLCDetectionValidator._add_cls_head_hooks(self, model)

    def _extract_instance_embeddings(self, preds, batch) -> list[np.ndarray]:
        """Extract per-instance embeddings using bboxes (same as detection)."""
        from tlc_ultralytics.utils.embeddings import _extract_instance_embeddings_bbox

        feature_map = self._instance_feature_map

        # Use model-input (letterboxed) coords — pred["bboxes"] are already in this space.
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
            filtered_conf = pred["conf"][mask]

            max_det = self._settings.max_det
            if len(filtered_conf) > max_det:
                topk = filtered_conf.topk(max_det).indices
                filtered_bboxes = filtered_bboxes[topk]

            bboxes_list.append(filtered_bboxes)

        return _extract_instance_embeddings_bbox(feature_map, bboxes_list, image_sizes)

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
        from tlc_ultralytics.utils.embeddings import _extract_instance_embeddings_bbox

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

        return _extract_instance_embeddings_bbox(feature_map, bboxes_list, image_sizes)

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
