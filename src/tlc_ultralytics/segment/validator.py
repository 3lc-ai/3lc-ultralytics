import numpy as np
import tlc
import torch
from tlc.data_types import SegmentationMasks
from tlc.schemas import ConfidenceSchema
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import ops

from tlc_ultralytics.constants import (
    CONFIDENCE,
    IMAGE_COLUMN_NAME,
    PREDICTED_SEGMENTATIONS,
    SEGMENTATION_LABEL_COLUMN_NAME,
)
from tlc_ultralytics.detect.validator import TLCDetectionValidator
from tlc_ultralytics.utils.dataset import check_tlc_dataset


class TLCSegmentationValidator(TLCDetectionValidator, SegmentationValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = SEGMENTATION_LABEL_COLUMN_NAME

    def check_dataset(self, *args, **kwargs):
        return check_tlc_dataset(*args, task="segment", settings=self._settings, **kwargs)

    def init_metrics(self, model):
        """Initialize metrics and use native mask processing for full-size masks."""
        super().init_metrics(model)
        # Always use native mask processing for 3LC to get full-size masks
        self.process = ops.process_mask_native

    def _get_metrics_schemas(self) -> dict[str, tlc.Schema]:
        instance_properties_structure = {
            CONFIDENCE: ConfidenceSchema(writable=False),
        }

        segment_schema = SegmentationMasks.schema(
            classes=self.data["names_3lc"],
            per_instance_schemas=instance_properties_structure,
            writable=False,
        )

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

        return {PREDICTED_SEGMENTATIONS: segment_schema, **emb_schemas}

    def _compute_3lc_metrics(self, preds, batch):
        return {PREDICTED_SEGMENTATIONS: self._process_predictions(preds, batch)}

    def _build_annotation(self, scaled, mapped_classes, h, w):
        return tlc.data_types.SegmentationMasks(
            image_height=h,
            image_width=w,
            masks=scaled["masks"].cpu().numpy(),  # PyTorch-native (N, H, W); transposed by mask_format below
            mask_format="nhw",
            labels=mapped_classes,
            confidences=scaled["conf"].tolist(),
        )

    def _empty_annotation(self, h, w):
        return tlc.data_types.SegmentationMasks.create_empty(image_height=h, image_width=w)

    def _extract_instance_embeddings(self, preds, batch) -> list[np.ndarray]:
        """Extract per-instance embeddings using mask-weighted average pooling."""
        from tlc_ultralytics.utils.embeddings import extract_instance_embeddings_mask

        feature_map = self._instance_feature_map

        masks_list = []
        image_sizes = []
        for i, pred in enumerate(preds):
            pbatch = self._prepare_batch(i, batch)
            h, w = pbatch["ori_shape"]
            image_sizes.append((h, w))

            mask = pred["conf"] >= self._settings.conf_thres
            if not mask.any():
                masks_list.append(torch.empty((0, h, w), device=feature_map.device))
                continue

            filtered_masks = pred["masks"][mask]
            filtered_conf = pred["conf"][mask]

            # Keep only top max_det
            max_det = self._settings.max_det
            if len(filtered_conf) > max_det:
                topk = filtered_conf.topk(max_det).indices
                filtered_masks = filtered_masks[topk]

            masks_list.append(filtered_masks)

        return extract_instance_embeddings_mask(feature_map, masks_list, image_sizes)

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
        """Extract per-instance embeddings for ground-truth masks."""
        from tlc_ultralytics.utils.embeddings import extract_instance_embeddings_mask

        feature_map = self._instance_feature_map

        masks_list = []
        image_sizes = []
        for i in range(len(preds)):
            pbatch = self._prepare_batch(i, batch)
            h, w = pbatch["ori_shape"]
            image_sizes.append((h, w))

            gt_masks = pbatch.get("masks")
            if gt_masks is None or gt_masks.numel() == 0:
                masks_list.append(torch.empty((0, h, w), device=feature_map.device))
            else:
                masks_list.append(gt_masks.to(feature_map.device))

        return extract_instance_embeddings_mask(feature_map, masks_list, image_sizes)

    def _inject_gt_instance_embeddings(self, batch_metrics, reduced_embeddings):
        """Inject reduced GT instance embeddings as a top-level metric column."""
        gt_embeddings = []
        for emb_array in reduced_embeddings:
            if emb_array.shape[0] > 0:
                gt_embeddings.append([emb_array[i].astype(np.float32).tolist() for i in range(len(emb_array))])
            else:
                gt_embeddings.append([])
        batch_metrics["ground_truth_instance_embedding"] = gt_embeddings
