import numpy as np
import tlc
import torch
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import ops

from tlc_ultralytics.constants import (
    IMAGE_COLUMN_NAME,
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
        # TODO: Ensure class  mapping is the same as in input table
        instance_properties_structure = {
            tlc.LABEL: tlc.CategoricalLabel(name=tlc.LABEL, classes=self.data["names_3lc"]),
            tlc.CONFIDENCE: tlc.Float(name=tlc.CONFIDENCE, number_role=tlc.NUMBER_ROLE_CONFIDENCE),
        }

        segment_sample_type = tlc.InstanceSegmentationMasks(
            name=tlc.PREDICTED_SEGMENTATIONS,
            instance_properties_structure=instance_properties_structure,
            is_prediction=True,
        )

        seg_schema = segment_sample_type.schema

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

        return {tlc.PREDICTED_SEGMENTATIONS: seg_schema, **emb_schemas}

    def _compute_3lc_metrics(self, preds, batch):
        return {tlc.PREDICTED_SEGMENTATIONS: self._process_predictions(preds, batch)}

    def _build_annotation(self, scaled, mapped_classes, h, w):
        masks = scaled["masks"].cpu().numpy()
        masks = np.transpose(masks, (1, 2, 0))  # (N, H, W) -> (H, W, N)
        return {
            tlc.IMAGE_HEIGHT: h,
            tlc.IMAGE_WIDTH: w,
            tlc.INSTANCE_PROPERTIES: {
                tlc.LABEL: mapped_classes,
                tlc.CONFIDENCE: scaled["conf"].tolist(),
            },
            tlc.MASKS: masks,
        }

    def _empty_annotation(self, h, w):
        return {
            tlc.IMAGE_HEIGHT: h,
            tlc.IMAGE_WIDTH: w,
            tlc.INSTANCE_PROPERTIES: {tlc.LABEL: [], tlc.CONFIDENCE: []},
            tlc.MASKS: np.zeros((h, w, 0), dtype=np.uint8),
        }

    def _extract_instance_embeddings(self, preds, batch) -> list[np.ndarray]:
        """Extract per-instance embeddings using mask-weighted average pooling."""
        from tlc_ultralytics.utils.embeddings import extract_instance_embeddings_mask

        feature_map = self._instance_feature_map

        # pred["masks"] are in model-input (letterboxed) coords, same as the feature map.
        masks_list = []
        image_sizes = []
        for i, pred in enumerate(preds):
            pbatch = self._prepare_batch(i, batch)
            imgsz = pbatch["imgsz"]
            h_in, w_in = int(imgsz[0]), int(imgsz[1])
            image_sizes.append((h_in, w_in))

            mask = pred["conf"] >= self._settings.conf_thres
            if not mask.any():
                masks_list.append(torch.empty((0, h_in, w_in), device=feature_map.device))
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

        # GT masks from _prepare_batch are in model-input (letterboxed) coords
        masks_list = []
        image_sizes = []
        for i in range(len(preds)):
            pbatch = self._prepare_batch(i, batch)
            imgsz = pbatch["imgsz"]
            h_in, w_in = int(imgsz[0]), int(imgsz[1])
            image_sizes.append((h_in, w_in))

            gt_masks = pbatch.get("masks")
            if gt_masks is None or gt_masks.numel() == 0:
                masks_list.append(torch.empty((0, h_in, w_in), device=feature_map.device))
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
