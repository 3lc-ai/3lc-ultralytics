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

        return {tlc.PREDICTED_SEGMENTATIONS: segment_sample_type.schema}

    def _compute_3lc_metrics(self, preds, batch) -> dict[str, list[dict[str, any]]]:
        """Compute 3LC metrics for instance segmentation.

        :param preds: Predictions returned by YOLO segmentation model.
        :param batch: Batch of data presented to the YOLO segmentation model.
        :returns: Metrics dict with predicted instance data for each sample in a batch.
        """
        predicted_batch_segmentations = []

        # Reimplements SegmentationValidator, but with control over mask processing
        for i, pred in enumerate(preds):
            pbatch = self._prepare_batch(i, batch)

            conf = pred["conf"]
            keep_indices = conf >= self._settings.conf_thres
            if not torch.any(keep_indices):
                height, width = pbatch["ori_shape"]
                predicted_instances = {
                    tlc.IMAGE_HEIGHT: height,
                    tlc.IMAGE_WIDTH: width,
                    tlc.INSTANCE_PROPERTIES: {
                        tlc.LABEL: [],
                        tlc.CONFIDENCE: [],
                    },
                    tlc.MASKS: np.zeros((height, width, 0), dtype=np.uint8),
                }
                predicted_batch_segmentations.append(predicted_instances)
                continue

            # Filter out low confidence predictions
            pred_conf = conf[keep_indices].tolist()
            pred_cls = pred["cls"][keep_indices].tolist()
            predicted_labels = [self.data["range_to_3lc_class"][int(p)] for p in pred_cls]
            predicted_masks = pred["masks"].clone()[keep_indices]

            # Get masks in resized dimensions (scale_masks expects (N, C, H, W) tensor)
            coco_masks = (
                ops.scale_masks(
                    predicted_masks[None],  # (N_masks, H, W) -> (1, N_masks, H, W)
                    pbatch["ori_shape"],
                    ratio_pad=pbatch["ratio_pad"],
                )[0]
                .byte()
                .cpu()
                .numpy()
            )
            coco_masks = np.transpose(coco_masks, (1, 2, 0))  # (N_masks, H, W) -> (H, W, N_masks)

            predicted_instances = {
                tlc.IMAGE_HEIGHT: pbatch["ori_shape"][0],
                tlc.IMAGE_WIDTH: pbatch["ori_shape"][1],
                tlc.INSTANCE_PROPERTIES: {
                    tlc.LABEL: predicted_labels,
                    tlc.CONFIDENCE: pred_conf,
                },
                tlc.MASKS: coco_masks,
            }

            predicted_batch_segmentations.append(predicted_instances)

        return {tlc.PREDICTED_SEGMENTATIONS: predicted_batch_segmentations}
