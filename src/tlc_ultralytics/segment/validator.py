import numpy as np
import tlc
import torch
from tlc.client.data_format import InstanceSegmentationDict
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import ops

from tlc_ultralytics.constants import (
    IMAGE_COLUMN_NAME,
    SEGMENTATION_LABEL_COLUMN_NAME,
)
from tlc_ultralytics.detect.validator import TLCDetectionValidator
from tlc_ultralytics.segment.utils import tlc_check_seg_dataset


class TLCSegmentationValidator(TLCDetectionValidator, SegmentationValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = SEGMENTATION_LABEL_COLUMN_NAME

    def check_dataset(self, *args, **kwargs):
        return tlc_check_seg_dataset(*args, **kwargs)

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

    def _compute_3lc_metrics(self, preds, batch) -> list[dict[str, InstanceSegmentationDict]]:
        """Compute 3LC metrics for instance segmentation.

        :param preds: Predictions returned by YOLO segmentation model.
        :param batch: Batch of data presented to the YOLO segmentation model.
        :returns: Metrics dict with predicted instance data for each sample in a batch.
        """
        predicted_batch_segmentations = []

        # Reimplements SegmentationValidator, but with control over mask processing
        for si, (pred, proto) in enumerate(zip(preds[0], preds[1])):
            pbatch = self._prepare_batch(si, batch)

            conf = pred[:, 4]
            pred_cls = pred[:, 5]

            # Filter out predictions first
            keep_indices = conf >= self._settings.conf_thres

            # Handle case where no predictions are kept
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

            pred_cls = pred_cls[keep_indices]
            conf = conf[keep_indices]
            pred = pred.detach().clone()[keep_indices]

            # Native upsampling to bounding boxes
            prev_process = self.process
            self.process = ops.process_mask_native
            predn, pred_masks = self._prepare_pred(pred, pbatch, proto)
            self.process = prev_process

            # Scale masks to image size and handle padding
            pred_masks = torch.as_tensor(pred_masks, dtype=torch.uint8)

            scaled_masks = ops.scale_image(
                pred_masks.permute(1, 2, 0).contiguous().cpu().numpy(),
                pbatch["ori_shape"],
                ratio_pad=batch["ratio_pad"][si],
            )

            result_masks = np.asfortranarray(scaled_masks.astype(np.uint8))

            # Map predicted labels in 0, 1, ... back to possibly non-contiguous 3LC classes
            predicted_labels = [self.data["range_to_3lc_class"][int(p)] for p in pred_cls.tolist()]

            predicted_instances = {
                tlc.IMAGE_HEIGHT: pbatch["ori_shape"][0],
                tlc.IMAGE_WIDTH: pbatch["ori_shape"][1],
                tlc.INSTANCE_PROPERTIES: {
                    tlc.LABEL: predicted_labels,
                    tlc.CONFIDENCE: conf.tolist(),
                },
                tlc.MASKS: result_masks,
            }

            predicted_batch_segmentations.append(predicted_instances)

        return {tlc.PREDICTED_SEGMENTATIONS: predicted_batch_segmentations}
