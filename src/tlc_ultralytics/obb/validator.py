import tlc
import torch
from tlc.core.builtins.schemas import Geometry2DSchema
from ultralytics.models.yolo.obb.val import OBBValidator

from tlc_ultralytics.constants import (
    IMAGE_COLUMN_NAME,
    OBB_LABEL_COLUMN_NAME,
)
from tlc_ultralytics.detect.validator import TLCDetectionValidator
from tlc_ultralytics.utils.dataset import check_tlc_dataset


class TLCOBBValidator(TLCDetectionValidator, OBBValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = OBB_LABEL_COLUMN_NAME

    def check_dataset(self, *args, **kwargs):
        return check_tlc_dataset(*args, task="obb", settings=self._settings, **kwargs)

    def _get_metrics_schemas(self) -> dict[str, tlc.Schema]:
        return {
            "oriented_bbs_2d_predicted": Geometry2DSchema(
                include_2d_oriented_bounding_boxes=True,
            )
        }

    # def postprocess(self, preds: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    #     """Post-process predictions. Use native mask processing to get full-size masks with higher accuracy.
    #     These are later used to compute COCO masks which are collected in the 3LC Run.

    #     preds: Predictions passed to the validator postprocess method.
    #     returns: Predictions with full-size masks, to be used by Ultralytics and 3LC metrics collection."""

    #     prev_process = self.process
    #     self.process = ops.process_mask_native
    #     preds = SegmentationValidator.postprocess(self, preds)
    #     self.process = prev_process

    #     return preds

    def _compute_3lc_metrics(self, preds, batch) -> dict[str, list[dict[str, any]]]:
        """Compute 3LC metrics for instance segmentation.

        :param preds: Predictions returned by YOLO segmentation model.
        :param batch: Batch of data presented to the YOLO segmentation model.
        :returns: Metrics dict with predicted instance data for each sample in a batch.
        """
        predicted_bbs = []

        for i, pred in enumerate(preds):
            conf = pred["conf"]
            keep_indices = conf >= self._settings.conf_thres
            if not torch.any(keep_indices):
                continue

        return {"oriented_bbs_2d_predicted": predicted_bbs}
