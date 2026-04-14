import tlc
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
            tlc.CONFIDENCE: tlc.Float32Schema(number_role=tlc.NUMBER_ROLE_CONFIDENCE),
        }

        segment_schema = tlc.SegmentationMasksSchema(
            classes=self.data["names_3lc"],
            per_instance_schemas=instance_properties_structure,
            writable=False,
        )

        return {tlc.PREDICTED_SEGMENTATIONS: segment_schema}

    def _compute_3lc_metrics(self, preds, batch):
        return {tlc.PREDICTED_SEGMENTATIONS: self._process_predictions(preds, batch)}

    def _build_annotation(self, scaled, mapped_classes, h, w):
        return tlc.SegmentationMasks(
            image_height=h,
            image_width=w,
            masks=scaled["masks"].cpu().numpy(),  # PyTorch-native (N, H, W); transposed by mask_format below
            mask_format="nhw",
            labels=mapped_classes,
            confidences=scaled["conf"].tolist(),
        )

    def _empty_annotation(self, h, w):
        return tlc.SegmentationMasks.create_empty(image_height=h, image_width=w)
