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
)
from tlc_ultralytics.detect.validator import TLCDetectionValidator
from tlc_ultralytics.utils.dataset import check_tlc_dataset


class TLCSegmentationValidator(TLCDetectionValidator, SegmentationValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME

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

        # Instance-embedding columns are added by the mixin (raw during streaming,
        # reduced during the end-of-pass rewrite).
        return {PREDICTED_SEGMENTATIONS: segment_schema}

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

    # Instance embeddings pool the feature map with the predicted/GT segmentation masks.
    _instance_geometry_kind = "mask"

    def _instance_regions(self, source, h: int, w: int, device) -> torch.Tensor:
        masks = source.get("masks") if source is not None else None
        if masks is None or masks.numel() == 0:
            return torch.empty((0, h, w), device=device)
        return masks.to(device)
