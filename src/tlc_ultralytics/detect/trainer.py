from __future__ import annotations

from functools import partial
from typing import ClassVar

import ultralytics
from ultralytics.models.yolo.detect import DetectionTrainer

from tlc_ultralytics.constants import (
    DETECTION_LABEL_COLUMN_NAME,
    IMAGE_COLUMN_NAME,
)
from tlc_ultralytics.detect.utils import (
    build_tlc_yolo_dataset,
)
from tlc_ultralytics.detect.validator import TLCDetectionValidator
from tlc_ultralytics.engine.trainer import TLCTrainerMixin


class TLCDetectionTrainer(TLCTrainerMixin, DetectionTrainer):
    """Trainer class for YOLO object detection with 3LC"""

    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = DETECTION_LABEL_COLUMN_NAME
    _validator_class = TLCDetectionValidator
    _loss_names = ("box_loss", "cls_loss", "dfl_loss")
    _metric_replacements: ClassVar[list[tuple[str, str]]] = [("(B)", ""), ("metrics", "val"), ("/", "_")]
    _build_dataloader_module = ultralytics.models.yolo.detect.train

    def build_dataset(self, *args, **kwargs):
        from ultralytics.models.yolo.detect.train import build_yolo_dataset as original_build_yolo_dataset

        mode = kwargs.get("mode") or args[1]

        exclude_zero = mode == "val" and self._settings.exclude_zero_weight_collection
        ultralytics.models.yolo.detect.train.build_yolo_dataset = partial(
            build_tlc_yolo_dataset,
            exclude_zero=exclude_zero,
            class_map=self.data["3lc_class_to_range"],
            image_column_name=self._settings.image_column_name,
            label_column_name=self._settings.label_column_name,
        )

        try:
            result = DetectionTrainer.build_dataset(self, *args, **kwargs)
        finally:
            ultralytics.models.yolo.detect.train.build_yolo_dataset = original_build_yolo_dataset

        return result
