from __future__ import annotations

from typing import ClassVar

import ultralytics
from ultralytics.models import yolo

from tlc_ultralytics.classify.dataset import TLCClassificationDataset
from tlc_ultralytics.classify.validator import TLCClassificationValidator
from tlc_ultralytics.constants import (
    IMAGE_COLUMN_NAME,
)
from tlc_ultralytics.engine.trainer import TLCTrainerMixin


class TLCClassificationTrainer(TLCTrainerMixin, yolo.classify.ClassificationTrainer):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _validator_class = TLCClassificationValidator
    _loss_names: ClassVar[list[str]] = ["loss"]
    _build_dataloader_module = ultralytics.models.yolo.classify.train

    def build_dataset(self, table, mode="train", batch=None):
        exclude_zero = mode == "val" and self._settings.exclude_zero_weight_collection

        return TLCClassificationDataset(
            table,
            args=self.args,
            augment=mode == "train",
            prefix=mode,
            image_column_name=self._settings.image_column_name,
            label_column_name=self._settings.label_column_name,
            exclude_zero=exclude_zero,
            class_map=self.data["3lc_class_to_range"],
        )
