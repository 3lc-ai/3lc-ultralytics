from __future__ import annotations

from functools import partial

import ultralytics
from ultralytics.models import yolo

from tlc_ultralytics.classify.dataset import TLCClassificationDataset
from tlc_ultralytics.classify.utils import tlc_check_cls_dataset
from tlc_ultralytics.classify.validator import TLCClassificationValidator
from tlc_ultralytics.constants import (
    CLASSIFY_LABEL_COLUMN_NAME,
    IMAGE_COLUMN_NAME,
)
from tlc_ultralytics.engine.trainer import TLCTrainerMixin
from tlc_ultralytics.overrides import build_dataloader
from tlc_ultralytics.utils import create_sampler


class TLCClassificationTrainer(TLCTrainerMixin, yolo.classify.ClassificationTrainer):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = CLASSIFY_LABEL_COLUMN_NAME

    def get_dataset(self):
        """Overrides the get_dataset method to get or create 3LC tables."""
        self.data = tlc_check_cls_dataset(
            self.args.data,
            self._tables,
            self._image_column_name,
            self._label_column_name,
            project_name=self._settings.project_name,
            splits=("train", "val"),
        )
        if "val" not in self.data:
            data_test = tlc_check_cls_dataset(
                self.args.data,
                self._tables,
                self._image_column_name,
                self._label_column_name,
                project_name=self._settings.project_name,
                splits=("test",),
            )
            self.data["test"] = data_test["test"]

        return self.data

    def build_dataset(self, table, mode="train", batch=None):
        exclude_zero = mode == "val" and self._settings.exclude_zero_weight_collection

        return TLCClassificationDataset(
            table,
            args=self.args,
            augment=mode == "train",
            prefix=mode,
            image_column_name=self._image_column_name,
            label_column_name=self._label_column_name,
            exclude_zero=exclude_zero,
            class_map=self.data["3lc_class_to_range"],
        )

    def get_validator(self, dataloader=None):
        self.loss_names = ["loss"]
        dataloader = dataloader or self.test_loader
        return TLCClassificationValidator(
            dataloader,
            self.save_dir,
            _callbacks=self.callbacks,
            run=self._run,
            image_column_name=self._image_column_name,
            label_column_name=self._label_column_name,
            settings=self._settings,
            training=True,
        )

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        sampler = create_sampler(dataset_path, mode, self._settings, distributed=rank != -1)

        # Patch parent class module to use our build_dataloader
        trainer_build_dataloader = ultralytics.models.yolo.classify.train.build_dataloader
        ultralytics.models.yolo.classify.train.build_dataloader = partial(build_dataloader, sampler=sampler)

        dataloader = super().get_dataloader(dataset_path, batch_size, rank, mode)

        ultralytics.models.yolo.classify.train.build_dataloader = trainer_build_dataloader

        return dataloader
