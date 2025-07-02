from __future__ import annotations

from functools import partial

import ultralytics
from ultralytics.models.yolo.detect import DetectionTrainer

from tlc_ultralytics.constants import (
    DETECTION_LABEL_COLUMN_NAME,
    IMAGE_COLUMN_NAME,
)
from tlc_ultralytics.detect.utils import (
    build_tlc_yolo_dataset,
    tlc_check_det_dataset,
)
from tlc_ultralytics.detect.validator import TLCDetectionValidator
from tlc_ultralytics.engine.trainer import TLCTrainerMixin
from tlc_ultralytics.overrides import build_dataloader
from tlc_ultralytics.utils import create_sampler


class TLCDetectionTrainer(TLCTrainerMixin, DetectionTrainer):
    """Trainer class for YOLO object detection with 3LC"""

    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = DETECTION_LABEL_COLUMN_NAME

    def get_dataset(self):
        # Parse yaml and create tables
        self.data = tlc_check_det_dataset(
            self.args.data,
            self._tables,
            self._image_column_name,
            self._label_column_name,
            project_name=self._settings.project_name,
            splits=("train", "val"),
        )

        # Get test data if val not present
        if "val" not in self.data:
            data_test = tlc_check_det_dataset(
                self.args.data,
                self._tables,
                self._image_column_name,
                self._label_column_name,
                project_name=self._settings.project_name,
                splits=("test",),
            )
            self.data["test"] = data_test["test"]

        return self.data

    def build_dataset(self, *args, **kwargs):
        from ultralytics.models.yolo.detect.train import build_yolo_dataset as original_build_yolo_dataset

        mode = kwargs.get("mode") or args[1]

        exclude_zero = mode == "val" and self._settings.exclude_zero_weight_collection
        ultralytics.models.yolo.detect.train.build_yolo_dataset = partial(
            build_tlc_yolo_dataset,
            exclude_zero=exclude_zero,
            class_map=self.data["3lc_class_to_range"],
            image_column_name=self._image_column_name,
            label_column_name=self._label_column_name,
        )

        result = DetectionTrainer.build_dataset(self, *args, **kwargs)

        ultralytics.models.yolo.detect.train.build_yolo_dataset = original_build_yolo_dataset

        return result

    def get_validator(self, dataloader=None):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        if not dataloader:
            dataloader = self.test_loader

        return TLCDetectionValidator(
            dataloader,
            save_dir=self.save_dir,
            args=self.args,
            run=self._run,
            image_column_name=self._image_column_name,
            label_column_name=self._label_column_name,
            settings=self._settings,
            training=True,
        )

    def _process_metrics(self, metrics):
        return {
            metric.removesuffix("(B)").replace("metrics", "val").replace("/", "_"): value
            for metric, value in metrics.items()
        }

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""

        sampler = create_sampler(dataset_path, mode, self._settings, distributed=rank != -1)

        # Patch parent class module to use our build_dataloader
        trainer_build_dataloader = ultralytics.models.yolo.detect.train.build_dataloader
        ultralytics.models.yolo.detect.train.build_dataloader = partial(build_dataloader, sampler=sampler)

        dataloader = super().get_dataloader(dataset_path, batch_size, rank, mode)

        # Restore parent class module
        ultralytics.models.yolo.detect.train.build_dataloader = trainer_build_dataloader

        return dataloader
