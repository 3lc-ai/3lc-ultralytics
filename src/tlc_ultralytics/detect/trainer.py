from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionTrainer

from tlc_ultralytics.overrides import build_dataloader
from ultralytics.utils import LOGGER
from tlc_ultralytics.constants import (
    IMAGE_COLUMN_NAME,
    DETECTION_LABEL_COLUMN_NAME,
)
from tlc_ultralytics.detect.utils import (
    build_tlc_yolo_dataset,
    tlc_check_det_dataset,
)
from tlc_ultralytics.detect.validator import TLCDetectionValidator
from tlc_ultralytics.engine.trainer import TLCTrainerMixin
from tlc_ultralytics.utils import create_sampler
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


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

    def build_dataset(self, table, mode="train", batch=None):
        # Dataset object for training / validation
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        exclude_zero = mode == "val" and self._settings.exclude_zero_weight_collection
        return build_tlc_yolo_dataset(
            self.args,
            table,
            batch,
            self.data,
            mode=mode,
            rect=mode == "val",
            stride=gs,
            exclude_zero=exclude_zero,
            class_map=self.data["3lc_class_to_range"],
            image_column_name=self._image_column_name,
            label_column_name=self._label_column_name,
        )

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
        )

    def _process_metrics(self, metrics):
        return {
            metric.removesuffix("(B)").replace("metrics", "val").replace("/", "_"): value
            for metric, value in metrics.items()
        }

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2

        sampler = create_sampler(dataset.table, mode, self._settings, distributed=rank != -1)

        return build_dataloader(dataset, batch_size, workers, shuffle, rank, sampler)  # return dataloader
