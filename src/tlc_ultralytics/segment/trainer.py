from copy import deepcopy

from ultralytics.models.yolo.segment.train import SegmentationTrainer

from tlc_ultralytics.constants import SEGMENTATION_LABEL_COLUMN_NAME
from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.segment.validator import TLCSegmentationValidator
from tlc_ultralytics.utils.dataset import check_tlc_dataset


class TLCSegmentationTrainer(SegmentationTrainer, TLCDetectionTrainer):
    _default_label_column_name = SEGMENTATION_LABEL_COLUMN_NAME

    def get_dataset(self):
        # Parse yaml and create tables
        self.data = check_tlc_dataset(
            self.args.data,
            self._tables,
            self._settings.image_column_name,
            self._settings.label_column_name,
            project_name=self._settings.project_name,
            splits=("train", "val"),
            task="segment",
            settings=self._settings,
        )

        # Get test data if val not present
        if "val" not in self.data:
            data_test = check_tlc_dataset(
                self.args.data,
                self._tables,
                self._settings.image_column_name,
                self._settings.label_column_name,
                project_name=self._settings.project_name,
                splits=("test",),
                task="segment",
                settings=self._settings,
            )
            self.data["test"] = data_test["test"]

        return self.data

    def _process_metrics(self, metrics):
        detection_metrics = super()._process_metrics(metrics)
        return {metric_name.replace("(M)", "_seg"): value for metric_name, value in detection_metrics.items()}

    def get_validator(self, dataloader=None):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"

        if not dataloader:
            dataloader = self.test_loader

        return TLCSegmentationValidator(
            dataloader,
            save_dir=self.save_dir,
            args=deepcopy(self.args),
            _callbacks=self.callbacks,
            run=self._run,
            settings=self._settings,
            training=True,
        )
