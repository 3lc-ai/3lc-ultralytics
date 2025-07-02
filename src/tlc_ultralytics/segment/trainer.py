from copy import deepcopy

from ultralytics.models.yolo.segment.train import SegmentationTrainer

from tlc_ultralytics.constants import SEGMENTATION_LABEL_COLUMN_NAME
from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.segment.utils import tlc_check_seg_dataset
from tlc_ultralytics.segment.validator import TLCSegmentationValidator


class TLCSegmentationTrainer(SegmentationTrainer, TLCDetectionTrainer):
    _default_label_column_name = SEGMENTATION_LABEL_COLUMN_NAME

    def get_dataset(self):
        # Parse yaml and create tables
        self.data = tlc_check_seg_dataset(
            self.args.data,
            self._tables,
            self._image_column_name,
            self._label_column_name,
            project_name=self._settings.project_name,
            splits=("train", "val"),
        )

        # Get test data if val not present
        if "val" not in self.data:
            data_test = tlc_check_seg_dataset(
                self.args.data,
                self._tables,
                self._image_column_name,
                self._label_column_name,
                project_name=self._settings.project_name,
                splits=("test",),
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
