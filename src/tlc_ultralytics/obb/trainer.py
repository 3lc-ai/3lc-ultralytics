from copy import deepcopy

from ultralytics.models.yolo.obb.train import OBBTrainer

from tlc_ultralytics.constants import OBB_LABEL_COLUMN_NAME
from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.obb.validator import TLCOBBValidator
from tlc_ultralytics.utils.dataset import check_tlc_dataset


class TLCOBBTrainer(OBBTrainer, TLCDetectionTrainer):
    _default_label_column_name = OBB_LABEL_COLUMN_NAME

    def get_dataset(self):
        # Parse yaml and create tables
        self.data = check_tlc_dataset(
            self.args.data,
            self._tables,
            self._settings.image_column_name,
            self._settings.label_column_name,
            project_name=self._settings.project_name,
            splits=("train", "val"),
            task="obb",
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
                task="obb",
                settings=self._settings,
            )
            self.data["test"] = data_test["test"]

        return self.data

    def _process_metrics(self, metrics):
        detection_metrics = super()._process_metrics(metrics)
        return detection_metrics

    def get_validator(self, dataloader=None):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss"

        if not dataloader:
            dataloader = self.test_loader

        return TLCOBBValidator(
            dataloader,
            save_dir=self.save_dir,
            args=deepcopy(self.args),
            _callbacks=self.callbacks,
            run=self._run,
            settings=self._settings,
            training=True,
        )
