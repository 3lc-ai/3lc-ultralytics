from copy import deepcopy

from ultralytics.models.yolo.obb.train import OBBTrainer

from tlc_ultralytics.constants import OBB_LABEL_COLUMN_NAME
from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.obb.validator import TLCOBBValidator


class TLCOBBTrainer(OBBTrainer, TLCDetectionTrainer):
    _default_label_column_name = OBB_LABEL_COLUMN_NAME

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
