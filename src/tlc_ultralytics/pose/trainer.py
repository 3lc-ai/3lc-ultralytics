from __future__ import annotations

from copy import deepcopy

from ultralytics.models.yolo.pose.train import PoseTrainer

from tlc_ultralytics.constants import IMAGE_COLUMN_NAME, POSE_LABEL_COLUMN_NAME
from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.pose.validator import TLCPoseValidator
from tlc_ultralytics.utils.dataset import check_tlc_dataset


class TLCPoseTrainer(PoseTrainer, TLCDetectionTrainer):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = POSE_LABEL_COLUMN_NAME

    def get_dataset(self):
        self.data = check_tlc_dataset(
            self.args.data,
            self._tables,
            self._image_column_name,
            self._label_column_name,
            project_name=self._settings.project_name,
            splits=("train", "val"),
            task="pose",
        )

        # Get test data if val not present
        if "val" not in self.data:
            data_test = check_tlc_dataset(
                self.args.data,
                self._tables,
                self._image_column_name,
                self._label_column_name,
                project_name=self._settings.project_name,
                splits=("test",),
                task="pose",
            )
            self.data["test"] = data_test["test"]

        return self.data

    def get_validator(self, dataloader=None):
        self.loss_names = ("box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss")
        if not dataloader:
            dataloader = self.test_loader

        return TLCPoseValidator(
            dataloader,
            save_dir=self.save_dir,
            args=deepcopy(self.args),
            _callbacks=self.callbacks,
            run=self._run,
            settings=self._settings,
            training=True,
        )
