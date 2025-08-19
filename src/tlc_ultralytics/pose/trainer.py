from copy import deepcopy

from ultralytics.models.yolo.pose.train import PoseTrainer

from tlc_ultralytics.constants import IMAGE_COLUMN_NAME, POSE_LABEL_COLUMN_NAME
from tlc_ultralytics.detect.trainer import TLCTrainerMixin
from tlc_ultralytics.pose.utils import tlc_check_pose_dataset
from tlc_ultralytics.pose.validator import TLCPoseValidator


class TLCPoseTrainer(PoseTrainer, TLCTrainerMixin):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = POSE_LABEL_COLUMN_NAME

    def get_dataset(self):
        self.data = tlc_check_pose_dataset(
            self.args.data,
            self._tables,
            self._image_column_name,
            self._label_column_name,
            project_name=self._settings.project_name,
            splits=("train", "val"),
        )

        if "val" not in self.data:
            data_test = tlc_check_pose_dataset(
                self.args.data,
                self._tables,
                self._image_column_name,
                self._label_column_name,
                project_name=self._settings.project_name,
                splits=("test",),
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
