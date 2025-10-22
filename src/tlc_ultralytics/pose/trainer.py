from __future__ import annotations

from copy import deepcopy

from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import LOGGER

from tlc_ultralytics.constants import IMAGE_COLUMN_NAME, POSE_LABEL_COLUMN_NAME, TLC_COLORSTR
from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.pose.loss import TLCv8PoseLoss
from tlc_ultralytics.pose.validator import TLCPoseValidator
from tlc_ultralytics.utils.dataset import check_tlc_dataset


# Monkeypatch Ultralytics PoseModel to use TLCv8PoseLoss which respects dataset oks_sigmas
def _tlc_pose_init_criterion(self):
    return TLCv8PoseLoss(self)


# Apply the monkeypatch once at import time
PoseModel.init_criterion = _tlc_pose_init_criterion


class TLCPoseTrainer(PoseTrainer, TLCDetectionTrainer):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = POSE_LABEL_COLUMN_NAME

    def get_dataset(self):
        self.data = check_tlc_dataset(
            self.args.data,
            self._tables,
            self._settings.image_column_name,
            self._settings.label_column_name,
            project_name=self._settings.project_name,
            splits=("train", "val"),
            task="pose",
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
                task="pose",
                settings=self._settings,
            )
            self.data["test"] = data_test["test"]

        return self.data

    def set_model_attributes(self):
        """Set keypoints shape and attach dataset-provided OKS sigmas to the model if available."""
        super().set_model_attributes()
        oks_sigmas = self._settings.oks_sigmas or self.data.get("oks_sigmas")
        if oks_sigmas is not None:
            # Attach to model so TLCv8PoseLoss/v8UnreducedPoseLoss can pick it up
            self.model.oks_sigmas = oks_sigmas

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

    def _process_metrics(self, metrics):
        detection_metrics = super()._process_metrics(metrics)
        return {metric_name.replace("(P)", "_pose"): value for metric_name, value in detection_metrics.items()}

    def _print_task_specific_parameters(self):
        """Print task-specific parameters to the console."""
        table_sigmas = self.data.get("oks_sigmas")

        if table_sigmas is not None:
            table_sigmas_rounded = [round(x, 2) for x in table_sigmas]
            LOGGER.info(f"{TLC_COLORSTR}Using OKS sigmas: {table_sigmas_rounded} from Table for evaluation")
        if self._settings.oks_sigmas is not None:
            settings_oks_rounded = [round(x, 2) for x in self._settings.oks_sigmas]
            LOGGER.info(
                f"{TLC_COLORSTR}Using overridden OKS sigmas: {settings_oks_rounded} from Settings for computing loss"
            )
