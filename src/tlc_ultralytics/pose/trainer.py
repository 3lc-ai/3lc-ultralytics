from __future__ import annotations

from typing import ClassVar

from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import LOGGER

from tlc_ultralytics.constants import IMAGE_COLUMN_NAME, TLC_COLORSTR
from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.engine.trainer import TLCTrainerMixin
from tlc_ultralytics.pose.loss import apply_oks_sigmas
from tlc_ultralytics.pose.validator import TLCPoseValidator

# Monkeypatch Ultralytics PoseModel so its pose loss respects dataset oks_sigmas
_ultralytics_pose_init_criterion = PoseModel.init_criterion


def _tlc_pose_init_criterion(self):
    return apply_oks_sigmas(_ultralytics_pose_init_criterion(self), self)


# Apply the monkeypatch once at import time
PoseModel.init_criterion = _tlc_pose_init_criterion


class TLCPoseTrainer(PoseTrainer, TLCDetectionTrainer):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _validator_class = TLCPoseValidator
    _metric_replacements: ClassVar[list[tuple[str, str]]] = [
        ("(B)", ""),
        ("(P)", "_pose"),
        ("metrics", "val"),
        ("/", "_"),
    ]

    # Explicit binding to ensure TLCTrainerMixin.get_validator wins over PoseTrainer.get_validator in MRO
    get_validator = TLCTrainerMixin.get_validator

    def set_model_attributes(self):
        """Set keypoints shape and attach dataset-provided OKS sigmas to the model if available."""
        super().set_model_attributes()
        oks_sigmas = self._settings.oks_sigmas or self.data.get("oks_sigmas")
        if oks_sigmas is not None:
            # Attach to model so the pose loss and v8UnreducedPoseLoss can pick it up
            self.model.oks_sigmas = oks_sigmas

    def _print_task_specific_parameters(self):
        """Print task-specific parameters to the console."""
        table_sigmas = self.data.get("oks_sigmas")

        if table_sigmas is not None and isinstance(table_sigmas, list):
            table_sigmas_rounded = [round(x, 2) for x in table_sigmas]  # type: ignore[no-matching-overload]
            LOGGER.info(f"{TLC_COLORSTR}Using OKS sigmas: {table_sigmas_rounded} from Table for evaluation")
        if self._settings.oks_sigmas is not None:
            settings_oks_rounded = [round(x, 2) for x in self._settings.oks_sigmas]
            LOGGER.info(
                f"{TLC_COLORSTR}Using overridden OKS sigmas: {settings_oks_rounded} from Settings for computing loss"
            )
