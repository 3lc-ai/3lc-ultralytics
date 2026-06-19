from __future__ import annotations

from typing import TYPE_CHECKING

import tlc
import ultralytics
import ultralytics.utils.checks
from ultralytics.models import yolo
from ultralytics.models.yolo.model import YOLO as YOLOBase
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel
from ultralytics.utils import LOGGER

from tlc_ultralytics.classify import (
    TLCClassificationTrainer,
    TLCClassificationValidator,
)
from tlc_ultralytics.constants import DEFAULT_COLLECT_RUN_DESCRIPTION
from tlc_ultralytics.detect import TLCDetectionTrainer, TLCDetectionValidator
from tlc_ultralytics.obb import TLCOBBTrainer, TLCOBBValidator
from tlc_ultralytics.pose import TLCPoseTrainer, TLCPoseValidator
from tlc_ultralytics.segment import TLCSegmentationTrainer, TLCSegmentationValidator
from tlc_ultralytics.settings import Settings
from tlc_ultralytics.utils import check_requirements, reduce_embeddings

if TYPE_CHECKING:
    from collections.abc import Iterable


class YOLO(YOLOBase):
    """YOLO (You Only Look Once) object detection model with 3LC integration."""

    def __init__(self, *args, **kwargs):
        """Initialize YOLO model with 3LC integration. Checks that the installed version of 3LC is compatible."""

        check_requirements()

        super().__init__(*args, **kwargs)

    def train(self, *args, **kwargs):
        """Train the model."""

        # Patch the check_pip_update_available function to avoid prompting for an update
        def check_pip_update_available_return_false():
            return False

        ultralytics_check_pip_update_available = ultralytics.utils.checks.check_pip_update_available
        ultralytics.utils.checks.check_pip_update_available = check_pip_update_available_return_false

        # Ensure 'model' key exists in overrides (may be cleared after previous train() call)
        if "model" not in self.overrides:
            self.overrides["model"] = self.model_name

        output = super().train(*args, **kwargs)

        # Restore the original function
        ultralytics.utils.checks.check_pip_update_available = ultralytics_check_pip_update_available

        return output

    @property
    def task_map(self):
        """Map head to 3LC model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": TLCDetectionTrainer,
                "validator": TLCDetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "classify": {
                "model": ClassificationModel,
                "trainer": TLCClassificationTrainer,
                "validator": TLCClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": TLCSegmentationTrainer,
                "validator": TLCSegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": TLCPoseTrainer,
                "validator": TLCPoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": TLCOBBTrainer,
                "validator": TLCOBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }

    def collect(
        self,
        data: str | None = None,
        splits: Iterable[str] | None = None,
        tables: dict[str, str | tlc.Url | tlc.Table] | None = None,
        settings: Settings | None = None,
        progress_callback: object | None = None,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """Perform calls to model.val() to collect metrics on a set of splits, all under one tlc.Run.

        If enabled, embeddings are reduced at the end of validation. When instance
        embeddings are enabled and multiple splits are collected, the first split
        (train) defines the reduction space and subsequent splits are projected
        into it.

        :param data: Path to a YOLO or 3LC YAML file. If provided, splits must also be provided.
        :param splits: List of splits to collect metrics for. If provided, data must also be provided.
        :param tables: Dictionary of splits to tables to collect metrics for. Mutually exclusive with data and splits.
        :param settings: 3LC settings to use for collecting metrics. If None, default settings are used.
        :param progress_callback: Optional callable(phase, current, total) for reporting reduction progress.
        :param kwargs: Additional keyword arguments are forwarded as model.val(**kwargs).
        :return: Dictionary of split names to results returned by model.val().
        """
        from tlc_ultralytics.constants import TLC_COLORSTR

        # Verify only data+splits or tables are provided
        if not ((data and splits) or tables):
            raise ValueError("Either data and splits or tables must be provided to collect.")

        if settings is None:
            settings = Settings()

        if not settings.run_description:
            settings.run_description = DEFAULT_COLLECT_RUN_DESCRIPTION

        # TEMP(instance-embeddings): stash the progress callback on settings so the
        # in-process reducer can report fit/transform phases. Remove when native
        # 3LC reduction of variable-length embedding list columns ships upstream.
        if progress_callback is not None:
            settings._reduction_progress_callback = progress_callback

        # Build a uniform {split: val_kwargs} mapping so both the data+splits and
        # tables branches share a single iteration loop.
        if data and splits:
            split_val_kwargs = {s: {"data": data, "split": s} for s in splits}
        else:
            assert tables is not None
            split_val_kwargs = {s: {"table": t} for s, t in tables.items()}

        # TEMP(instance-embeddings): run the train split first so its fitted
        # reducer (shared via the per-run registry in _instance_reduce) is
        # reused when transforming subsequent splits. Remove the ordering when
        # upstream reduction lands — 3LC's native flow handles cross-split
        # fit/transform itself.
        ordered_splits = sorted(split_val_kwargs, key=lambda s: 0 if s == "train" else 1)

        results_dict = {}
        try:
            for split in ordered_splits:
                LOGGER.info(TLC_COLORSTR + f"Collecting metrics for split: {split}")
                if progress_callback:
                    progress_callback("split_start", 0, 0)
                results_dict[split] = self.val(settings=settings, **split_val_kwargs[split], **kwargs)

            if settings.image_embeddings_dim > 0:
                self._reduce_image_embeddings(settings, progress_callback)
        finally:
            # TEMP(instance-embeddings): drop this run's fitted reducer (also on
            # failure — a later collect() may reuse the same active run and must
            # not inherit a stale embedding space).
            if settings.instance_embeddings_dim > 0 and tlc.active_run() is not None:
                from tlc_ultralytics.utils._instance_reduce import _clear_fitted_reducer

                _clear_fitted_reducer(tlc.active_run().url.to_str())

        tlc.active_run().set_status_completed()

        return results_dict

    @staticmethod
    def _reduce_image_embeddings(settings: Settings, progress_callback: object | None) -> None:
        """Reduce image embeddings server-side across all collected splits."""
        if progress_callback:
            progress_callback("image_embeddings", 0, 0)

        reduce_embeddings(
            tlc.active_run(),
            method=settings.image_embeddings_reducer,
            n_components=settings.image_embeddings_dim,
            reducer_args=settings.image_embeddings_reducer_args,
        )

        if progress_callback:
            progress_callback("image_embeddings_done", 0, 0)


class TLCYOLO(YOLO):
    def __init__(self, *args, **kwargs):
        LOGGER.warning("TLCYOLO is deprecated and will be removed in a future version. Use YOLO instead.")

        super().__init__(*args, **kwargs)
