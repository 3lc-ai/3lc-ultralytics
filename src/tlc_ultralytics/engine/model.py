from __future__ import annotations

from collections.abc import Iterable

import tlc
import ultralytics
from ultralytics.models import yolo
from ultralytics.models.yolo.model import YOLO as YOLOBase
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, SegmentationModel
from ultralytics.utils import LOGGER

from tlc_ultralytics.classify import (
    TLCClassificationTrainer,
    TLCClassificationValidator,
)
from tlc_ultralytics.constants import DEFAULT_COLLECT_RUN_DESCRIPTION
from tlc_ultralytics.detect import TLCDetectionTrainer, TLCDetectionValidator
from tlc_ultralytics.segment import (
    TLCSegmentationTrainer,
    TLCSegmentationValidator,
)
from tlc_ultralytics.settings import Settings
from tlc_ultralytics.utils import check_requirements, reduce_embeddings


class YOLO(YOLOBase):
    """YOLO (You Only Look Once) object detection model with 3LC integration."""

    def __init__(self, *args, **kwargs):
        """Initialize YOLO model with 3LC integration. Checks that the installed version of 3LC is compatible."""

        check_requirements()

        super().__init__(*args, **kwargs)

    def train(self, *args, **kwargs):
        """Train the model."""

        # Patch the check_pip_update_available function to avoid prompting for an update
        def check_pip_update_avaliable_return_false():
            return False

        ultralytics_check_pip_update_available = ultralytics.utils.checks.check_pip_update_available
        ultralytics.utils.checks.check_pip_update_available = check_pip_update_avaliable_return_false

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
        }

    def collect(
        self,
        data: str | None = None,
        splits: Iterable[str] | None = None,
        tables: dict[str, str | tlc.Url | tlc.Table] | None = None,
        settings: Settings | None = None,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """Perform calls to model.val() to collect metrics on a set of splits, all under one tlc.Run.
        If enabled, embeddings are reduced at the end of validation.

        :param data: Path to a YOLO or 3LC YAML file. If provided, splits must also be provided.
        :param splits: List of splits to collect metrics for. If provided, data must also be provided.
        :param tables: Dictionary of splits to tables to collect metrics for. Mutually exclusive with data and splits.
        :param settings: 3LC settings to use for collecting metrics. If None, default settings are used.
        :param kwargs: Additional keyword arguments are forwarded as model.val(**kwargs).
        :return: Dictionary of split names to results returned by model.val().
        """
        # Verify only data+splits or tables are provided
        if not ((data and splits) or tables):
            raise ValueError("Either data and splits or tables must be provided to collect.")

        if settings is None:
            settings = Settings()

        settings.verify(training=False)

        if not settings.run_description:
            settings.run_description = DEFAULT_COLLECT_RUN_DESCRIPTION

        results_dict = {}
        # Call val for each split or table
        if data and splits:
            for split in splits:
                results_dict[split] = self.val(data=data, split=split, settings=settings, **kwargs)
        elif tables:
            for split in tables:
                results_dict[split] = self.val(table=tables[split], settings=settings, **kwargs)

        # Reduce embeddings
        if settings and settings.image_embeddings_dim > 0:
            # TODO: Allow user to pass in preferred foreign_table_url

            reduce_embeddings(
                tlc.active_run(),
                method=settings.image_embeddings_reducer,
                n_components=settings.image_embeddings_dim,
                reducer_args=settings.image_embeddings_reducer_args,
            )

        tlc.active_run().set_status_completed()

        return results_dict


class TLCYOLO(YOLO):
    def __init__(self, *args, **kwargs):
        LOGGER.warning("TLCYOLO is deprecated and will be removed in a future version. Use YOLO instead.")

        super().__init__(*args, **kwargs)
