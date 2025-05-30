# Override specific classes and methods on ultralytics
import ultralytics
from tlc_ultralytics.overrides import build_dataloader

ultralytics.data.build.build_dataloader = build_dataloader

from tlc_ultralytics.classify import TLCClassificationTrainer, TLCClassificationValidator  # noqa: E402
from tlc_ultralytics.detect import TLCDetectionTrainer, TLCDetectionValidator  # noqa: E402
from tlc_ultralytics.engine.model import TLCYOLO  # noqa: E402
from tlc_ultralytics.segment import TLCSegmentationTrainer, TLCSegmentationValidator  # noqa: E402
from tlc_ultralytics.settings import Settings  # noqa: E402

__all__ = [
    "Settings",
    "TLCYOLO",
    "TLCClassificationTrainer",
    "TLCClassificationValidator",
    "TLCDetectionTrainer",
    "TLCDetectionValidator",
    "TLCSegmentationTrainer",
    "TLCSegmentationValidator",
]
