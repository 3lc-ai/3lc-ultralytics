from typing import ClassVar

from ultralytics.models.yolo.segment.train import SegmentationTrainer

from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.engine.trainer import TLCTrainerMixin
from tlc_ultralytics.segment.validator import TLCSegmentationValidator


class TLCSegmentationTrainer(SegmentationTrainer, TLCDetectionTrainer):
    _validator_class = TLCSegmentationValidator
    _loss_names = ("box_loss", "seg_loss", "cls_loss", "dfl_loss")
    _metric_replacements: ClassVar[list[tuple[str, str]]] = [
        ("(B)", ""),
        ("(M)", "_seg"),
        ("metrics", "val"),
        ("/", "_"),
    ]

    # Explicit binding to ensure TLCTrainerMixin.get_validator wins over SegmentationTrainer.get_validator in MRO
    get_validator = TLCTrainerMixin.get_validator
