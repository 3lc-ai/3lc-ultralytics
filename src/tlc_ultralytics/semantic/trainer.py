from typing import ClassVar

from ultralytics.models.yolo.semantic.train import SemanticSegmentationTrainer

from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.engine.trainer import TLCTrainerMixin
from tlc_ultralytics.semantic.validator import TLCSemanticSegmentationValidator


class TLCSemanticSegmentationTrainer(SemanticSegmentationTrainer, TLCDetectionTrainer):
    """Trainer class for YOLO semantic segmentation with 3LC"""

    _validator_class = TLCSemanticSegmentationValidator
    _loss_names = ("ce_loss", "dice_loss", "aux_loss")
    _metric_replacements: ClassVar[list[tuple[str, str]]] = [("metrics", "val"), ("/", "_")]

    # Explicit bindings to ensure the TLCTrainerMixin methods win over SemanticSegmentationTrainer's in the MRO. Its
    # `get_dataset` adds a background class for polygon datasets, which a 3LC table declares itself.
    get_validator = TLCTrainerMixin.get_validator
    get_dataset = TLCTrainerMixin.get_dataset
