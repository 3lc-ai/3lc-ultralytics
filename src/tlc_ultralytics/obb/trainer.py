from ultralytics.models.yolo.obb.train import OBBTrainer

from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.engine.trainer import TLCTrainerMixin
from tlc_ultralytics.obb.validator import TLCOBBValidator


class TLCOBBTrainer(OBBTrainer, TLCDetectionTrainer):
    _validator_class = TLCOBBValidator

    # Explicit binding to ensure TLCTrainerMixin.get_validator wins over OBBTrainer.get_validator in MRO
    get_validator = TLCTrainerMixin.get_validator
