from ultralytics.models.yolo.obb.train import OBBTrainer

from tlc_ultralytics.constants import OBB_LABEL_COLUMN_NAME
from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.engine.trainer import TLCTrainerMixin
from tlc_ultralytics.obb.validator import TLCOBBValidator


class TLCOBBTrainer(OBBTrainer, TLCDetectionTrainer):
    _default_label_column_name = OBB_LABEL_COLUMN_NAME
    _validator_class = TLCOBBValidator
    _loss_names = ("box_loss", "seg_loss", "cls_loss", "dfl_loss")

    # Explicit binding to ensure TLCTrainerMixin.get_validator wins over OBBTrainer.get_validator in MRO
    get_validator = TLCTrainerMixin.get_validator
