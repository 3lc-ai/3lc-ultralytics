import numpy as np
import tlc
from tlc.core.builtins.constants import (
    CONFIDENCE,
    INSTANCES,
    INSTANCES_ADDITIONAL_DATA,
    LABEL,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
)
from tlc.core.builtins.schemas import CategoricalLabelListSchema, Float32ListSchema, Geometry2DSchema
from ultralytics.models.yolo.obb.val import OBBValidator

from tlc_ultralytics.constants import (
    IMAGE_COLUMN_NAME,
    OBB_LABEL_COLUMN_NAME,
)
from tlc_ultralytics.detect.validator import TLCDetectionValidator
from tlc_ultralytics.utils.dataset import check_tlc_dataset


class TLCOBBValidator(TLCDetectionValidator, OBBValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = OBB_LABEL_COLUMN_NAME

    def check_dataset(self, *args, **kwargs):
        return check_tlc_dataset(*args, task="obb", settings=self._settings, **kwargs)

    def _get_metrics_schemas(self) -> dict[str, tlc.Schema]:
        return {
            "oriented_bbs_2d_predicted": Geometry2DSchema(
                per_instance_schemas={
                    LABEL: CategoricalLabelListSchema(classes=self.data["names"]),
                    CONFIDENCE: Float32ListSchema(),
                },
            )
        }

    def _compute_3lc_metrics(self, preds, batch):
        return {"oriented_bbs_2d_predicted": self._process_predictions(preds, batch)}

    def _build_annotation(self, scaled, mapped_classes, h, w):
        instances = []
        for j in range(len(mapped_classes)):
            bb = scaled["bboxes"][j].cpu().numpy().astype(np.float32).tolist()
            instances.append(
                {
                    "oriented_bbs_2d": [
                        {
                            "center_x": bb[0],
                            "center_y": bb[1],
                            "size_x": bb[2],
                            "size_y": bb[3],
                            "rotation": bb[4],
                        }
                    ],
                }
            )
        return {
            X_MIN: 0,
            Y_MIN: 0,
            X_MAX: w,
            Y_MAX: h,
            INSTANCES: instances,
            INSTANCES_ADDITIONAL_DATA: {
                LABEL: [int(c) for c in mapped_classes],
                CONFIDENCE: scaled["conf"].cpu().numpy().astype(np.float32).tolist(),
            },
        }

    def _empty_annotation(self, h, w):
        return {
            X_MIN: 0,
            Y_MIN: 0,
            X_MAX: w,
            Y_MAX: h,
            INSTANCES: [],
            INSTANCES_ADDITIONAL_DATA: {LABEL: [], CONFIDENCE: []},
        }
