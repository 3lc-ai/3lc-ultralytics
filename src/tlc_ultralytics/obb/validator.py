import numpy as np
import tlc
import torch
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
from ultralytics.utils import ops

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
                include_2d_oriented_bounding_boxes=True,
                per_instance_schemas={
                    LABEL: CategoricalLabelListSchema(classes=self.data["names"]),
                    CONFIDENCE: Float32ListSchema(),
                },
            )
        }

    def _compute_3lc_metrics(self, preds, batch) -> dict[str, list[dict[str, any]]]:
        """Compute 3LC metrics for instance segmentation.

        :param preds: Predictions returned by YOLO segmentation model.
        :param batch: Batch of data presented to the YOLO segmentation model.
        :returns: Metrics dict with predicted instance data for each sample in a batch.
        """
        predicted = []

        for i, pred in enumerate(preds):
            predicted_bboxes, predicted_classes, predicted_confidences = (
                pred["bboxes"].clone(),
                pred["cls"].clone(),
                pred["conf"].clone(),
            )
            h, w = batch["ori_shape"][i]
            mask = predicted_confidences >= self._settings.conf_thres

            if len(pred) == 0 or not torch.any(mask):
                predicted.append(
                    {
                        X_MIN: 0,
                        Y_MIN: 0,
                        X_MAX: w,
                        Y_MAX: h,
                        INSTANCES: [],
                        INSTANCES_ADDITIONAL_DATA: {LABEL: [], CONFIDENCE: []},
                    }
                )
                continue

            predicted_bboxes = predicted_bboxes[mask]
            predicted_classes = predicted_classes[mask].cpu().numpy().astype(np.int32).tolist()
            predicted_confidences = predicted_confidences[mask].cpu().numpy().astype(np.float32).tolist()

            resized_shape = batch["resized_shape"][i]
            ori_shape = batch["ori_shape"][i]
            ratio_pad = batch["ratio_pad"][i]
            scaled_bboxes = ops.scale_boxes(resized_shape, predicted_bboxes, ori_shape, ratio_pad, xywh=True)

            instances = []
            for j in range(len(predicted_bboxes)):
                predicted_bbox = scaled_bboxes[j].cpu().numpy().astype(np.float32).tolist()
                instances.append(
                    {
                        "oriented_bbs_2d": [
                            {
                                "center_x": predicted_bbox[0],
                                "center_y": predicted_bbox[1],
                                "size_x": predicted_bbox[2],
                                "size_y": predicted_bbox[3],
                                "rotation": predicted_bbox[4],
                            }
                        ],
                    }
                )

            predicted.append(
                {
                    X_MIN: 0,
                    Y_MIN: 0,
                    X_MAX: w,
                    Y_MAX: h,
                    INSTANCES: instances,
                    INSTANCES_ADDITIONAL_DATA: {LABEL: predicted_classes, CONFIDENCE: predicted_confidences},
                }
            )

        return {"oriented_bbs_2d_predicted": predicted}
