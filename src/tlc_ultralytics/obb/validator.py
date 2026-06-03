import tlc
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
            "oriented_bbs_2d_predicted": tlc.data_types.OrientedBoundingBoxes2D.schema(
                classes=self.data["names_3lc"],
                include_per_instance_confidence=True,
            )
        }

    def _compute_3lc_metrics(self, preds, batch):
        return {"oriented_bbs_2d_predicted": self._process_predictions(preds, batch)}

    def _build_annotation(self, scaled, mapped_classes, h, w):
        # OrientedBoundingBoxes2D stores all OBBs in a single (N, 5) ndarray.
        return tlc.data_types.OrientedBoundingBoxes2D(
            obbs=scaled["bboxes"].cpu().numpy().astype("float32"),
            labels=[int(c) for c in mapped_classes],
            confidences=scaled["conf"].cpu().numpy().astype("float32").tolist(),
            x_max=w,
            y_max=h,
        )

    def _empty_annotation(self, h, w):
        return tlc.data_types.OrientedBoundingBoxes2D.create_empty(image_width=w, image_height=h)
