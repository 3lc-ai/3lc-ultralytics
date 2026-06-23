import numpy as np
import tlc
import torch
from ultralytics.models.yolo.obb.val import OBBValidator

from tlc_ultralytics.constants import (
    IMAGE_COLUMN_NAME,
)
from tlc_ultralytics.detect.validator import TLCDetectionValidator
from tlc_ultralytics.utils.dataset import check_tlc_dataset


class TLCOBBValidator(TLCDetectionValidator, OBBValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME

    def check_dataset(self, *args, **kwargs):
        return check_tlc_dataset(*args, task="obb", settings=self._settings, **kwargs)

    def _get_metrics_schemas(self) -> dict[str, tlc.Schema]:
        # Instance-embedding columns are added by the mixin (raw during streaming,
        # reduced during the end-of-pass rewrite).
        return {
            "oriented_bbs_2d_predicted": tlc.data_types.OrientedBoundingBoxes2D.schema(
                classes=self.data["names_3lc"],
                include_per_instance_confidence=True,
            ),
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

    # Instance embeddings pool the feature map with rasterized oriented-box masks.
    _instance_geometry_kind = "mask"

    def _instance_regions(self, source, h: int, w: int, device) -> torch.Tensor:
        # OBBs (cx, cy, w, h, rotation) in model-input coords, both for preds and GT.
        obbs = source.get("bboxes") if source is not None else None
        if obbs is None or obbs.numel() == 0:
            return torch.empty((0, h, w), device=device)
        return _obbs_to_masks(obbs, h, w, device=device)


def _obbs_to_masks(
    obbs: torch.Tensor,
    height: int,
    width: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Rasterize oriented bounding boxes into binary masks.

    Args:
        obbs: [N, 5+] tensor with columns (cx, cy, w, h, rotation, ...).
              Rotation is in radians.
        height: Output mask height (pixels).
        width: Output mask width (pixels).
        device: Target device for the masks.

    Returns:
        [N, height, width] float tensor with 1 inside the OBB, 0 outside.
    """
    import cv2

    n = obbs.shape[0]
    if n == 0:
        return torch.empty((0, height, width), device=device)

    obbs_cpu = obbs[:, :5].detach().cpu().numpy().astype(np.float64)
    masks = np.zeros((n, height, width), dtype=np.float32)

    for i in range(n):
        cx, cy, bw, bh, angle = obbs_cpu[i]
        # cv2.boxPoints expects ((cx, cy), (w, h), angle_degrees)
        rect = ((cx, cy), (bw, bh), np.degrees(angle))
        pts = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillConvexPoly(masks[i], pts, 1.0)

    return torch.from_numpy(masks).to(device=device)
