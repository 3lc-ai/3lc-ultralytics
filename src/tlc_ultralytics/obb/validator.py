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
        per_instance_schemas = {
            LABEL: CategoricalLabelListSchema(classes=self.data["names"]),
            CONFIDENCE: Float32ListSchema(),
        }

        emb_schemas = {}
        if self._settings.instance_embeddings_dim > 0:
            from tlc_ultralytics.utils.schemas import instance_embeddings_list_schema

            dim = self._settings.instance_embeddings_dim
            emb_schemas["predicted_instance_embedding"] = instance_embeddings_list_schema(
                dim, display_name=f"Predicted Instance Embedding ({dim}D)"
            )

            if self._settings.ground_truth_instance_embeddings:
                emb_schemas["ground_truth_instance_embedding"] = instance_embeddings_list_schema(
                    dim, display_name=f"Ground Truth Instance Embedding ({dim}D)"
                )

        return {
            "oriented_bbs_2d_predicted": Geometry2DSchema(
                include_2d_oriented_bounding_boxes=True,
                per_instance_schemas=per_instance_schemas,
            ),
            **emb_schemas,
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

    def _extract_instance_embeddings(self, preds, batch) -> list[np.ndarray]:
        """Extract per-instance embeddings for OBB using oriented box masks."""
        from tlc_ultralytics.utils.embeddings import extract_instance_embeddings_mask

        feature_map = self._instance_feature_map

        # Use model-input (letterboxed) coords — bboxes and masks align with the feature map.
        masks_list = []
        image_sizes = []
        for i, pred in enumerate(preds):
            pbatch = self._prepare_batch(i, batch)
            imgsz = pbatch["imgsz"]
            h_in, w_in = int(imgsz[0]), int(imgsz[1])
            image_sizes.append((h_in, w_in))

            mask = pred["conf"] >= self._settings.conf_thres
            if not mask.any():
                masks_list.append(torch.empty((0, h_in, w_in), device=feature_map.device))
                continue

            filtered_bboxes = pred["bboxes"][mask]  # OBB format: cx, cy, w, h, rotation
            filtered_conf = pred["conf"][mask]

            # Keep only top max_det
            max_det = self._settings.max_det
            if len(filtered_conf) > max_det:
                topk = filtered_conf.topk(max_det).indices
                filtered_bboxes = filtered_bboxes[topk]

            obb_masks = _obbs_to_masks(filtered_bboxes, h_in, w_in, device=feature_map.device)
            masks_list.append(obb_masks)

        return extract_instance_embeddings_mask(feature_map, masks_list, image_sizes)

    def _inject_instance_embeddings(self, batch_metrics, reduced_embeddings):
        """Inject reduced predicted instance embeddings as a top-level metric column."""
        pred_embeddings = []
        for emb_array in reduced_embeddings:
            if emb_array.shape[0] > 0:
                pred_embeddings.append([emb_array[i].astype(np.float32).tolist() for i in range(len(emb_array))])
            else:
                pred_embeddings.append([])
        batch_metrics["predicted_instance_embedding"] = pred_embeddings

    def _extract_gt_instance_embeddings(self, preds, batch) -> list[np.ndarray]:
        """Extract per-instance embeddings for ground-truth OBBs using oriented box masks."""
        from tlc_ultralytics.utils.embeddings import extract_instance_embeddings_mask

        feature_map = self._instance_feature_map

        # GT bboxes from _prepare_batch are already in model-input (letterboxed) coords
        masks_list = []
        image_sizes = []
        for i in range(len(preds)):
            pbatch = self._prepare_batch(i, batch)
            imgsz = pbatch["imgsz"]
            h_in, w_in = int(imgsz[0]), int(imgsz[1])
            image_sizes.append((h_in, w_in))

            gt_bboxes = pbatch["bboxes"]  # GT OBBs: cx, cy, w, h, rotation in model-input coords
            if gt_bboxes.numel() == 0:
                masks_list.append(torch.empty((0, h_in, w_in), device=feature_map.device))
            else:
                obb_masks = _obbs_to_masks(gt_bboxes, h_in, w_in, device=feature_map.device)
                masks_list.append(obb_masks)

        return extract_instance_embeddings_mask(feature_map, masks_list, image_sizes)

    def _inject_gt_instance_embeddings(self, batch_metrics, reduced_embeddings):
        """Inject reduced GT instance embeddings as a top-level metric column."""
        gt_embeddings = []
        for emb_array in reduced_embeddings:
            if emb_array.shape[0] > 0:
                gt_embeddings.append([emb_array[i].astype(np.float32).tolist() for i in range(len(emb_array))])
            else:
                gt_embeddings.append([])
        batch_metrics["ground_truth_instance_embedding"] = gt_embeddings


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
