from __future__ import annotations

from typing import Any

import tlc
import torch
from tlc.data_types import SegmentationMasks
from tlc.schemas import ConfidenceSchema
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import LOGGER, ops

from tlc_ultralytics.constants import (
    CONFIDENCE,
    IMAGE_COLUMN_NAME,
    PREDICTED_SEGMENTATIONS,
    TLC_COLORSTR,
)
from tlc_ultralytics.detect.validator import TLCDetectionValidator
from tlc_ultralytics.engine.validator import PREDICTION_INDEX
from tlc_ultralytics.utils.dataset import check_tlc_dataset


class TLCSegmentationValidator(TLCDetectionValidator, SegmentationValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME

    _mask_chunk_instances = 32
    """Upper bound on how many instances have their masks upsampled at a time in `_masks_at_original_resolution`."""

    _mask_chunk_pixels = 64 * 1024 * 1024
    """Pixel budget per mask chunk, bounding the float32 transient of the upsampling to ~256 MB.

    Chunks are sized from the original image area, so a chunk stays within this budget on large images (a 4K image
    gives 7 instances per chunk) and is capped by `_mask_chunk_instances` on small ones.
    """

    _mask_sources: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    """The current batch's per-image `(prototypes, mask coefficients)`, stashed by `postprocess`."""

    _mask_imgsz: list[int] | None = None
    """The current batch's model-input mask size, derived from the prototype resolution as Ultralytics does."""

    def check_dataset(self, *args, **kwargs):
        return check_tlc_dataset(*args, task="segment", settings=self._settings, **kwargs)

    def postprocess(self, preds: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Post-process predictions, keeping the mask prototypes and coefficients for the 3LC annotation path.

        This mirrors `SegmentationValidator.postprocess`, which pops each image's mask coefficients while turning
        them into `pred["masks"]`. It does that for every NMS survivor — up to `max_det=300` at `conf=0.001` — so
        producing those masks at full model-input resolution costs gigabytes of transients per image. Ultralytics'
        own resolution (the prototype resolution, a quarter of the model input) is left in place for its metrics,
        and the prototypes and coefficients are stashed here so `_scale_filtered_pred` can build full-resolution
        masks for the far smaller set of predictions that survive `Settings.conf_thres` / `Settings.max_det`.
        """
        proto = preds[0][1] if isinstance(preds[0], tuple) else preds[1]

        # Called explicitly rather than through `super()`: the next `postprocess` in the MRO is
        # `SegmentationValidator`'s, which is exactly the mask handling being replaced below. Loss collection's
        # stash of the raw predictions (`TLCDetectionValidator.postprocess`) is kept.
        self._curr_raw_preds = preds if self._settings.collect_loss else None
        outputs = DetectionValidator.postprocess(self, preds[0])

        process = self.process
        assert process is not None, "init_metrics selects the mask processing function and runs before postprocess"

        imgsz = [4 * x for x in proto.shape[2:]]  # get image size from proto
        self._mask_imgsz = imgsz
        self._mask_sources = []
        for i, pred in enumerate(outputs):
            coefficient = pred.pop("extra")
            self._mask_sources.append((proto[i], coefficient))
            pred["masks"] = (
                process(proto[i], coefficient, pred["bboxes"], shape=imgsz)
                if coefficient.shape[0]
                else torch.zeros(
                    (0, *(imgsz if process is ops.process_mask_native else proto.shape[2:])),
                    dtype=torch.uint8,
                    device=pred["bboxes"].device,
                )
            )
        return outputs

    def _scale_filtered_pred(self, i, filtered, pbatch):
        """Scale image `i`'s filtered predictions and build their masks at the original image resolution.

        `pred["masks"]` as produced by Ultralytics covers every NMS survivor at the prototype resolution, which is
        neither the instance set nor the resolution 3LC writes. Masks are therefore generated from the stashed
        prototypes and the filtered instances' mask coefficients, so both the mask generation and the upsampling
        scale with the number of predictions actually written instead of with `max_det=300`.
        """
        proto, coefficients = self._mask_sources[i]

        # `DetectionValidator.scale_preds` scales the boxes only; `SegmentationValidator`'s (the next one in the
        # MRO) would scale the quarter-resolution masks being replaced here in one unchunked allocation.
        scaled = DetectionValidator.scale_preds(self, filtered, pbatch)
        scaled["masks"] = self._masks_at_original_resolution(
            proto,
            coefficients[filtered[PREDICTION_INDEX]],
            filtered["bboxes"],  # model-input coords, as `crop_mask` expects; `scaled["bboxes"]` are not
            pbatch,
        )
        return scaled

    def _masks_at_original_resolution(
        self,
        proto: torch.Tensor,
        coefficients: torch.Tensor,
        bboxes: torch.Tensor,
        pbatch: dict[str, Any],
    ) -> torch.Tensor:
        """Build binary masks for one image's filtered instances at its original resolution.

        Upsampling is chunked (see `_mask_chunk_pixels`) and each chunk is reduced to uint8 before the next is
        allocated: a single `N x H_ori x W_ori` float32 tensor is ~18 GB for 300 instances on a 4K image, which is
        one allocation the CUDA caching allocator cannot be expected to serve, let alone repeatedly at varying
        sizes.
        """
        h, w = pbatch["ori_shape"]
        chunk_size = max(1, min(self._mask_chunk_instances, self._mask_chunk_pixels // max(1, int(h) * int(w))))

        chunks = []
        for start in range(0, coefficients.shape[0], chunk_size):
            stop = start + chunk_size
            native = ops.process_mask_native(
                proto, coefficients[start:stop], bboxes[start:stop], shape=self._mask_imgsz
            )
            chunks.append(ops.scale_masks(native[None], (h, w), ratio_pad=pbatch["ratio_pad"])[0].byte())
            del native

        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

    def _get_metrics_schemas(self) -> dict[str, tlc.Schema]:
        instance_properties_structure = {
            CONFIDENCE: ConfidenceSchema(writable=False),
        }

        segment_schema = SegmentationMasks.schema(
            classes=self.data["names_3lc"],
            per_instance_schemas=instance_properties_structure,
            writable=False,
        )

        # Instance-embedding columns are added by the mixin (raw during streaming,
        # reduced during the end-of-pass rewrite).
        return {PREDICTED_SEGMENTATIONS: segment_schema}

    def _compute_3lc_metrics(self, preds, batch):
        return {PREDICTED_SEGMENTATIONS: self._process_predictions(preds, batch)}

    def _prepare_loss_fn(self, model):
        if self._settings.collect_loss:
            LOGGER.warning(
                f"{TLC_COLORSTR}Per-sample loss collection is not supported for the 'segment' task. "
                "Disabling loss collection for this run."
            )
            self._settings.collect_loss = False

    def _build_annotation(self, scaled, mapped_classes, h, w):
        return tlc.data_types.SegmentationMasks(
            image_height=h,
            image_width=w,
            masks=scaled["masks"].cpu().numpy(),  # PyTorch-native (N, H, W); transposed by mask_format below
            mask_format="nhw",
            labels=mapped_classes,
            confidences=scaled["conf"].tolist(),
        )

    def _empty_annotation(self, h, w):
        return tlc.data_types.SegmentationMasks.create_empty(image_height=h, image_width=w)

    # Instance embeddings pool the feature map with the predicted/GT segmentation masks.
    _instance_geometry_kind = "mask"

    def _instance_regions(self, source, h: int, w: int, device) -> torch.Tensor:
        masks = source.get("masks") if source is not None else None
        if masks is None or masks.numel() == 0:
            return torch.empty((0, h, w), device=device)
        return masks.to(device)
