from __future__ import annotations

from typing import Any

import numpy as np
import tlc
import torch
import torch.nn.functional as F
from tlc.constants import IMAGE_HEIGHT, IMAGE_WIDTH, INSTANCE_PROPERTIES, LABEL, RLES
from ultralytics.models.yolo.semantic.val import SemanticSegmentationValidator
from ultralytics.utils import LOGGER, ops

from tlc_ultralytics.constants import (
    IMAGE_COLUMN_NAME,
    IOU,
    NUM_IMAGES,
    NUM_PIXELS,
    PIXEL_ACCURACY,
    PREDICTED_SEMANTIC_SEGMENTATION,
    TLC_COLORSTR,
)
from tlc_ultralytics.detect.validator import TLCDetectionValidator
from tlc_ultralytics.semantic.dataset import RESIZED_SHAPE, SEMANTIC_SHAPE
from tlc_ultralytics.semantic.utils import (
    IGNORE_INDEX,
    apply_dataset_class_weights,
    check_cityscapes_class_weights,
    dataset_class_weights_message,
)
from tlc_ultralytics.utils.dataset import check_tlc_dataset
from tlc_ultralytics.utils.rle import rles_from_column_major_chunk


class TLCSemanticSegmentationValidator(TLCDetectionValidator, SemanticSegmentationValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME

    _chunk_pixels = 64 * 1024 * 1024
    """Pixel budget per chunk of classes, bounding the transients of building an image's predictions at its original
    resolution: the float32 logits being upsampled (~256 MB) and the uint8 per-class masks being RLE-encoded (~64 MB).

    Chunks are sized from the original image area, so a model with many classes on large images upsamples and encodes a
    few classes at a time instead of materializing `nc x H x W`.
    """

    _logits: torch.Tensor | None = None
    """The current batch's raw logits, `[B, nc, H/8, W/8]`, stashed by `postprocess`."""

    _logged_class_weights: bool = False
    """Whether the class weights of the per-sample losses have been reported, so they are reported once."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._settings.instance_embeddings_dim > 0:
            LOGGER.warning(
                f"{TLC_COLORSTR}Instance embeddings are not supported for the 'semantic' task. "
                "Disabling instance embeddings for this run."
            )
            self._settings.instance_embeddings_dim = 0
            self._settings.ground_truth_instance_embeddings = False

    def check_dataset(self, *args, **kwargs):
        data = check_tlc_dataset(*args, task="semantic", settings=self._settings, **kwargs)
        check_cityscapes_class_weights(self._settings.cityscapes_class_weights, data["nc"])
        return data

    def _verify_model_data_compatibility(self, model_class_names):
        """Verify that the model's classes match the tables' classes, except for the background's name.

        3LC stores only the id of a column's background class, not its name, so the tables' class names call it
        `background` whatever the table's author named it (see `get_semantic_classes`). A model trained with plain
        Ultralytics on the same data has the original name at that index, which is accepted. Every other class must
        have the same name at the same training index, and the model and tables the same number of classes.
        """
        dataset_class_names = self.data["names"]
        if len(model_class_names) != len(dataset_class_names):
            msg = (
                f"The model and data are incompatible. The model was trained on {len(model_class_names)} classes, "
                f"but the data has {len(dataset_class_names)} classes."
            )
            raise ValueError(msg)

        background = self.data.get("semantic_background")
        background_index = None if background is None else self.data["3lc_class_to_range"][background]
        mismatched = {
            index: (model_class_names.get(index), name)
            for index, name in dataset_class_names.items()
            if index != background_index and model_class_names.get(index) != name
        }
        if mismatched:
            details = "; ".join(
                f"{i}: model '{model}', data '{data}'" for i, (model, data) in sorted(mismatched.items())
            )
            msg = (
                "The model was trained on a different set of classes to the classes in the dataset, or the classes are "
                f"in a different order. Differing class names by index: {details}."
            )
            raise ValueError(msg)

    def postprocess(self, preds):
        """Post-process like Ultralytics, keeping the raw logits to build full-resolution predictions from."""
        self._logits = preds[0] if isinstance(preds, (tuple, list)) else preds
        return super().postprocess(preds)

    def _get_metrics_schemas(self) -> dict[str, tlc.Schema]:
        schemas = {
            PREDICTED_SEMANTIC_SEGMENTATION: tlc.schemas.SemanticSegmentationRleSchema(
                classes=self.data["names_3lc"],
                background=self.data["semantic_background"],
                description="Predicted semantic segmentation",
                writable=False,
            ),
        }
        if self._settings.collect_loss:
            schemas["ce_loss"] = tlc.schemas.Float32Schema(description="Cross-entropy loss", writable=False)
            schemas["dice_loss"] = tlc.schemas.Float32Schema(description="Dice loss", writable=False)
            schemas["loss"] = tlc.schemas.Float32Schema(description="Sum of the two losses", writable=False)
        return schemas

    def _prepare_loss_fn(self, model):
        if not self._settings.collect_loss:
            return

        from ultralytics.utils.loss import SemanticSegmentationLoss

        inner_model = model.model if hasattr(model.model, "model") else model
        self.loss_fn = SemanticSegmentationLoss(inner_model)
        # Ultralytics weights the loss from `model.args.data`, which names the dataset only while training through
        # `data=`. Deciding from the tables instead weights the per-sample losses the same in every collection.
        ultralytics_dataset = self.data.get("ultralytics_dataset")
        cityscapes_class_weights = self._settings.cityscapes_class_weights
        apply_dataset_class_weights(self.loss_fn, ultralytics_dataset, cityscapes_class_weights)
        # The losses are computed in float32 whatever the model's precision (see `_per_sample_losses`), so the class
        # weights, which the loss takes in the model's dtype, must be too: a half-precision model (`half=True`) has
        # float16 weights.
        self.loss_fn.float()
        if not self.training and not self._logged_class_weights:  # the trainer logs it when training
            message = dataset_class_weights_message(
                ultralytics_dataset, self.loss_fn.nc, cityscapes_class_weights, "the per-sample cross-entropy loss"
            )
            if message:
                LOGGER.info(f"{TLC_COLORSTR}{message}")
            self._logged_class_weights = True

    def _pre_validation(self, model):
        dataset = self.dataloader.dataset
        self._example_id_to_index = {label["example_id"]: i for i, label in enumerate(dataset.labels)}
        self._index_to_3lc_class = [self.data["range_to_3lc_class"][i] for i in range(len(self.data["names"]))]
        super()._pre_validation(model)

    def _compute_3lc_metrics(self, preds, batch):
        """Build each image's predicted segmentation at its original resolution.

        Like `SemanticSegmentationPredictor`, logits are upsampled to the model input, cropped out of the letterbox
        padding and resized to the original image, then argmaxed. The crop inverts the validation letterbox exactly
        (see `_letterbox_ratio_pad`), so predictions line up with the ground truth pixel for pixel. They equal
        `model.predict`'s where the image scales to the model input by a whole number of pixels, and differ by at most a
        pixel's worth of resampling elsewhere, since the predictor letterboxes the original image directly.
        """
        logits, self._logits = self._logits, None
        if logits.ndim != 4:
            msg = (
                "3LC metrics collection for semantic segmentation needs the model's logits, but got class maps of "
                f"shape {tuple(logits.shape)}, as exported models with the argmax in the graph produce."
            )
            raise ValueError(msg)

        imgsz = tuple(batch["img"].shape[2:])
        dataset = self.dataloader.dataset

        # Off CUDA the predictions are built on the host, where they are headed anyway. On MPS, upsampling is an order
        # of magnitude slower than on the CPU of the same machine.
        map_logits = logits if logits.is_cuda else logits.cpu()

        predictions = []
        for i, example_id in enumerate(batch["example_id"]):
            index = self._example_id_to_index[int(example_id)]
            h, w = dataset.labels[index][SEMANTIC_SHAPE]
            ratio_pad = self._letterbox_ratio_pad(tuple(int(x) for x in batch[RESIZED_SHAPE][i]), imgsz)
            class_map = self._class_map_at_original_resolution(map_logits[i], imgsz, (h, w), ratio_pad)
            predictions.append(self._prediction_row(class_map, h, w))

        results = {PREDICTED_SEMANTIC_SEGMENTATION: predictions}
        if self._settings.collect_loss:
            results.update(self._per_sample_losses(logits, batch))
        return results

    @staticmethod
    def _letterbox_ratio_pad(
        resized_shape: tuple[int, int], imgsz: tuple[int, int]
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """The `ratio_pad` that makes `ops.scale_masks` crop exactly the image content out of a validation letterbox.

        Reproduces the padding `LetterBox` (centered, no upscaling, as Ultralytics validates) adds to an image resized
        to `resized_shape` on load, with the same rounding. Without it `scale_masks` derives the padding from the
        original shape, which misses the content by a row or column wherever the load-time resize rounded up.
        """
        h, w = resized_shape
        r = min(imgsz[0] / h, imgsz[1] / w, 1.0)
        return (r, r), ((imgsz[1] - round(w * r)) / 2, (imgsz[0] - round(h * r)) / 2)

    def _class_map_at_original_resolution(
        self,
        logits: torch.Tensor,
        imgsz: tuple[int, int],
        ori_shape: tuple[int, int],
        ratio_pad: tuple[tuple[float, float], tuple[float, float]] | None,
    ) -> torch.Tensor:
        """Predict one image's `(H, W)` uint8 class indices at its original resolution from its `[nc, h, w]` logits.

        Both resizes are per-class linear interpolations, so the classes are processed in chunks (see `_chunk_pixels`)
        with a running maximum over them, which gives the argmax over all classes without upsampling them all at once.
        The running maximum is kept a class at a time, so beyond the chunk the full-resolution state is a float32 score
        and a uint8 class index per pixel, never the int64 indices of `max` or `argmax`.
        """
        h, w = ori_shape
        num_classes = logits.shape[0]
        chunk_size = max(1, self._chunk_pixels // max(h * w, imgsz[0] * imgsz[1]))

        best_score = torch.full((h, w), float("-inf"), device=logits.device)
        best_class = torch.zeros((h, w), dtype=torch.uint8, device=logits.device)
        for start in range(0, num_classes, chunk_size):
            chunk = logits[start : start + chunk_size][None].float()
            if chunk.shape[2:] != imgsz:  # upsample to the model input first, so the letterbox padding is integer
                chunk = F.interpolate(chunk, imgsz, mode="bilinear", align_corners=False)
            chunk = ops.scale_masks(chunk, (h, w), ratio_pad=ratio_pad)[0]  # crop the padding, resize to the original
            for offset, score in enumerate(chunk):
                better = score > best_score  # strict, so ties keep the lower class index, as argmax does
                best_class.masked_fill_(better, start + offset)
                torch.maximum(best_score, score, out=best_score)
            del chunk
        return best_class

    def _prediction_row(self, class_map: torch.Tensor, h: int, w: int) -> dict[str, Any]:
        """Return one image's predicted segmentation in the metrics column's row form, RLE-encoded on its device.

        This is exactly what the column's `SemanticSegmentationRleSchema` makes of the dense label map in 3LC class ids:
        one RLE per class present in ascending id order, without the background, which 3LC restores as the fill on
        read. The metrics writer stores values its sample type does not claim unchanged, so handing it the row form
        means the dense map is never copied to the host. The per-class masks are built and encoded a chunk of classes at
        a time, transposed to `(n, W, H)`, column-major per mask, the order RLE counts runs in.
        """
        background = self.data["semantic_background"]
        present = torch.bincount(class_map.flatten(), minlength=len(self._index_to_3lc_class)).nonzero().flatten()
        classes = sorted(
            (self._index_to_3lc_class[i], i) for i in present.tolist() if self._index_to_3lc_class[i] != background
        )

        transposed = class_map.t().contiguous()  # (W, H)
        chunk_size = max(1, self._chunk_pixels // max(1, h * w))
        rles: list[dict[str, Any]] = []
        for start in range(0, len(classes), chunk_size):
            indices = torch.tensor(
                [i for _, i in classes[start : start + chunk_size]], dtype=torch.uint8, device=class_map.device
            )
            chunk = (transposed[None] == indices[:, None, None]).to(torch.uint8)  # (n, W, H)
            rles.extend(rles_from_column_major_chunk(chunk, h, w))
            del chunk

        return {
            IMAGE_HEIGHT: h,
            IMAGE_WIDTH: w,
            INSTANCE_PROPERTIES: {LABEL: [class_id for class_id, _ in classes]},
            RLES: [rle["counts"] for rle in rles],
        }

    def _per_sample_losses(self, logits: torch.Tensor, batch: dict[str, Any]) -> dict[str, list[float]]:
        """Compute Ultralytics' semantic segmentation loss for each image on its own, as a batch of one.

        The logits are cast to float32: Ultralytics runs `update_metrics` outside its autocast, so during AMP training
        they arrive as float16, which the cross-entropy rejects next to its float32 class weights.
        """
        losses: dict[str, list[float]] = {"ce_loss": [], "dice_loss": [], "loss": []}
        for i in range(logits.shape[0]):
            mask = batch["semantic_mask"][i : i + 1]
            if not (mask != IGNORE_INDEX).any():
                # An image with no pixel to learn from (all void) has no loss. Ultralytics' dice term is 0 for it, but
                # its cross-entropy averages over zero pixels, which is NaN.
                ce_loss = dice_loss = 0.0
            else:
                _, loss_items = self.loss_fn(logits[i : i + 1].float(), {"semantic_mask": mask})
                ce_loss, dice_loss = loss_items[:2].tolist()  # the auxiliary loss is zero outside training
            losses["ce_loss"].append(ce_loss)
            losses["dice_loss"].append(dice_loss)
            losses["loss"].append(ce_loss + dice_loss)
        return losses

    def _per_class_base_schemas(self) -> dict[str, tlc.Schema]:
        """The base per-class schemas, with the classes named as in the tables rather than the model.

        A model trained with plain Ultralytics may name the background differently from the tables (see
        `_verify_model_data_compatibility`). The per-class table names it as the tables do, like the input table and
        the predicted segmentation column.
        """
        schemas = super()._per_class_base_schemas()
        schemas[LABEL] = tlc.schemas.CategoricalLabelSchema(classes={**self.data["names"], self.nc: "all"})
        return schemas

    def _per_class_metrics_schemas(self):
        return {
            **self._per_class_base_schemas(),
            NUM_PIXELS: tlc.schemas.Int64Schema(description="Number of non-void ground truth pixels of the class"),
            IOU: tlc.schemas.Float32Schema(description="IoU of the class, mIoU for 'all'"),
            PIXEL_ACCURACY: tlc.schemas.Float32Schema(
                description="Fraction of the class' pixels predicted correctly, overall pixel accuracy for 'all'"
            ),
        }

    def _per_class_counts(self):
        # Ultralytics accumulates the confusion matrix as float32 and reports pixel counts as int32, both of which lose
        # precision on large datasets, so the pixel counts are summed in float64 here.
        pixels = self.metrics.matrix.cpu().double().sum(dim=1).numpy().astype(np.int64)
        return {
            NUM_PIXELS: np.append(pixels, pixels.sum()),
            NUM_IMAGES: np.append(np.asarray(self.metrics.nt_per_image), [int(self.seen)]),
        }

    def _generate_per_class_metrics(self):
        return {
            IOU: np.append(np.asarray(self.metrics.per_class_iou), [self.metrics.miou]),
            PIXEL_ACCURACY: np.append(np.asarray(self.metrics.per_class_pixel_accuracy), [self.metrics.pixel_accuracy]),
        }
