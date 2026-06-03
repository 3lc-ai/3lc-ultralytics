from __future__ import annotations

import weakref

import tlc
import torch
from ultralytics.models import yolo

from tlc_ultralytics.classify.dataset import TLCClassificationDataset
from tlc_ultralytics.constants import (
    CLASSIFY_LABEL_COLUMN_NAME,
    IMAGE_COLUMN_NAME,
)
from tlc_ultralytics.engine.validator import TLCValidatorMixin
from tlc_ultralytics.utils.dataset import check_tlc_dataset


class TLCClassificationValidator(TLCValidatorMixin, yolo.classify.ClassificationValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = CLASSIFY_LABEL_COLUMN_NAME

    def check_dataset(self, *args, **kwargs):
        return check_tlc_dataset(*args, task="classify", settings=self._settings, **kwargs)

    def build_dataset(self, table):
        return TLCClassificationDataset(
            table=table,
            args=self.args,
            augment=False,
            prefix=self.args.split,
            image_column_name=self._settings.image_column_name,
            label_column_name=self._settings.label_column_name,
            exclude_zero=self._settings.exclude_zero_weight_collection,
            class_map=self.data["3lc_class_to_range"],
        )

    def _get_metrics_schemas(self):

        column_schemas = {
            "loss": tlc.schemas.Float32Schema(display_name="Loss", description="Cross Entropy Loss", writable=False),
            "predicted": tlc.schemas.CategoricalLabelSchema(
                classes=self.data["names_3lc"],
                display_name="Predicted",
                description="The highest confidence class predicted by the model.",
                writable=False,
            ),
            "confidence": tlc.schemas.ConfidenceSchema(
                display_name="Confidence",
                description="The confidence of the prediction",
                writable=False,
            ),
            "top1_accuracy": tlc.schemas.Float32Schema(
                display_name="Top-1 Accuracy",
                description="The correctness of the prediction",
            ),
        }

        if len(self.data["names_3lc"]) > 5:
            column_schemas["top5_accuracy"] = tlc.schemas.Float32Schema(
                display_name="Top-5 Accuracy",
                description="The correctness of any of the top five confidence predictions",
            )

        return column_schemas

    def _compute_3lc_metrics(self, preds, batch):
        """Compute 3LC classification metrics for each sample."""
        confidence, predicted = preds.max(dim=1)

        batch_metrics = {
            "loss": torch.nn.functional.nll_loss(
                torch.log(preds), batch["cls"], reduction="none"
            ),  # nll since preds are normalized
            "predicted": [self.data["range_to_3lc_class"][int(p)] for p in predicted.tolist()],
            "confidence": confidence,
            "top1_accuracy": (torch.argmax(preds, dim=1) == batch["cls"]).to(torch.float32),
        }

        if len(self.dataloader.dataset.table.get_value_map(self._settings.label_column_name)) > 5:
            _, top5_pred = torch.topk(preds, 5, dim=1)
            labels_expanded = batch["cls"].view(-1, 1).expand_as(top5_pred)
            top5_correct = torch.any(top5_pred == labels_expanded, dim=1)
            batch_metrics["top5_accuracy"] = top5_correct.to(torch.float32)

        return batch_metrics

    def _add_embeddings_hook(self, model):
        """Add a hook to extract embeddings from the model, and infer the activation size. For a classification model,
        this amounts to finding the linear layer and extracting the input size."""

        # Unwrap AutoBackend to get the underlying nn.Module.
        # In ultralytics >=8.4.10, AutoBackend delegates to a non-nn.Module backend,
        # so model.modules() no longer recurses into the actual model layers.
        if hasattr(model, "model"):
            inner_model = model.model
        else:
            inner_model = model

        # Find index of the linear layer
        linear_layer_index: int | None = None
        activation_size: int | None = None
        for index, module in enumerate(inner_model.modules()):
            if isinstance(module, torch.nn.Linear):
                activation_size = module.in_features
                linear_layer_index = index
                break

        if linear_layer_index is None or activation_size is None:
            raise ValueError("No linear layer found in model, cannot collect embeddings.")

        weak_self = weakref.ref(self)  # Avoid circular reference (self <-> hook_fn)

        def hook_fn(_module, _input, output):
            # Store embeddings
            self_ref = weak_self()
            embeddings = output.detach().cpu().numpy()
            self_ref.embeddings = embeddings

        # Add forward hook to collect embeddings
        for i, module in enumerate(inner_model.modules()):
            if i == linear_layer_index - 1:
                self._hook_handles.append(module.register_forward_hook(hook_fn))

        return activation_size

    def _infer_batch_size(self, preds, batch) -> int:
        return preds.size(0)
