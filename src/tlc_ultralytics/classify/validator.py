from __future__ import annotations

import weakref

import tlc
import torch
from ultralytics.models import yolo

from tlc_ultralytics.classify.dataset import TLCClassificationDataset
from tlc_ultralytics.classify.utils import tlc_check_cls_dataset
from tlc_ultralytics.constants import (
    CLASSIFY_LABEL_COLUMN_NAME,
    IMAGE_COLUMN_NAME,
)
from tlc_ultralytics.engine.validator import TLCValidatorMixin


class TLCClassificationValidator(TLCValidatorMixin, yolo.classify.ClassificationValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = CLASSIFY_LABEL_COLUMN_NAME

    def check_dataset(self, *args, **kwargs):
        return tlc_check_cls_dataset(*args, **kwargs)

    def build_dataset(self, table):
        return TLCClassificationDataset(
            table=table,
            args=self.args,
            augment=False,
            prefix=self.args.split,
            image_column_name=self._image_column_name,
            label_column_name=self._label_column_name,
            exclude_zero=self._settings.exclude_zero_weight_collection,
            class_map=self.data["3lc_class_to_range"],
        )

    def _get_metrics_schemas(self):
        predicted_value_map = self.data["names_3lc"]
        predicted_value = tlc.Int32Value(value_map=predicted_value_map, number_role=tlc.NUMBER_ROLE_LABEL)
        predicted_schema = tlc.Schema(
            "Predicted",
            "The highest confidence class predicted by the model.",
            writable=False,
            value=predicted_value,
        )

        column_schemas = {
            "loss": tlc.Schema("Loss", "Cross Entropy Loss", writable=False, value=tlc.Float32Value()),
            "predicted": predicted_schema,
            "confidence": tlc.Schema(
                "Confidence",
                "The confidence of the prediction",
                value=tlc.Float32Value(value_min=0.0, value_max=1.0),
            ),
            "top1_accuracy": tlc.Schema(
                "Top-1 Accuracy",
                "The correctness of the prediction",
                value=tlc.Float32Value(),
            ),
        }

        if len(predicted_value_map) > 5:
            column_schemas["top5_accuracy"] = tlc.Schema(
                "Top-5 Accuracy",
                "The correctness of any of the top five confidence predictions",
                value=tlc.Float32Value(),
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

        if len(self.dataloader.dataset.table.get_value_map(self._label_column_name)) > 5:
            _, top5_pred = torch.topk(preds, 5, dim=1)
            labels_expanded = batch["cls"].view(-1, 1).expand_as(top5_pred)
            top5_correct = torch.any(top5_pred == labels_expanded, dim=1)
            batch_metrics["top5_accuracy"] = top5_correct.to(torch.float32)

        return batch_metrics

    def _add_embeddings_hook(self, model):
        """Add a hook to extract embeddings from the model, and infer the activation size. For a classification model,
        this amounts to finding the linear layer and extracting the input size."""

        # Find index of the linear layer
        linear_layer_index: int | None = None
        for index, module in enumerate(model.modules()):
            if isinstance(module, torch.nn.Linear):
                activation_size = module.in_features
                linear_layer_index = index
                break

        if linear_layer_index is None:
            raise ValueError("No linear layer found in model, cannot collect embeddings.")

        weak_self = weakref.ref(self)  # Avoid circular reference (self <-> hook_fn)

        def hook_fn(_module, _input, output):
            # Store embeddings
            self_ref = weak_self()
            embeddings = output.detach().cpu().numpy()
            self_ref.embeddings = embeddings

        # Add forward hook to collect embeddings
        for i, module in enumerate(model.modules()):
            if i == linear_layer_index - 1:
                self._hook_handles.append(module.register_forward_hook(hook_fn))

        return activation_size

    def _infer_batch_size(self, preds, batch) -> int:
        return preds.size(0)
