from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
from ultralytics.models.yolo.semantic.train import SemanticSegmentationTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import SemanticSegmentationLoss
from ultralytics.utils.plotting import colors, plt_settings

from tlc_ultralytics.constants import TLC_COLORSTR
from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.engine.trainer import TLCTrainerMixin
from tlc_ultralytics.semantic.utils import (
    apply_dataset_class_weights,
    check_cityscapes_class_weights,
    dataset_class_weights_message,
)
from tlc_ultralytics.semantic.validator import TLCSemanticSegmentationValidator


class TLCSemanticSegmentationTrainer(SemanticSegmentationTrainer, TLCDetectionTrainer):
    """Trainer class for YOLO semantic segmentation with 3LC"""

    _validator_class = TLCSemanticSegmentationValidator
    _loss_names = ("ce_loss", "dice_loss", "aux_loss")
    _metric_replacements: ClassVar[list[tuple[str, str]]] = [("metrics", "val"), ("/", "_")]

    # Explicit binding to ensure TLCTrainerMixin.get_validator wins over SemanticSegmentationTrainer's in the MRO
    get_validator = TLCTrainerMixin.get_validator

    def get_dataset(self):
        """Get the 3LC dataset, and verify that `Settings.cityscapes_class_weights` can be applied to it.

        `TLCTrainerMixin.get_dataset` is used rather than `SemanticSegmentationTrainer.get_dataset`, which adds a
        background class for polygon datasets, which a 3LC table declares itself.
        """
        data = TLCTrainerMixin.get_dataset(self)
        check_cityscapes_class_weights(self._settings.cityscapes_class_weights, data["nc"])
        return data

    def set_model_attributes(self):
        """Set model attributes, and give the model a loss weighted from the dataset its tables were created from.

        Ultralytics' `SemanticSegmentationModel.init_criterion` weights the loss from `model.args.data`, which names no
        dataset when training from tables. The criterion is built here on the trainer's own model instead, weighted with
        `apply_dataset_class_weights`: `BaseModel.loss` only calls `init_criterion` while the model's `criterion` is
        None, so this one is used. `BaseTrainer._setup_train` calls this after moving the model to its device, whose
        device and dtype the loss takes, and before compiling it, wrapping it in DDP and deep-copying it into the EMA
        model, which computes the validation loss during training, so all of them share the weighted loss. Checkpoints
        carry the criterion only as Ultralytics' own loss class, and the final ones not at all (`strip_optimizer`).
        """
        super().set_model_attributes()
        criterion = SemanticSegmentationLoss(self.model)
        apply_dataset_class_weights(
            criterion, self.data.get("ultralytics_dataset"), self._settings.cityscapes_class_weights
        )
        self.model.criterion = criterion

    def _print_task_specific_parameters(self):
        """Print task-specific parameters to the console."""
        message = dataset_class_weights_message(
            self.data.get("ultralytics_dataset"),
            self.data["nc"],
            self._settings.cityscapes_class_weights,
            "the cross-entropy loss",
        )
        if message:
            LOGGER.info(f"{TLC_COLORSTR}{message}")

    @plt_settings()
    def plot_training_labels(self):
        """Plot the training labels' class distribution for semantic segmentation.

        Same plot as `SemanticSegmentationTrainer.plot_training_labels`, which reads the dataset's mask files and skips
        the plot when there are none, as for a 3LC table. The masks are decoded from the table with
        `TLCSemanticDataset.load_mask` instead, already in training class indices, so no label mapping is applied.
        """
        LOGGER.info(f"Plotting labels to {self.save_dir / 'labels.jpg'}...")
        nc = int(self.data["nc"])  # the data dict is typed as a union of its values
        names = self.data["names"]
        pixel_counts = np.zeros(nc, dtype=np.int64)

        dataset = self.train_loader.dataset
        sample_size = min(1000, len(dataset))
        indices = np.linspace(0, len(dataset) - 1, sample_size).astype(int)

        for idx in indices:
            mask = dataset.load_mask(int(idx))
            valid = (mask >= 0) & (mask < nc) & (mask != 255)
            if valid.any():
                classes, counts = np.unique(mask[valid], return_counts=True)
                for c, count in zip(classes, counts, strict=True):
                    pixel_counts[int(c)] += int(count)

        _, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
        bars = ax.bar(range(nc), pixel_counts, color=[[c / 255.0 for c in colors(i, False)] for i in range(nc)])
        ax.set_xlabel("Class")
        ax.set_ylabel("Pixels")
        ax.set_title("Training Labels Class Distribution")
        if 0 < len(names) < 30:
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(list(names.values()), rotation=90, fontsize=10)
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height):,}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        for spine in ax.spines.values():
            spine.set_visible(False)

        fname = self.save_dir / "labels.jpg"
        plt.savefig(fname, dpi=200)
        plt.close()
        if self.on_plot:
            self.on_plot(fname)
