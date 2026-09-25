from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
from ultralytics.models.yolo.semantic.train import SemanticSegmentationTrainer
from ultralytics.nn.tasks import SemanticSegmentationModel
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import SemanticSegmentationLoss
from ultralytics.utils.plotting import colors, plt_settings

from tlc_ultralytics.constants import TLC_COLORSTR
from tlc_ultralytics.detect.trainer import TLCDetectionTrainer
from tlc_ultralytics.engine.trainer import TLCTrainerMixin
from tlc_ultralytics.semantic.utils import apply_dataset_class_weights, uses_dataset_class_weights
from tlc_ultralytics.semantic.validator import TLCSemanticSegmentationValidator


# Monkeypatch Ultralytics' SemanticSegmentationModel to weight its loss from the dataset its 3LC table was created from,
# which `TLCSemanticSegmentationTrainer.set_model_attributes` attaches to the model. Ultralytics decides from
# `model.args.data` instead, which names no dataset when training from tables. Models without the attribute, like
# those plain Ultralytics trains, get Ultralytics' loss unchanged.
def _tlc_semantic_init_criterion(self):
    loss = SemanticSegmentationLoss(self)
    apply_dataset_class_weights(loss, getattr(self, "tlc_ultralytics_dataset", None))
    return loss


# Apply the monkeypatch once at import time
SemanticSegmentationModel.init_criterion = _tlc_semantic_init_criterion


class TLCSemanticSegmentationTrainer(SemanticSegmentationTrainer, TLCDetectionTrainer):
    """Trainer class for YOLO semantic segmentation with 3LC"""

    _validator_class = TLCSemanticSegmentationValidator
    _loss_names = ("ce_loss", "dice_loss", "aux_loss")
    _metric_replacements: ClassVar[list[tuple[str, str]]] = [("metrics", "val"), ("/", "_")]

    # Explicit bindings to ensure the TLCTrainerMixin methods win over SemanticSegmentationTrainer's in the MRO. Its
    # `get_dataset` adds a background class for polygon datasets, which a 3LC table declares itself.
    get_validator = TLCTrainerMixin.get_validator
    get_dataset = TLCTrainerMixin.get_dataset

    def set_model_attributes(self):
        """Set model attributes, and attach the Ultralytics dataset the tables were created from to the model.

        The model's loss weights its classes from it (see `apply_dataset_class_weights`). The EMA model is deep-copied
        from the model after this, so it carries the dataset too.
        """
        super().set_model_attributes()
        self.model.tlc_ultralytics_dataset = self.data.get("ultralytics_dataset")

    def _print_task_specific_parameters(self):
        """Print task-specific parameters to the console."""
        ultralytics_dataset = self.data.get("ultralytics_dataset")
        if uses_dataset_class_weights(ultralytics_dataset, self.data["nc"]):
            LOGGER.info(
                f"{TLC_COLORSTR}Weighting the cross-entropy loss with Ultralytics' Cityscapes class weights, since the "
                f"tables were created from '{ultralytics_dataset}'"
            )

    @plt_settings()
    def plot_training_labels(self):
        """Plot the training labels' class distribution for semantic segmentation.

        Same plot as `SemanticSegmentationTrainer.plot_training_labels`, which reads the dataset's mask files and skips
        the plot when there are none, as for a 3LC table. The masks are decoded from the table with
        `TLCSemanticDataset.load_mask` instead, already in training class indices, so no label mapping is applied.
        """
        LOGGER.info(f"Plotting labels to {self.save_dir / 'labels.jpg'}...")
        nc = self.data["nc"]
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
