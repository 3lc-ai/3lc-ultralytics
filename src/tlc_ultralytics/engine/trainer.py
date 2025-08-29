from __future__ import annotations

from pathlib import Path

import tlc
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.metrics import smooth

from tlc_ultralytics.constants import DEFAULT_TRAIN_RUN_DESCRIPTION, TLC_COLORSTR
from tlc_ultralytics.engine.utils import _complete_label_column_name, _restore_random_state
from tlc_ultralytics.settings import Settings
from tlc_ultralytics.utils import reduce_embeddings


class TLCTrainerMixin(BaseTrainer):
    """A class extending the BaseTrainer class for training Ultralytics YOLO models with 3LC,
    which implements common 3LC-specific behavior across tasks. Use as a Mixin class for task-specific
    trainers.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        LOGGER.info("Using 3LC Trainer 🌟")
        self._settings = overrides.pop("settings", Settings())
        self._settings.verify(training=True)

        assert "data" in overrides or "tables" in overrides, (
            "You must provide either a data path or tables to train with 3LC."
        )

        if "data" in overrides and not isinstance(overrides["data"], (str, Path)):
            msg = (
                "`data` must be a string or pathlib.Path. "
                "If you are using `tlc.Table` objects, provide `tables` instead."
            )
            raise ValueError(msg)

        self._tables = overrides.pop("tables", None)

        # Column names
        self._image_column_name = overrides.pop("image_column_name", self._default_image_column_name)
        self._label_column_name = overrides.pop("label_column_name", self._default_label_column_name)
        self._label_column_name = _complete_label_column_name(self._label_column_name, self._default_label_column_name)

        super().__init__(cfg, overrides, _callbacks)

        self._train_validator = None
        self._train_equals_val_result = None

        if RANK in {-1, 0}:
            self._metrics_collection_epochs = set(self._settings.get_metrics_collection_epochs(self.epochs))

            # Create a 3LC run
            description = (
                self._settings.run_description if self._settings.run_description else DEFAULT_TRAIN_RUN_DESCRIPTION
            )

            project_name = (
                self._settings.project_name if self._settings.project_name else self.data["train"].project_name
            )
            self._run = tlc.init(
                project_name=project_name,
                description=description,
                run_name=self._settings.run_name,
            )

            LOGGER.info(
                f"{TLC_COLORSTR}Created run named '{self._run.url.parts[-1]}' in project {self._run.project_name}."
            )

            # Log parameters to 3LC
            self._log_3lc_parameters()
            self._run.set_status_running()
            self._print_metrics_collection_epochs()

    def _log_3lc_parameters(self):
        """Log various data as parameters to the tlc.Run."""
        if "val" in self.data:
            val_url = str(self.data["val"].url)
        else:
            val_url = str(self.data["test"].url)

        parameters = {
            **vars(self.args),  # YOLO arguments
            "3LC/train_url": str(self.data.get("train").url),  # 3LC table used for training
            "3LC/val_url": val_url,  # 3LC table used for validation
            **{f"3LC/{k}": v for k, v in vars(self._settings).items()},  # 3LC settings
        }
        self._run.set_parameters(parameters)

    def _print_metrics_collection_epochs(self):
        """Print collection epochs to the console."""

        # Special message when no collection is enabled
        if self._settings.collection_disable:
            message = "No metrics collection is enabled."
        # No collection during training
        elif not self._metrics_collection_epochs:
            message = "Metrics will be collected after training only."
        # Print collection epochs
        else:
            if len(self._metrics_collection_epochs) == 1:
                epoch = str(next(iter(self._metrics_collection_epochs)))
                message = f"Metrics will be collected after training and after epoch {epoch}."
            else:
                epochs = ", ".join(str(epoch) for epoch in sorted(self._metrics_collection_epochs))
                message = f"Metrics will be collected after training and after the following epochs: {epochs}"

        LOGGER.info(f"{TLC_COLORSTR}{message}")

    def get_dataset(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def build_dataset(self, table, mode="train", batch=None):
        raise NotImplementedError("Subclasses must implement this method.")

    def get_validator(self, dataloader):
        raise NotImplementedError("Subclasses must implement this method.")

    @property
    def train_validator(self):
        if RANK in {-1, 0}:
            if not self._train_validator:
                train_validator_dataloader = self.get_dataloader(
                    self.data["train"],
                    batch_size=self.batch_size if self.args.task == "obb" else self.batch_size * 2,
                    rank=-1,
                    mode="val",
                )
                self._train_validator = self.get_validator(dataloader=train_validator_dataloader)
            return self._train_validator
        else:
            return None

    def validate(self):
        """Perform validation with 3LC metrics collection, also on the training data, if applicable."""

        # Validate on the training set, unless the training and validation sets are identical
        if (
            not self._settings.collection_disable
            and not self._settings.collection_val_only
            and self.epoch + 1 in self._metrics_collection_epochs
            and not self._train_equals_val()
        ):
            with _restore_random_state():
                self.train_validator(trainer=self)

        # Validate on the validation/test set like usual
        return super().validate()

    def _train_equals_val(self):
        if self._train_equals_val_result is not None:
            return self._train_equals_val_result

        self._train_equals_val_result = self.data["train"] is (self.data.get("val") or self.data["test"])
        if self._train_equals_val_result:
            LOGGER.info(
                f"{TLC_COLORSTR}Training and validation sets are identical. "
                "Skipping duplicate validation on the training set."
            )
        return self._train_equals_val_result

    def final_eval(self):
        """Perform normal final validation with metrics collection on the val set, after first doing metrics collection
        on the train set.
        """
        if not self._settings.collection_val_only and not self._settings.collection_disable:
            if self.best.exists() and not self._train_equals_val():
                with _restore_random_state():
                    self.train_validator._final_validation = True
                    self.train_validator._epoch = self.epoch
                    self.train_validator.data = self.data
                    self.train_validator(model=self.best)

        self.validator._final_validation = True
        super().final_eval()
        self._save_confidence_metrics()

        if RANK in {-1, 0}:
            if self._settings.image_embeddings_dim > 0:
                train_url = self.data["train"].url
                val_url = self.data["val"].url if "val" in self.data else self.data["test"].url
                foreign_table_url = train_url if not self._settings.collection_val_only else val_url
                reduce_embeddings(
                    self._run,
                    method=self._settings.image_embeddings_reducer,
                    n_components=self._settings.image_embeddings_dim,
                    foreign_table_url=foreign_table_url,
                    reducer_args=self._settings.image_embeddings_reducer_args,
                )
            self._run.set_status_completed()

    def _save_confidence_metrics(self):
        if self.args.task not in ("detect", "segment"):
            return

        try:
            curves = [
                self.validator.metrics.box.f1_curve,  # (nc, 1000)
                self.validator.metrics.box.r_curve,  # (nc, 1000)
                self.validator.metrics.box.p_curve,  # (nc, 1000)
            ]
            names = ["F1_score", "Recall", "Precision"]
            px = self.validator.metrics.box.px  # (1000,) (linspace(0, 1)

            values = {}
            for py, name in zip(curves, names):
                y = smooth(py.mean(0), 0.05)
                best_val = y.max()
                best_conf = px[y.argmax()]
                values[f"3LC/{name}"] = {"best_val": best_val, "best_conf": best_conf}

            self._run.set_parameters(values)
        except Exception as e:
            LOGGER.error(TLC_COLORSTR + f"Failed to save confidence metrics: {e}")

    def save_metrics(self, metrics):
        # Log aggregate metrics after every epoch
        processed_metrics = self._process_metrics(metrics)

        self._run.add_output_value({"epoch": self.epoch + 1, **processed_metrics})

        super().save_metrics(metrics=metrics)

    def _process_metrics(self, metrics):
        return metrics
