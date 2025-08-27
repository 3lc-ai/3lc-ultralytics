from __future__ import annotations

import numpy as np
import tlc
import ultralytics
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, colorstr

from tlc_ultralytics.constants import (
    DEFAULT_COLLECT_RUN_DESCRIPTION,
    MAP,
    MAP50_95,
    MAP50_95_SEG,
    MAP_SEG,
    NUM_IMAGES,
    NUM_INSTANCES,
    PER_CLASS_METRICS_STREAM_NAME,
    PRECISION,
    PRECISION_SEG,
    RECALL,
    RECALL_SEG,
    TLC_COLORSTR,
    TRAINING_PHASE,
)
from tlc_ultralytics.engine.utils import _complete_label_column_name
from tlc_ultralytics.settings import Settings
from tlc_ultralytics.utils import image_embeddings_schema, training_phase_schema


def execute_when_collecting(method):
    def wrapper(self, *args, **kwargs):
        if self._should_collect:
            return method(self, *args, **kwargs)

    return wrapper


class TLCValidatorMixin(BaseValidator):
    def __init__(
        self,
        *args,
        run: tlc.Run | None = None,
        image_column_name: str | None = None,
        label_column_name: str | None = None,
        settings: Settings | None = None,
        training: bool = False,
        **kwargs,
    ):
        self._run = run
        self._image_column_name = image_column_name or self._default_image_column_name
        self._label_column_name = _complete_label_column_name(label_column_name, self._default_label_column_name)
        # Settings can be passed as an argument directly to the validator, or as a keyword from the trainer
        self._settings = settings or kwargs.get("args", {}).pop("settings", None) or Settings()
        self._training = training

        # Table is passed when doing validation only, not when training
        self._table = kwargs.get("args", {}).pop("table", None) if not training else None

        # State
        self._epoch = None
        self._should_collect = None
        self._seen = None
        self._final_validation = False
        self._hook_handles = []

        super().__init__(*args, **kwargs)

        if not self._training:
            # Set up dataset checking bypass before calling check_dataset
            self.data = self.check_dataset(
                self.args.data,
                {self.args.split: self._table} if self._table is not None else None,
                self._image_column_name,
                self._label_column_name,
                project_name=self._settings.project_name,
                splits=(self.args.split,),
            )

        # Create a run if not provided
        if self._run is None:
            # Reuse active run only if it has the same project name (if a different run is active)
            if self._settings.project_name:
                project_name = self._settings.project_name
            else:
                first_split = next(iter(self.data.keys()))
                project_name = self.data[first_split].project_name
                LOGGER.info(f"{TLC_COLORSTR}Using project name '{project_name}' from the provided table to create run.")

            if tlc.active_run() and tlc.active_run().project_name == project_name:
                self._run = tlc.active_run()
                run_name = self._run.url.parts[-1]
                LOGGER.info(f"{TLC_COLORSTR}Using active run named '{run_name}' in project {self._run.project_name}.")
            else:
                self._run = tlc.init(
                    project_name=project_name,
                    description=self._settings.run_description
                    if self._settings.run_description
                    else DEFAULT_COLLECT_RUN_DESCRIPTION,
                    run_name=self._settings.run_name,
                )
                LOGGER.info(
                    f"{TLC_COLORSTR}Created run named '{self._run.url.parts[-1]}' in project {self._run.project_name}."
                )

        self.metrics.run_url = self._run.url

    def __call__(self, trainer=None, model=None):
        self._epoch = trainer.epoch if trainer is not None else self._epoch

        if trainer:
            self._should_collect = (
                not self._settings.collection_disable and self._epoch + 1 in trainer._metrics_collection_epochs
            )
        else:
            self._should_collect = not self._settings.collection_disable

        # Define bypass functions that use our data object
        def bypass_check_det_dataset(*args, **kwargs):
            return self.data

        def bypass_check_cls_dataset(*args, **kwargs):
            return self.data

        # Patch the functions in ultralytics.data.utils on the validator module
        ultralytics.engine.validator.check_det_dataset = bypass_check_det_dataset
        ultralytics.engine.validator.check_cls_dataset = bypass_check_cls_dataset

        try:
            # Call parent to perform the validation
            out = super().__call__(trainer, model)
        finally:
            # Restore original functions
            ultralytics.engine.validator.check_det_dataset = ultralytics.data.utils.check_det_dataset
            ultralytics.engine.validator.check_cls_dataset = ultralytics.data.utils.check_cls_dataset

        self._write_per_class_metrics_tables()
        self._post_validation()

        return out

    def get_desc(self):
        """Add the split name next to the validation description"""
        desc = super().get_desc()

        split = self.dataloader.dataset.display_name.split("-")[-1]  # get final part
        initial_spaces = len(desc) - len(desc.lstrip())
        split_centered = split.center(initial_spaces)
        split_str = f"{colorstr(split_centered)}"
        desc = split_str + desc[len(split_centered) :]

        return desc

    def init_metrics(self, model):
        super().init_metrics(model)

        self._verify_model_data_compatibility(model.names)
        self._pre_validation(model)

    def build_dataset(self, table):
        """Build a dataset from a table"""
        raise NotImplementedError("Subclasses must implement this method.")

    def _verify_model_data_compatibility(self, names):
        """Verify that the model being validated is compatible with the data"""
        raise NotImplementedError("Subclasses must implement this method.")

    def _get_metrics_schemas(self) -> dict[str, tlc.Schema]:
        """Get the metrics schemas for the 3LC metrics data"""
        raise NotImplementedError("Subclasses must implement this method.")

    def _compute_3lc_metrics(self, preds, batch) -> dict[str, tlc.MetricData]:
        """Compute 3LC metrics for a batch of predictions and targets"""
        raise NotImplementedError("Subclasses must implement this method.")

    def _add_embeddings_hook(self, model) -> int:
        """Add a hook to extract embeddings from the model, and infer the activation size"""
        raise NotImplementedError("Subclasses must implement this method.")

    def _infer_batch_size(self, preds) -> int:
        """Infer the batch size from the predictions"""
        raise NotImplementedError("Subclasses must implement this method.")

    def _prepare_loss_fn(self, model):
        pass

    def update_metrics(self, preds, batch):
        """Collect 3LC metrics"""
        self._update_metrics(preds, batch)

        # Let parent collect its own metrics
        super().update_metrics(preds, batch)

    @execute_when_collecting
    def _update_metrics(self, preds, batch):
        """Update 3LC metrics with common and task-specific metrics"""
        batch_size = self._infer_batch_size(preds, batch)

        batch_metrics = {
            tlc.EXAMPLE_ID: [int(example_id) for example_id in batch["example_id"]],
            **self._compute_3lc_metrics(preds, batch),  # Task specific metrics
        }

        if self._settings.metrics_collection_function:
            batch_metrics.update(self._settings.metrics_collection_function(preds, batch))

        if self._settings.image_embeddings_dim > 0:
            batch_metrics["embeddings"] = self.embeddings

        if self._training:
            batch_metrics[tlc.EPOCH] = [self._epoch + 1] * batch_size
            training_phase = 1 if self._final_validation else 0
            batch_metrics[TRAINING_PHASE] = [training_phase] * batch_size

        self._metrics_writer.add_batch(batch_metrics)
        self._seen += batch_size

    @execute_when_collecting
    def _pre_validation(self, model):
        """Prepare the validator for metrics collection"""
        column_schemas = {}
        column_schemas.update(self._get_metrics_schemas())  # Add task-specific metrics schema

        if self._settings.metrics_schemas:
            column_schemas.update(self._settings.metrics_schemas)

        self._prepare_loss_fn(model)

        if self._settings.image_embeddings_dim > 0:
            # Add hook and get the activation size
            activation_size = self._add_embeddings_hook(model)

            column_schemas["embeddings"] = image_embeddings_schema(activation_size=activation_size)

        if self._epoch is not None:
            column_schemas[TRAINING_PHASE] = training_phase_schema()

        self._run.set_status_collecting()

        self._metrics_writer = tlc.MetricsTableWriter(
            run_url=self._run.url,
            foreign_table_url=self.dataloader.dataset.table.url,
            column_schemas=column_schemas,
        )

        self._seen = 0

    @execute_when_collecting
    def _post_validation(self):
        """Clean up the validator after one validation pass"""
        # Write metrics data to 3LC run
        self._metrics_writer.finalize()
        metrics_infos = self._metrics_writer.get_written_metrics_infos()
        self._run.update_metrics(metrics_infos)

        self._run.add_input_table(self.dataloader.dataset.table.url)

        # Improve memory usage - don't cache metrics data
        for metrics_info in metrics_infos:
            tlc.ObjectRegistry._delete_object_from_caches(tlc.Url(metrics_info["url"]).to_absolute(self._run.url))

        self._run.set_status_running()

        # Remove hook handles
        if self._settings.image_embeddings_dim > 0:
            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles.clear()

        # Reset state
        self._seen = None
        self._training_phase = None
        self._final_validation = None

    def _write_per_class_metrics_tables(self) -> None:
        if self.args.task not in ("detect", "segment"):
            # Per-class metrics currently only supported for detection and segmentation tasks
            return

        metrics_writer = tlc.MetricsTableWriter(
            run_url=self._run.url,
            column_schemas=self._per_class_metrics_schemas(),
        )

        epoch = self._epoch + 1 if self._epoch is not None else -1
        training_phase = 1 if self._final_validation else 0
        num_classes = self.nc + 1  # all classes plus "all"

        metrics_batch = (
            {
                tlc.EPOCH: [epoch] * num_classes,
                TRAINING_PHASE: [training_phase] * num_classes,
            }
            if self._training
            else {}
        )

        metrics_batch.update(
            {
                tlc.FOREIGN_TABLE_ID: [0] * num_classes,
                tlc.LABEL: list(range(num_classes)),
                NUM_INSTANCES: np.append(self.metrics.nt_per_class, self.metrics.nt_per_class.sum()),
                NUM_IMAGES: np.append(self.metrics.nt_per_image, self.seen),
                **self._generate_per_class_metrics(),
            }
        )

        metrics_writer.add_batch(metrics_batch)
        metrics_writer.finalize()
        metrics_infos = metrics_writer.get_written_metrics_infos()
        for m in metrics_infos:
            # Set the stream name to the per-class metrics stream
            # TODO: This should be a constructor argument to MetricsTableWriter
            m["stream_name"] = PER_CLASS_METRICS_STREAM_NAME
        self._run.update_metrics(metrics_infos)

    def _per_class_metrics_schemas(self):
        metrics_schemas = {
            TRAINING_PHASE: training_phase_schema(),
            tlc.FOREIGN_TABLE_ID: tlc.ForeignTableIdSchema(
                self.dataloader.dataset.table.url.to_relative(self._run.url / "metrics").to_str(),
            ),
            tlc.LABEL: tlc.CategoricalLabel("class", {**self.names, self.nc: "all"}).schema,
            NUM_IMAGES: tlc.Schema(
                value=tlc.Int32Value(),
                description="Number of images with at least one instance of the class",
            ),
            NUM_INSTANCES: tlc.Schema(
                value=tlc.Int32Value(),
                description="Total number of instances of the class in all images",
            ),
            PRECISION: tlc.Schema(
                value=tlc.Float32Value(),
                description="Precision of the class",
            ),
            RECALL: tlc.Schema(
                value=tlc.Float32Value(),
                description="Recall of the class",
            ),
            MAP: tlc.Schema(
                value=tlc.Float32Value(),
                description="mAP of the class",
            ),
            MAP50_95: tlc.Schema(
                value=tlc.Float32Value(),
                description="mAP50-95 of the class",
            ),
        }

        if self.args.task == "segment":
            metrics_schemas[PRECISION_SEG] = tlc.Schema(
                value=tlc.Float32Value(),
                description="Mask precision of the class",
            )

            metrics_schemas[RECALL_SEG] = tlc.Schema(
                value=tlc.Float32Value(),
                description="Mask recall of the class",
            )

            metrics_schemas[MAP_SEG] = tlc.Schema(
                value=tlc.Float32Value(),
                description="Mask mAP of the class",
            )

            metrics_schemas[MAP50_95_SEG] = tlc.Schema(
                value=tlc.Float32Value(),
                description="Mask mAP50-95 of the class",
            )
        return metrics_schemas

    def _generate_per_class_metrics(self):
        """Transform metrics from self.metrics to a format suitable for 3LC"""
        # Consider moving this to TLCDetectionValidator when supporting other tasks
        precisions = np.zeros(self.nc + 1)
        recalls = np.zeros(self.nc + 1)
        mAPs = np.zeros(self.nc + 1)
        mAP50_95s = np.zeros(self.nc + 1)

        if self.args.task == "segment":
            precisions_seg = np.zeros(self.nc + 1)
            recalls_seg = np.zeros(self.nc + 1)
            mAPs_seg = np.zeros(self.nc + 1)
            mAP50_95s_seg = np.zeros(self.nc + 1)

        for i in range(self.nc):
            if i in self.metrics.ap_class_index:
                class_results = self.metrics.class_result(np.where(self.metrics.ap_class_index == i)[0][0])
                if self.args.task == "detect":
                    p, r, ap50, ap5095 = class_results
                else:
                    p, r, ap50, ap5095, p_seg, r_seg, ap50_seg, ap5095_seg = class_results
            else:
                p, r, ap50, ap5095 = 0.0, 0.0, 0.0, 0.0
                if self.args.task == "segment":
                    p_seg, r_seg, ap50_seg, ap5095_seg = 0.0, 0.0, 0.0, 0.0

            precisions[i] = p
            recalls[i] = r
            mAPs[i] = ap50
            mAP50_95s[i] = ap5095

            if self.args.task == "segment":
                precisions_seg[i] = p_seg
                recalls_seg[i] = r_seg
                mAPs_seg[i] = ap50_seg
                mAP50_95s_seg[i] = ap5095_seg

        mean_results = self.metrics.mean_results()
        if self.args.task == "detect":
            all_p, all_r, all_mAP50, all_mAP50_95 = mean_results
        else:
            (
                all_p,
                all_r,
                all_mAP50,
                all_mAP50_95,
                all_p_seg,
                all_r_seg,
                all_mAP50_seg,
                all_mAP50_95_seg,
            ) = mean_results

        precisions[self.nc] = all_p
        recalls[self.nc] = all_r
        mAPs[self.nc] = all_mAP50
        mAP50_95s[self.nc] = all_mAP50_95

        metrics = {
            PRECISION: precisions,
            RECALL: recalls,
            MAP: mAPs,
            MAP50_95: mAP50_95s,
        }

        if self.args.task == "segment":
            precisions_seg[self.nc] = all_p_seg
            recalls_seg[self.nc] = all_r_seg
            mAPs_seg[self.nc] = all_mAP50_seg
            mAP50_95s_seg[self.nc] = all_mAP50_95_seg

            metrics[PRECISION_SEG] = precisions_seg
            metrics[RECALL_SEG] = recalls_seg
            metrics[MAP_SEG] = mAPs_seg
            metrics[MAP50_95_SEG] = mAP50_95s_seg

        return metrics

    def _verify_model_data_compatibility(self, model_class_names):
        """Verify that the model classes match the dataset classes. For a classification model, this amounts to checking
        that the order of the class names match and that they have the same number of classes."""
        dataset_class_names = self.data["names"]

        if len(model_class_names) != len(dataset_class_names):
            raise ValueError(
                f"The model and data are incompatible. The model was trained on {len(model_class_names)} classes, "
                f"but the data has {len(dataset_class_names)} classes. "
            )

        # Imagenet has a class name transform in YOLO which is not applied on table creation.
        # TODO: Remove when image_folder takes a sparse class name mapping to change these
        if "n01440764" not in set(dataset_class_names.values()):
            if model_class_names != dataset_class_names:
                raise ValueError(
                    "The model was trained on a different set of classes to the classes in the dataset, "
                    "or the classes are in a different order."
                )
