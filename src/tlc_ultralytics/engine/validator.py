from __future__ import annotations

import numpy as np
import tlc
import torch.distributed as dist
import ultralytics
from tlc._core.object_registry import ObjectRegistry
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, RANK, colorstr

from tlc_ultralytics.constants import (
    DEFAULT_COLLECT_RUN_DESCRIPTION,
    EPOCH,
    EXAMPLE_ID,
    FOREIGN_TABLE_ID,
    LABEL,
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
from tlc_ultralytics.engine.utils import _complete_label_column_name, _handle_deprecated_column_name
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
        # Settings can be passed as an argument directly to the validator, or as a keyword from the trainer
        self._settings = settings or kwargs.get("args", {}).pop("settings", None) or Settings()

        self._settings.image_column_name = _handle_deprecated_column_name(
            image_column_name,
            self._settings.image_column_name,
            self._default_image_column_name,
            column_name="image_column_name",
        )
        self._settings.label_column_name = _handle_deprecated_column_name(
            label_column_name,
            self._settings.label_column_name,
            self._default_label_column_name,
            column_name="label_column_name",
        )
        self._settings.label_column_name = _complete_label_column_name(
            self._settings.label_column_name,
            self._default_label_column_name,
        )

        self._training = training

        if not training:
            self._settings.verify(training=False)

        # Table is passed when doing validation only, not when training
        self._table = kwargs.get("args", {}).pop("table", None) if not training else None

        # State
        self._epoch = None
        self._should_collect = None
        self._seen = None
        self._final_validation = False
        self._hook_handles = []

        # Instance embeddings state
        self._instance_feature_map = None
        self._buffered_metrics = []
        self._buffered_raw_instance_embeddings = []
        self._buffered_raw_gt_instance_embeddings = []

        super().__init__(*args, **kwargs)

        if not self._training:
            # Set up dataset checking bypass before calling check_dataset
            self.data = self.check_dataset(
                self.args.data,
                {self.args.split: self._table} if self._table is not None else None,
                self._settings.image_column_name,
                self._settings.label_column_name,
                project_name=self._settings.project_name,
                splits=(self.args.split,),
            )

        # Create a run if not provided
        if self._run is None:
            # Reuse active run only if it has the same project name (if a different run is active)
            if self._settings.project_name:
                project_name = self._settings.project_name
                first_split = next(iter(self.data.keys()))
            else:
                first_split = next(iter(self.data.keys()))
                project_name = self.data[first_split].project_name
                LOGGER.info(f"{TLC_COLORSTR}Using project name '{project_name}' from the provided table to create run.")

            if tlc.active_run() and tlc.active_run().project_name == project_name:
                self._run = tlc.active_run()
                run_name = self._run.url.parts[-1]
                LOGGER.info(f"{TLC_COLORSTR}Using active run named '{run_name}' in project {self._run.project_name}.")
            else:
                try:
                    root_url = self.data[first_split].root
                except Exception:
                    root_url = None

                self._run = tlc.init(
                    project_name=project_name,
                    description=self._settings.run_description or DEFAULT_COLLECT_RUN_DESCRIPTION,
                    run_name=self._settings.run_name,
                    root_url=root_url,
                )
                LOGGER.info(
                    f"{TLC_COLORSTR}Created run named '{self._run.url.parts[-1]}' in project {self._run.project_name}."
                )

        if self.args.task == "pose" and not self._training:
            table_sigmas = self.data.get("oks_sigmas")
            if table_sigmas is not None and isinstance(table_sigmas, list):
                table_sigmas_rounded = [round(x, 2) for x in table_sigmas]
                LOGGER.info(f"{TLC_COLORSTR}Using OKS sigmas: {table_sigmas_rounded} from Table for validation")

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

        # Per-class metrics only on RANK 0 (uses aggregate metrics from gather_stats)
        if RANK in {-1, 0}:
            self._write_per_class_metrics_tables()

        # All ranks call _post_validation to participate in distributed gathering
        # (the method handles RANK checks internally for the actual run updates)
        self._post_validation()

        return out

    def get_desc(self):
        """Add the split name next to the validation description"""
        desc = super().get_desc()

        split = self.dataloader.dataset.display_name.split("-")[-1]  # get final part
        initial_spaces = len(desc) - len(desc.lstrip())
        split_centered = split.center(initial_spaces)
        split_str = f"{colorstr(split_centered)}"
        desc = split_str + desc[len(split_centered):]

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

    def _process_predictions(self, preds, batch):
        """Filter, scale, and build 3LC annotations for a batch of predictions."""
        results = []
        for i, pred in enumerate(preds):
            pbatch = self._prepare_batch(i, batch)
            h, w = pbatch["ori_shape"]

            mask = pred["conf"] >= self._settings.conf_thres
            if not mask.any():
                results.append(self._empty_annotation(h, w))
                continue

            filtered = {k: v[mask] for k, v in pred.items()}

            # Keep only top max_det predictions by confidence
            max_det = self._settings.max_det
            if len(filtered["conf"]) > max_det:
                topk = filtered["conf"].topk(max_det).indices
                filtered = {k: v[topk] for k, v in filtered.items()}

            scaled = self.scale_preds(filtered, pbatch)
            mapped_classes = [self.data["range_to_3lc_class"][int(c)] for c in scaled["cls"].tolist()]
            results.append(self._build_annotation(scaled, mapped_classes, h, w))

        return results

    def _add_embeddings_hook(self, model) -> int:
        """Add a hook to extract embeddings from the model, and infer the activation size"""
        raise NotImplementedError("Subclasses must implement this method.")

    def _add_instance_embeddings_hook(self, model) -> int:
        """Add a hook to capture high-resolution feature maps for instance embeddings.

        Subclasses can override for custom behavior. Default delegates to detect validator.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _extract_instance_embeddings(self, preds, batch) -> list[np.ndarray]:
        """Extract per-instance raw embeddings from the captured feature map.

        Subclasses must implement this for their specific prediction format.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _inject_instance_embeddings(self, batch_metrics, reduced_embeddings):
        """Inject reduced instance embeddings into the prediction struct.

        Subclasses must implement this for their specific prediction format.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _extract_gt_instance_embeddings(self, preds, batch) -> list[np.ndarray]:
        """Extract per-instance raw embeddings for ground-truth annotations.

        Subclasses must implement this for their specific GT annotation format.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _inject_gt_instance_embeddings(self, batch_metrics, reduced_embeddings):
        """Inject reduced ground-truth instance embeddings as a top-level metric column.

        Subclasses must implement this for their specific GT annotation format.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _infer_batch_size(self, preds, batch=None) -> int:
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
        """Update 3LC metrics with common and task-specific metrics.

        In DDP mode, each rank collects metrics for its portion of the data.
        These are gathered to RANK 0 in _post_validation.

        When instance_embeddings_dim > 0, metrics are buffered instead of
        being written to the MetricsTableWriter immediately.
        """
        batch_size = self._infer_batch_size(preds, batch)

        batch_metrics = {
            EXAMPLE_ID: [int(example_id) for example_id in batch["example_id"]],
            **self._compute_3lc_metrics(preds, batch),  # Task specific metrics
        }

        if self._settings.metrics_collection_function:
            batch_metrics.update(self._settings.metrics_collection_function(preds, batch))

        if self._settings.image_embeddings_dim > 0:
            batch_metrics["embeddings"] = self.embeddings

        if self._training:
            batch_metrics[EPOCH] = [self._epoch + 1] * batch_size
            training_phase = 1 if self._final_validation else 0
            batch_metrics[TRAINING_PHASE] = [training_phase] * batch_size

        # Add DDP rank for distributed validation debugging
        if RANK >= 0:
            batch_metrics["ddp_rank"] = [RANK] * batch_size

        # Extract and buffer instance embeddings if enabled
        if self._settings.instance_embeddings_dim > 0:
            raw_instance_embs = self._extract_instance_embeddings(preds, batch)
            self._buffered_raw_instance_embeddings.extend(raw_instance_embs)

            if self._settings.ground_truth_instance_embeddings:
                raw_gt_embs = self._extract_gt_instance_embeddings(preds, batch)
                self._buffered_raw_gt_instance_embeddings.extend(raw_gt_embs)

            self._buffered_metrics.append(batch_metrics)
        else:
            self._metrics_writer.add_batch(batch_metrics)

        self._seen += batch_size

    @execute_when_collecting
    def _pre_validation(self, model):
        """Prepare the validator for metrics collection.

        In DDP mode, each rank prepares its own metrics writer.
        """
        column_schemas = {}
        column_schemas.update(self._get_metrics_schemas())  # Add task-specific metrics schema

        if self._settings.metrics_schemas:
            column_schemas.update(self._settings.metrics_schemas)

        self._prepare_loss_fn(model)

        if self._settings.image_embeddings_dim > 0:
            # Add hook and get the activation size
            activation_size = self._add_embeddings_hook(model)

            column_schemas["embeddings"] = image_embeddings_schema(activation_size=activation_size)

        if self._settings.instance_embeddings_dim > 0:
            self._instance_embeddings_channel_size = self._add_instance_embeddings_hook(model)
            # Schema for instance embeddings is added by subclasses in _get_metrics_schemas
            # Clear buffers for this validation pass
            self._buffered_metrics = []
            self._buffered_raw_instance_embeddings = []
            self._buffered_raw_gt_instance_embeddings = []

        if self._settings.ground_truth_instance_embeddings:
            from tlc_ultralytics.utils.schemas import instance_embeddings_list_schema

            dim = self._settings.instance_embeddings_dim
            column_schemas["ground_truth_instance_embedding"] = instance_embeddings_list_schema(
                dim, display_name=f"Ground Truth Instance Embedding ({dim}D)"
            )

        if self._epoch is not None:
            column_schemas[TRAINING_PHASE] = training_phase_schema()

        # Add DDP rank column for distributed validation debugging
        if RANK >= 0:
            column_schemas["ddp_rank"] = tlc.schemas.Int32Schema(
                display_name="DDP rank",
                description="DDP rank that processed this sample",
                default_visible=False,
            )

        # Only RANK 0 (or single GPU) updates run status
        if RANK in {-1, 0}:
            self._run.set_status_collecting()

        self._metrics_writer = tlc.MetricsTableWriter(
            run_url=self._run.url,
            foreign_table_url=self.dataloader.dataset.table.url,
            schema=column_schemas,
        )

        self._seen = 0

    @execute_when_collecting
    def _post_validation(self):
        """Clean up the validator after one validation pass.

        When instance_embeddings_dim > 0, reduces all buffered instance embeddings
        and writes them along with buffered metrics to the MetricsTableWriter.

        In DDP mode, gathers metrics_infos and input table URLs from all ranks
        to RANK 0, which then updates the run with all collected data.
        """
        # If instance embeddings are enabled, reduce and write buffered metrics
        if self._settings.instance_embeddings_dim > 0 and self._buffered_metrics:
            self._reduce_and_write_buffered_metrics()

        # Each rank finalizes its own metrics writer
        self._metrics_writer.finalize()
        metrics_infos = self._metrics_writer.get_written_metrics_infos()
        input_table_url = self.dataloader.dataset.table.url.to_str()

        # Gather metrics from all ranks to RANK 0 in DDP mode
        input_table_urls: list[str] = []
        if RANK >= 0:
            world_size = dist.get_world_size()  # type: ignore[possibly-missing-attribute]
            gathered_metrics_infos = [None] * world_size if RANK == 0 else None
            gathered_input_urls = [None] * world_size if RANK == 0 else None

            dist.gather_object(metrics_infos, gathered_metrics_infos, dst=0)  # type: ignore[possibly-missing-attribute]
            dist.gather_object(input_table_url, gathered_input_urls, dst=0)  # type: ignore[possibly-missing-attribute]

            if RANK == 0:
                assert gathered_metrics_infos is not None
                assert gathered_input_urls is not None

                # Flatten metrics_infos from all ranks
                all_metrics_infos = []
                for rank_metrics in gathered_metrics_infos:
                    all_metrics_infos.extend(rank_metrics)
                metrics_infos = all_metrics_infos

                # Collect unique input table URLs (should all be the same in distributed validation)
                input_table_urls = list(set(gathered_input_urls))
        else:
            # Single GPU mode (RANK == -1)
            input_table_urls = [input_table_url]

        # Only RANK 0 (or single GPU) updates the run
        if RANK in {-1, 0}:
            self._run.update_metrics(metrics_infos)

            for url in input_table_urls:
                self._run.add_input_table(tlc.Url(url))

            # Improve memory usage - don't cache metrics data
            for metrics_info in metrics_infos:
                ObjectRegistry._delete_object_from_caches(tlc.Url(metrics_info["url"]).to_absolute(self._run.url))

            self._run.set_status_running()

        # Remove hook handles (all ranks)
        if self._settings.image_embeddings_dim > 0 or self._settings.instance_embeddings_dim > 0:
            for handle in self._hook_handles:
                handle.remove()
            self._hook_handles.clear()

        # Reset state (all ranks)
        self._seen = None
        self._training_phase = None
        self._final_validation = None
        self._instance_feature_map = None
        self._buffered_metrics = []
        self._buffered_raw_instance_embeddings = []
        self._buffered_raw_gt_instance_embeddings = []

    def _reduce_and_write_buffered_metrics(self):
        """Reduce all buffered instance embeddings and write metrics to the writer."""
        from tlc_ultralytics.utils.embeddings import reduce_instance_embeddings, transform_instance_embeddings

        # Reduce all predicted instance embeddings at once (fit_transform)
        reduced_per_image, reducer = reduce_instance_embeddings(
            self._buffered_raw_instance_embeddings,
            method=self._settings.image_embeddings_reducer,
            n_components=self._settings.instance_embeddings_dim,
        )

        # Transform GT embeddings with the same fitted reducer
        gt_reduced_per_image = None
        if self._settings.ground_truth_instance_embeddings and self._buffered_raw_gt_instance_embeddings:
            if reducer is not None:
                gt_reduced_per_image = transform_instance_embeddings(
                    self._buffered_raw_gt_instance_embeddings,
                    reducer,
                    n_components=self._settings.instance_embeddings_dim,
                )
            else:
                # No reducer (all predicted were empty) — produce empty GT embeddings too
                gt_reduced_per_image = [
                    np.empty((0, self._settings.instance_embeddings_dim), dtype=np.float32)
                    for _ in self._buffered_raw_gt_instance_embeddings
                ]

        # Inject reduced embeddings into buffered predictions and write
        emb_idx = 0
        gt_emb_idx = 0
        for batch_metrics in self._buffered_metrics:
            batch_size = len(batch_metrics[tlc.EXAMPLE_ID])
            batch_reduced = reduced_per_image[emb_idx : emb_idx + batch_size]
            emb_idx += batch_size

            self._inject_instance_embeddings(batch_metrics, batch_reduced)

            if gt_reduced_per_image is not None:
                gt_batch_reduced = gt_reduced_per_image[gt_emb_idx : gt_emb_idx + batch_size]
                gt_emb_idx += batch_size
                self._inject_gt_instance_embeddings(batch_metrics, gt_batch_reduced)

            self._metrics_writer.add_batch(batch_metrics)

    def _write_per_class_metrics_tables(self) -> None:
        if self.args.task not in ("detect", "segment", "obb"):
            # Per-class metrics currently only supported for detection, segmentation, and obb tasks
            return

        metrics_writer = tlc.MetricsTableWriter(
            run_url=self._run.url,
            schema=self._per_class_metrics_schemas(),
            stream_name=PER_CLASS_METRICS_STREAM_NAME,
        )

        epoch = self._epoch + 1 if self._epoch is not None else -1
        training_phase = 1 if self._final_validation else 0
        num_classes = self.nc + 1  # all classes plus "all"

        metrics_batch = (
            {
                EPOCH: [epoch] * num_classes,
                TRAINING_PHASE: [training_phase] * num_classes,
            }
            if self._training
            else {}
        )

        metrics_batch.update(
            {
                FOREIGN_TABLE_ID: [0] * num_classes,
                LABEL: list(range(num_classes)),
                NUM_INSTANCES: np.append(self.metrics.nt_per_class, self.metrics.nt_per_class.sum()),
                NUM_IMAGES: np.append(self.metrics.nt_per_image, np.int64(self.seen)),
                **self._generate_per_class_metrics(),
            }
        )

        metrics_writer.add_batch(metrics_batch)
        metrics_writer.finalize()

    def _per_class_metrics_schemas(self):
        metrics_schemas = {
            TRAINING_PHASE: training_phase_schema(),
            FOREIGN_TABLE_ID: tlc.schemas.ForeignTableIdSchema(
                self.dataloader.dataset.table.url.to_relative(self._run.url / "metrics").to_str(),
            ),
            LABEL: tlc.schemas.CategoricalLabelSchema(classes={**self.names, self.nc: "all"}),
            NUM_IMAGES: tlc.schemas.Int32Schema(
                description="Number of images with at least one instance of the class",
            ),
            NUM_INSTANCES: tlc.schemas.Int32Schema(
                description="Total number of instances of the class in all images",
            ),
            PRECISION: tlc.schemas.Float32Schema(description="Precision of the class"),
            RECALL: tlc.schemas.Float32Schema(description="Recall of the class"),
            MAP: tlc.schemas.Float32Schema(description="mAP of the class"),
            MAP50_95: tlc.schemas.Float32Schema(description="mAP50-95 of the class"),
        }

        if self.args.task == "segment":
            metrics_schemas[PRECISION_SEG] = tlc.schemas.Float32Schema(description="Mask precision of the class")

            metrics_schemas[RECALL_SEG] = tlc.schemas.Float32Schema(description="Mask recall of the class")

            metrics_schemas[MAP_SEG] = tlc.schemas.Float32Schema(description="Mask mAP of the class")

            metrics_schemas[MAP50_95_SEG] = tlc.schemas.Float32Schema(description="Mask mAP50-95 of the class")
        return metrics_schemas

    def _generate_per_class_metrics(self):
        """Transform metrics from self.metrics to a format suitable for 3LC"""
        # Consider moving this to TLCDetectionValidator when supporting other tasks
        is_segment = self.args.task == "segment"

        precisions = np.zeros(self.nc + 1)
        recalls = np.zeros(self.nc + 1)
        mAPs = np.zeros(self.nc + 1)
        mAP50_95s = np.zeros(self.nc + 1)

        precisions_seg = np.zeros(self.nc + 1) if is_segment else None
        recalls_seg = np.zeros(self.nc + 1) if is_segment else None
        mAPs_seg = np.zeros(self.nc + 1) if is_segment else None
        mAP50_95s_seg = np.zeros(self.nc + 1) if is_segment else None

        for i in range(self.nc):
            p_seg = r_seg = ap50_seg = ap5095_seg = 0.0
            if i in self.metrics.ap_class_index:
                class_results = self.metrics.class_result(np.where(self.metrics.ap_class_index == i)[0][0])
                if self.args.task in ("detect", "obb"):
                    p, r, ap50, ap5095 = class_results
                else:  # seg and pose
                    p, r, ap50, ap5095, p_seg, r_seg, ap50_seg, ap5095_seg = class_results
            else:
                p, r, ap50, ap5095 = 0.0, 0.0, 0.0, 0.0

            precisions[i] = p
            recalls[i] = r
            mAPs[i] = ap50
            mAP50_95s[i] = ap5095

            if self.args.task == "segment":
                precisions_seg[i] = p_seg
                recalls_seg[i] = r_seg
                mAPs_seg[i] = ap50_seg
                mAP50_95s_seg[i] = ap5095_seg

        all_p_seg = all_r_seg = all_mAP50_seg = all_mAP50_95_seg = 0.0

        mean_results = self.metrics.mean_results()
        if self.args.task in ("detect", "obb"):
            all_p, all_r, all_mAP50, all_mAP50_95 = mean_results
        else:  # seg and pose
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

        if is_segment:
            assert precisions_seg is not None
            assert recalls_seg is not None
            assert mAPs_seg is not None
            assert mAP50_95s_seg is not None

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
