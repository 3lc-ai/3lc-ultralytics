from __future__ import annotations

import numpy as np
import tlc
import torch.distributed as dist
import ultralytics
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, RANK, colorstr

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
from tlc_ultralytics.engine.utils import _complete_label_column_name, _handle_deprecated_column_name
from tlc_ultralytics.settings import Settings
from tlc_ultralytics.utils import image_embeddings_schema, training_phase_schema
from tlc_ultralytics.utils.schemas import _instance_embeddings_list_schema, _raw_instance_embeddings_schema


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

        # TEMP(instance-embeddings): the cls-head hook stashes its captured feature
        # map here every forward pass; raw embeddings are extracted from it during
        # _update_metrics and discarded once written. The two `_raw_*_emb` lists
        # hold the per-image raw vectors (small — ~hundreds of MB at COCO-train
        # scale) that the reducer fits on at the end of validation.
        self._instance_feature_map = None
        self._raw_pred_emb: list[np.ndarray] = []
        self._raw_gt_emb: list[np.ndarray] = []

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

    def _extract_gt_instance_embeddings(self, preds, batch) -> list[np.ndarray]:
        """Extract per-instance raw embeddings for ground-truth annotations.

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

        When instance_embeddings_dim > 0, raw per-instance embeddings are
        written inline as ``predicted_instance_embedding_raw`` / ``..._raw_gt``
        and held in a small in-RAM list for fitting the reducer at end-of-pass.
        Heavy fields like segmentation masks RLE-encode at handoff via
        ``MetricsTableWriter.add_batch``, so peak memory stays bounded.
        """
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

        # Add DDP rank for distributed validation debugging
        if RANK >= 0:
            batch_metrics["ddp_rank"] = [RANK] * batch_size

        # TEMP(instance-embeddings): write raw vectors inline; the end-of-pass
        # rewrite swaps them for the reduced column. This goes away when core
        # 3LC reduces variable-length embedding list columns server-side.
        if self._settings.instance_embeddings_dim > 0:
            raw_instance_embs = self._extract_instance_embeddings(preds, batch)
            batch_metrics["predicted_instance_embedding_raw"] = [
                (a.tolist() if a.size else []) for a in raw_instance_embs
            ]
            self._raw_pred_emb.extend(raw_instance_embs)

            if self._settings.ground_truth_instance_embeddings:
                raw_gt_embs = self._extract_gt_instance_embeddings(preds, batch)
                batch_metrics["ground_truth_instance_embedding_raw"] = [
                    (a.tolist() if a.size else []) for a in raw_gt_embs
                ]
                self._raw_gt_emb.extend(raw_gt_embs)

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
            # TEMP(instance-embeddings): write raw embeddings as a column during
            # the streaming pass; _post_validation rewrites the table with a
            # reduced column in their place. Goes away once core 3LC reduces
            # variable-length embedding list columns server-side.
            c_raw = self._add_instance_embeddings_hook(model)
            self._instance_embeddings_channel_size = c_raw
            column_schemas["predicted_instance_embedding_raw"] = _raw_instance_embeddings_schema(
                c_raw, display_name="Predicted Instance Embedding (raw)"
            )
            if self._settings.ground_truth_instance_embeddings:
                column_schemas["ground_truth_instance_embedding_raw"] = _raw_instance_embeddings_schema(
                    c_raw, display_name="Ground Truth Instance Embedding (raw)"
                )
            self._raw_pred_emb = []
            self._raw_gt_emb = []

        if self._epoch is not None:
            column_schemas[TRAINING_PHASE] = training_phase_schema()

        # Add DDP rank column for distributed validation debugging
        if RANK >= 0:
            column_schemas["ddp_rank"] = tlc.Schema(
                display_name="DDP rank",
                value=tlc.Int32Value(),
                description="DDP rank that processed this sample",
                default_visible=False,
            )

        # Only RANK 0 (or single GPU) updates run status
        if RANK in {-1, 0}:
            self._run.set_status_collecting()

        self._metrics_writer = tlc.MetricsTableWriter(
            run_url=self._run.url,
            foreign_table_url=self.dataloader.dataset.table.url,
            column_schemas=column_schemas,
        )

        self._seen = 0

    @execute_when_collecting
    def _post_validation(self):
        """Clean up the validator after one validation pass.

        Finalizes the streaming metrics writer. When instance_embeddings_dim > 0,
        fits a reducer on the accumulated raw embeddings and rewrites the metrics
        table, replacing the raw embedding columns with reduced ones; the raw
        table is then deleted from disk and the reduced table is registered on
        the run. In DDP mode, gathers metrics_infos and input table URLs from
        all ranks to RANK 0, which then updates the run.
        """
        # Each rank finalizes its streaming writer. The raw table is written to
        # disk but NOT yet registered with the run (MetricsTableWriter.finalize
        # only writes; the integration calls run.update_metrics below).
        raw_table = self._metrics_writer.finalize()
        metrics_infos = self._metrics_writer.get_written_metrics_infos()
        input_table_url = self.dataloader.dataset.table.url.to_str()

        # TEMP(instance-embeddings): on each rank, fit reducer on local raw
        # embeddings, rewrite the just-finalized raw table into a reduced one,
        # and delete the raw table on disk. NOTE: under DDP this fits per-rank,
        # producing reduced spaces that are not comparable across ranks. Same
        # behavior as before this refactor — fixing it requires gathering raw
        # embeddings to RANK 0 for a single fit, broadcasting the fitted
        # reducer back, and per-rank rewrite. Tracked separately.
        if self._settings.instance_embeddings_dim > 0:
            reduced_table, reduced_metrics_infos = self._fit_and_rewrite(raw_table)
            # The raw table was never registered on the run; just remove it
            # from disk so we don't leave an orphaned directory under the run.
            self._delete_table_on_disk(raw_table)
            metrics_infos = reduced_metrics_infos

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
                tlc.ObjectRegistry._delete_object_from_caches(tlc.Url(metrics_info["url"]).to_absolute(self._run.url))

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
        self._raw_pred_emb = []
        self._raw_gt_emb = []

    def _fit_and_rewrite(self, raw_table):
        """TEMP(instance-embeddings): fit the reducer on accumulated raw
        embeddings, then rewrite the just-finalized metrics table into a new
        one with reduced ``predicted_instance_embedding`` /
        ``ground_truth_instance_embedding`` columns in place of the
        ``..._raw`` columns.

        Iterates *raw_table* sample-by-sample (RLE-encoded heavy fields decode
        to numpy and re-encode through ``add_batch``; this is bounded memory
        because only one image's worth of samples is materialized at a time
        before being flushed in a small batch). Cross-split fit/transform is
        preserved: if ``settings._fitted_instance_reducer`` is set (from an
        earlier split in the same ``collect()`` call), reused as-is.

        Goes away when core 3LC reduces variable-length embedding list
        columns server-side. The raw column it leaves behind is already tagged
        ``NUMBER_ROLE_NN_EMBEDDING`` for that future flow.
        """
        from tlc_ultralytics.utils._instance_reduce import (
            _reduce_instance_embeddings,
            _transform_instance_embeddings,
        )

        n = self._settings.instance_embeddings_dim
        method = self._settings.image_embeddings_reducer
        reducer_args = self._settings.image_embeddings_reducer_args or {}
        progress_cb = getattr(self._settings, "_reduction_progress_callback", None)
        existing_reducer = getattr(self._settings, "_fitted_instance_reducer", None)

        if existing_reducer is not None:
            pred_reduced = _transform_instance_embeddings(
                self._raw_pred_emb, existing_reducer, n_components=n,
                progress_callback=progress_cb, label="predicted",
            )
            reducer = existing_reducer
        else:
            pred_reduced, reducer = _reduce_instance_embeddings(
                self._raw_pred_emb, method=method, n_components=n,
                progress_callback=progress_cb, **reducer_args,
            )
            if reducer is not None:
                self._settings._fitted_instance_reducer = reducer

        if self._settings.ground_truth_instance_embeddings and self._raw_gt_emb:
            if reducer is not None:
                gt_reduced = _transform_instance_embeddings(
                    self._raw_gt_emb, reducer, n_components=n,
                    progress_callback=progress_cb, label="ground-truth",
                )
            else:
                gt_reduced = [
                    np.empty((0, n), dtype=np.float32) for _ in self._raw_gt_emb
                ]
        else:
            gt_reduced = None

        # Drop raw buffers — fit is done and we re-read raw values from the
        # source table during rewrite (where they're tiny pyarrow lists, not
        # numpy arrays).
        self._raw_pred_emb = []
        self._raw_gt_emb = []

        # Build destination schema: source columns minus the raw embedding
        # columns, plus the reduced ones.
        src_schema_values = dict(raw_table.rows_schema.values)
        src_schema_values.pop("predicted_instance_embedding_raw", None)
        src_schema_values.pop("ground_truth_instance_embedding_raw", None)
        src_schema_values["predicted_instance_embedding"] = _instance_embeddings_list_schema(
            n, display_name=f"Predicted Instance Embedding ({n}D)"
        )
        if gt_reduced is not None:
            src_schema_values["ground_truth_instance_embedding"] = _instance_embeddings_list_schema(
                n, display_name=f"Ground Truth Instance Embedding ({n}D)"
            )

        dst_writer = tlc.MetricsTableWriter(
            run_url=self._run.url,
            foreign_table_url=self.dataloader.dataset.table.url,
            column_schemas=src_schema_values,
        )

        raw_pred_key = "predicted_instance_embedding_raw"
        raw_gt_key = "ground_truth_instance_embedding_raw"
        BATCH_SIZE = 32

        chunk: dict[str, list] = {}
        chunk_size = 0
        pred_offset = 0
        gt_offset = 0

        def flush() -> None:
            nonlocal chunk, chunk_size, pred_offset, gt_offset
            if chunk_size == 0:
                return
            chunk["predicted_instance_embedding"] = [
                arr.astype(np.float32).tolist()
                for arr in pred_reduced[pred_offset : pred_offset + chunk_size]
            ]
            pred_offset += chunk_size
            if gt_reduced is not None:
                chunk["ground_truth_instance_embedding"] = [
                    arr.astype(np.float32).tolist()
                    for arr in gt_reduced[gt_offset : gt_offset + chunk_size]
                ]
                gt_offset += chunk_size
            dst_writer.add_batch(chunk)
            chunk = {}
            chunk_size = 0

        for sample in raw_table:
            for col, val in sample.items():
                if col == raw_pred_key or col == raw_gt_key:
                    continue
                chunk.setdefault(col, []).append(val)
            chunk_size += 1
            if chunk_size >= BATCH_SIZE:
                flush()
        flush()

        dst_table = dst_writer.finalize()
        return dst_table, dst_writer.get_written_metrics_infos()

    def _delete_table_on_disk(self, table: tlc.Table) -> None:
        """Best-effort removal of an unregistered intermediate metrics table.

        We keep the raw streaming table around just long enough to read it
        back during ``_fit_and_rewrite``; after that it's an orphaned
        directory under the run that nothing references.
        """
        try:
            tlc.ObjectRegistry._delete_object_from_caches(table.url)
        except Exception:
            pass
        try:
            table.url.delete()
        except Exception as exc:
            LOGGER.warning(
                f"{TLC_COLORSTR}Failed to delete intermediate raw metrics table at {table.url}: {exc}"
            )

    def _write_per_class_metrics_tables(self) -> None:
        if self.args.task not in ("detect", "segment", "obb"):
            # Per-class metrics currently only supported for detection, segmentation, and obb tasks
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
