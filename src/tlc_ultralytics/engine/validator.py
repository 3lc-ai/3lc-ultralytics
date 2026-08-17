from __future__ import annotations

import weakref
from typing import Any

import numpy as np
import tlc
import torch
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
    GROUND_TRUTH_INSTANCE_EMBEDDING,
    GROUND_TRUTH_INSTANCE_EMBEDDING_RAW,
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
    PREDICTED_INSTANCE_EMBEDDING,
    PREDICTED_INSTANCE_EMBEDDING_RAW,
    RECALL,
    RECALL_SEG,
    TLC_COLORSTR,
    TRAINING_PHASE,
)
from tlc_ultralytics.engine.utils import _handle_deprecated_column_name
from tlc_ultralytics.settings import Settings
from tlc_ultralytics.utils import image_embeddings_schema, training_phase_schema
from tlc_ultralytics.utils._rolling_writer import _RollingMetricsWriter
from tlc_ultralytics.utils.schemas import (
    _instance_embeddings_list_schema,
    _raw_instance_embeddings_schema,
    _reduced_image_embeddings_schema,
)


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
            None,
            column_name="label_column_name",
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
        # _update_metrics, written inline to the metrics tables, and discarded.
        # The reducer is fitted at end of pass from a sample read back from those
        # tables; _pred_instances_per_table tracks how many predicted instances
        # each flushed table holds so the sample can be allocated across tables
        # without re-reading them.
        self._instance_feature_map = None
        self._pred_instances_per_table: list[int] = []

        # Per-batch caches so _prepare_batch and _filter_top_predictions run once per image, shared by annotation
        # building and instance-embedding extraction.
        self._cur_pbatches: dict[int, Any] = {}
        self._cur_filtered_preds: dict[int, Any] = {}

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
        # COCO/LVIS JSON evaluation assumes on-disk annotation JSON files, which is incompatible with
        # current 3LC Tables. Disable with warning.
        # TODO: Consider override where evaluation is called from 3LC Table annotations
        if self.args.save_json:
            if RANK in {-1, 0}:
                LOGGER.warning(
                    f"{TLC_COLORSTR}save_json is not supported with 3LC datasets. COCO/LVIS JSON evaluation reads "
                    "on-disk annotation files that 3LC Tables don't have. Disabling it for this run, metrics are "
                    "collected into your 3LC Run instead."
                )
            self.args.save_json = False

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

        # Per-class metrics only on RANK 0 (uses aggregate metrics from gather_stats).
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

    def _compute_3lc_metrics(self, preds, batch) -> dict[str, Any]:
        """Compute 3LC metrics for a batch of predictions and targets"""
        raise NotImplementedError("Subclasses must implement this method.")

    def _filter_top_predictions(self, pred):
        """Filter a single image's predictions by confidence threshold, keeping the
        top max_det by confidence.

        Returns None if no predictions pass the threshold. All per-instance metrics
        columns must be filtered through this method so they stay index-aligned.
        """
        mask = pred["conf"] >= self._settings.conf_thres
        if not mask.any():
            return None

        filtered = {k: v[mask] for k, v in pred.items()}

        # Keep only top max_det predictions by confidence
        max_det = self._settings.max_det
        if len(filtered["conf"]) > max_det:
            topk = filtered["conf"].topk(max_det).indices
            filtered = {k: v[topk] for k, v in filtered.items()}

        return filtered

    def _prepared_batch(self, i, batch):
        """Return _prepare_batch(i, batch), cached for the current batch."""
        if i not in self._cur_pbatches:
            self._cur_pbatches[i] = self._prepare_batch(i, batch)
        return self._cur_pbatches[i]

    def _filtered_pred(self, i, pred):
        """Return _filter_top_predictions(pred), cached for the current batch.

        Caching ensures annotation building and instance-embedding extraction filter the
        same predictions once, keeping their per-instance columns index-aligned.
        """
        if i not in self._cur_filtered_preds:
            self._cur_filtered_preds[i] = self._filter_top_predictions(pred)
        return self._cur_filtered_preds[i]

    def _process_predictions(self, preds, batch):
        """Filter, scale, and build 3LC annotations for a batch of predictions."""
        results = []
        for i, pred in enumerate(preds):
            pbatch = self._prepared_batch(i, batch)
            h, w = pbatch["ori_shape"]

            filtered = self._filtered_pred(i, pred)
            if filtered is None:
                results.append(self._empty_annotation(h, w))
                continue

            scaled = self.scale_preds(filtered, pbatch)
            mapped_classes = [self.data["range_to_3lc_class"][int(c)] for c in scaled["cls"].tolist()]
            results.append(self._build_annotation(scaled, mapped_classes, h, w))

        return results

    def _add_embeddings_hook(self, model) -> int:
        """Add a hook to extract embeddings from the model, and infer the activation size"""
        raise NotImplementedError("Subclasses must implement this method.")

    def _add_instance_embeddings_hook(self, model) -> int:
        """Add a hook to capture class-discriminative feature maps for instance embeddings.

        By default, hooks into the classification branch (cv3) of the detection head,
        which produces features optimized for class discrimination rather than localization.
        All detection-based task heads (detect, segment, pose, obb) subclass ultralytics'
        Detect head, so this default applies to all of them. Uses a neck layer instead if
        instance_embeddings_layer is explicitly set.

        Returns the channel dimension size of the hooked layer(s).
        """
        if hasattr(model.model, "model"):
            model = model.model

        # If user explicitly set a layer index, use the neck-layer approach
        if self._settings.instance_embeddings_layer is not None:
            layer_index = self._settings.instance_embeddings_layer
            LOGGER.info(
                f"{TLC_COLORSTR}Using layer {layer_index} ({model.model[layer_index].type}) "
                "for instance embeddings extraction."
            )
            return self._add_feature_map_hook(model, layer_index)

        # Default: hook the cls branch (cv3) of the detection head for class-discriminative features
        return self._add_cls_head_hooks(model)

    def _add_feature_map_hook(self, model, layer_index: int) -> int:
        """Hook a single model layer, storing its output as the instance feature map."""
        from tlc_ultralytics.utils.embeddings import _infer_layer_channels

        weak_self = weakref.ref(self)  # Avoid circular reference (self <-> hook_fn)

        def hook_fn(_module, _input, output):
            weak_self()._instance_feature_map = output

        self._hook_handles.append(model.model[layer_index].register_forward_hook(hook_fn))
        return _infer_layer_channels(model.model[layer_index], layer_index)

    @staticmethod
    def _find_cls_head(model) -> torch.nn.ModuleList | None:
        """Find the cls head ModuleList from the detection head."""
        detect_head = model.model[-1]
        cv3 = getattr(detect_head, "cv3", None)
        if cv3 is not None:
            return cv3
        if hasattr(detect_head, "one2one"):
            return detect_head.one2one.get("cls_head")
        return None

    def _add_cls_head_hooks(self, model) -> int:
        """Hook the cls branch of the detection head at all FPN levels.

        Captures the penultimate layer output (before the final 1x1 conv to class logits)
        from each FPN level. These are resized to P3 resolution and concatenated into a
        single feature map stored in _instance_feature_map.

        Returns the total channel dimension across all levels.
        """
        import torch.nn.functional as F

        detect_head = model.model[-1]
        cv3 = self._find_cls_head(model)

        if cv3 is None:
            # Fallback to neck layer approach
            from tlc_ultralytics.utils.embeddings import _auto_detect_p3_layer

            layer_index = _auto_detect_p3_layer(model.model)
            LOGGER.info(
                f"{TLC_COLORSTR}No cls head found, falling back to neck layer {layer_index} "
                "for instance embeddings extraction."
            )
            return self._add_feature_map_hook(model, layer_index)

        # Hook penultimate sub-layer of each FPN level's cls branch
        # cv3[level] = Sequential([DWConv+Conv, DWConv+Conv, Conv2d])
        # We want [-2] (second DWConv+Conv block) — class-discriminative features
        hook_sub_index = len(cv3[0]) - 2

        # Read every level's output channel count up front. If we can't read it reliably for
        # all levels, fall back to the neck-layer path rather than guessing — a wrong count
        # would mislabel the raw embedding column and mismatch the captured feature map.
        try:
            level_channels = [cv3[level_idx][hook_sub_index][-1].conv.out_channels for level_idx in range(len(cv3))]
        except (AttributeError, IndexError, TypeError):
            from tlc_ultralytics.utils.embeddings import _auto_detect_p3_layer

            layer_index = _auto_detect_p3_layer(model.model)
            LOGGER.info(
                f"{TLC_COLORSTR}Could not read cls-head channel counts, falling back to neck layer "
                f"{layer_index} for instance embeddings extraction."
            )
            return self._add_feature_map_hook(model, layer_index)

        total_channels = sum(level_channels)
        level_features: list[torch.Tensor | None] = [None] * len(cv3)
        weak_self = weakref.ref(self)

        for level_idx in range(len(cv3)):
            target = cv3[level_idx][hook_sub_index]

            def make_hook(idx):
                def hook_fn(_module, _input, output):
                    level_features[idx] = output

                return hook_fn

            self._hook_handles.append(target.register_forward_hook(make_hook(level_idx)))

        def combine_hook(_module, _input, _output):
            self_ref = weak_self()
            if self_ref is None:
                return
            features = [f for f in level_features if f is not None]
            if not features:
                return
            target_size = features[0].shape[2:]
            resized = [
                F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
                if f.shape[2:] != target_size
                else f
                for f in features
            ]
            self_ref._instance_feature_map = torch.cat(resized, dim=1)

        self._hook_handles.append(detect_head.register_forward_hook(combine_hook))

        LOGGER.info(
            f"{TLC_COLORSTR}Using detection head cls branch (cv3) for instance embeddings "
            f"({len(cv3)} levels, {total_channels} total channels)."
        )
        return total_channels

    # Geometry used to pool the feature map into per-instance embeddings:
    # "bbox" pools with roi_align over xyxy boxes, "mask" with mask-weighted averaging.
    _instance_geometry_kind = "bbox"

    def _instance_regions(self, source, h: int, w: int, device) -> torch.Tensor:
        """Return one image's instance geometry, aligned with the feature map.

        ``source`` is either a filtered prediction dict or a prepared GT batch
        (both keyed the same way), or None when there are no instances. Both are
        in model-input (letterboxed) coords — the same spatial domain as the
        feature map. Returns [N, 4] xyxy bboxes (or [N, H, W] masks for
        subclasses with _instance_geometry_kind = "mask").
        """
        bboxes = source.get("bboxes") if source is not None else None
        if bboxes is None or bboxes.numel() == 0:
            return torch.empty((0, 4), device=device)
        return bboxes.to(device)

    def _extract_instance_embeddings(self, preds, batch, ground_truth: bool = False) -> list[np.ndarray]:
        """Extract per-instance raw embeddings from the captured feature map.

        For predictions, instances are filtered through _filter_top_predictions so
        the embeddings stay index-aligned with all other per-instance metrics
        columns. For ground truth (ground_truth=True), all annotations are used.
        """
        from tlc_ultralytics.utils.embeddings import (
            _extract_instance_embeddings_bbox,
            _extract_instance_embeddings_mask,
        )

        feature_map = self._instance_feature_map
        if feature_map is None:
            raise RuntimeError(
                "Instance embeddings are enabled but no feature map was captured during the forward "
                "pass. The embeddings hook may not have fired for this model; consider setting "
                "instance_embeddings_layer explicitly."
            )

        # The raw column schema was declared with this channel count; if the configured layer's
        # actual output width differs, its channel count was misread. Fail with an actionable
        # message instead of the opaque length error raised later when writing the batch.
        if feature_map.shape[1] != self._instance_embeddings_channel_size:
            raise RuntimeError(
                f"Instance embedding feature map has {feature_map.shape[1]} channels but the column schema "
                f"was declared with {self._instance_embeddings_channel_size}; the configured layer's channel "
                "count could not be determined correctly. Set instance_embeddings_layer to a different layer."
            )

        regions_list = []
        image_sizes = []
        for i, pred in enumerate(preds):
            pbatch = self._prepared_batch(i, batch)
            imgsz = pbatch["imgsz"]
            h, w = int(imgsz[0]), int(imgsz[1])
            image_sizes.append((h, w))

            source = pbatch if ground_truth else self._filtered_pred(i, pred)
            regions_list.append(self._instance_regions(source, h, w, feature_map.device))

        if self._instance_geometry_kind == "bbox":
            return _extract_instance_embeddings_bbox(feature_map, regions_list, image_sizes)
        return _extract_instance_embeddings_mask(feature_map, regions_list)

    def _extract_gt_instance_embeddings(self, preds, batch) -> list[np.ndarray]:
        """Extract per-instance raw embeddings for ground-truth annotations."""
        return self._extract_instance_embeddings(preds, batch, ground_truth=True)

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

        All metrics stream through a rolling writer that flushes to a new metrics table whenever its in-memory
        buffer exceeds `Settings.metrics_max_buffer_mb`, so the buffer itself does not grow with dataset size.
        When instance_embeddings_dim > 0, raw per-instance embeddings are written inline as
        `predicted_instance_embedding_raw` / `ground_truth_..._raw` columns and reduced from the written tables
        at end of pass, so they are not held in RAM here either.

        This bounds the metrics buffer only. Two other terms still scale with the pass and are outside this
        writer's reach: the dataloader worker processes, and Ultralytics' own `validator.metrics.stats`, which
        retains six numpy arrays per image so it can compute mAP over the full split (~4.5 MB per 1000 images).
        """
        batch_size = self._infer_batch_size(preds, batch)

        # Reset the per-batch caches; consumers below fill them lazily and share the result.
        self._cur_pbatches = {}
        self._cur_filtered_preds = {}

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

        # TEMP(instance-embeddings): write raw vectors inline; the end-of-pass
        # rewrite swaps them for the reduced column. This goes away when core
        # 3LC reduces variable-length embedding list columns server-side.
        if self._settings.instance_embeddings_dim > 0:
            raw_instance_embs = self._extract_instance_embeddings(preds, batch)
            batch_metrics[PREDICTED_INSTANCE_EMBEDDING_RAW] = [
                (a.tolist() if a.size else []) for a in raw_instance_embs
            ]

            # Count predicted instances per destination table for the end-of-pass reducer-fit
            # sampling. Read the table index before add_batch — the writer may roll inside it.
            table_index = self._metrics_writer.num_flushed_tables
            while len(self._pred_instances_per_table) <= table_index:
                self._pred_instances_per_table.append(0)
            self._pred_instances_per_table[table_index] += sum(a.shape[0] for a in raw_instance_embs)

            if self._settings.ground_truth_instance_embeddings:
                raw_gt_embs = self._extract_gt_instance_embeddings(preds, batch)
                batch_metrics[GROUND_TRUTH_INSTANCE_EMBEDDING_RAW] = [
                    (a.tolist() if a.size else []) for a in raw_gt_embs
                ]

        self._metrics_writer.add_batch(batch_metrics)
        self._seen += batch_size

    @execute_when_collecting
    def _pre_validation(self, model):
        """Prepare the validator for metrics collection.

        In DDP mode, each rank prepares its own metrics writer.
        """
        # Prepare the loss function before declaring schemas: tasks that don't support per-sample loss
        # disable `collect_loss` here, and the loss columns must not be declared in that case.
        self._prepare_loss_fn(model)

        column_schemas = {}
        column_schemas.update(self._get_metrics_schemas())  # Add task-specific metrics schema

        if self._settings.metrics_schemas:
            column_schemas.update(self._settings.metrics_schemas)

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
            # Declared channel count for the raw column; checked against the real feature map
            # width in _extract_instance_embeddings to catch a layer whose channels were misread.
            self._instance_embeddings_channel_size = c_raw
            column_schemas[PREDICTED_INSTANCE_EMBEDDING_RAW] = _raw_instance_embeddings_schema(
                c_raw, display_name="Predicted Instance Embedding (raw)"
            )
            if self._settings.ground_truth_instance_embeddings:
                column_schemas[GROUND_TRUTH_INSTANCE_EMBEDDING_RAW] = _raw_instance_embeddings_schema(
                    c_raw, display_name="Ground Truth Instance Embedding (raw)"
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

        # Rolling writer: flushes to a new metrics table whenever the in-memory buffer
        # exceeds the configured threshold, bounding peak host memory during collection.
        # The flushed tables share a stream and are joined in the Dashboard.
        self._metrics_writer = _RollingMetricsWriter(
            run_url=self._run.url,
            foreign_table_url=self.dataloader.dataset.table.url,
            schema=column_schemas,
            max_buffer_bytes=self._settings.metrics_max_buffer_mb * 1024 * 1024,
        )

        self._pred_instances_per_table = []
        self._seen = 0

    @execute_when_collecting
    def _post_validation(self):
        """Clean up the validator after one validation pass.

        Finalizes the rolling metrics writer, which may have flushed several
        tables during the pass. When instance or image embeddings are enabled,
        reducers are fitted on bounded samples of raw embeddings read back from
        the flushed tables, and each table is rewritten one at a time with the
        raw embedding columns replaced by reduced ones; the raw tables are then
        deleted from disk. In DDP mode, table urls and metrics infos are
        gathered from all ranks to RANK 0, which performs the reduction/rewrite
        for every rank's tables (they live on shared storage) and updates the
        run.
        """
        # Each rank finalizes its rolling writer. Flushed tables were already
        # registered on the run as they were written (the later
        # run.update_metrics call below deduplicates).
        table_urls, metrics_infos = self._metrics_writer.finalize()
        table_url_strs = [url.to_str() for url in table_urls]
        pred_instance_counts = list(self._pred_instances_per_table)
        pred_instance_counts += [0] * (len(table_url_strs) - len(pred_instance_counts))
        input_table_url = self.dataloader.dataset.table.url.to_str()

        # Gather every rank's table urls and metrics infos to RANK 0 in DDP mode
        table_url_strs, metrics_infos, pred_instance_counts, input_table_urls = self._gather_written_tables(
            table_url_strs, metrics_infos, pred_instance_counts, input_table_url
        )

        # Only RANK 0 (or single GPU) reduces embeddings and updates the run
        if RANK in {-1, 0}:
            # TEMP(embeddings): reduce the raw instance/image embeddings and
            # rewrite each just-finalized raw table into a reduced one, then
            # delete the raw tables on disk. Under DDP, RANK 0 rewrites all
            # ranks' tables.
            if self._settings.instance_embeddings_dim > 0 or self._settings.image_embeddings_dim > 0:
                row_counts = [info["row_count"] for info in metrics_infos]
                reduced_metrics_infos = self._reduce_and_rewrite_raw_tables(
                    table_url_strs, pred_instance_counts, row_counts
                )
                if reduced_metrics_infos is not None:
                    # Flushing registered the raw tables on the run; remove those
                    # registrations and the tables themselves on disk so the run
                    # doesn't reference deleted tables.
                    self._remove_metrics_infos_from_run(metrics_infos)
                    for url_str in table_url_strs:
                        self._delete_table_on_disk(tlc.Url(url_str))
                    metrics_infos = reduced_metrics_infos

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
        self._pred_instances_per_table = []

    @staticmethod
    def _gather_written_tables(table_url_strs, metrics_infos, pred_instance_counts, input_table_url):
        """Gather every rank's written-table urls, metrics infos and instance counts to RANK 0.

        Single-process (RANK == -1): passthrough. Under DDP, RANK 0 receives the
        flattened lists from all ranks (and performs all downstream run updates);
        other ranks get their inputs back unchanged but do not act on them.

        Returns (table_url_strs, metrics_infos, pred_instance_counts, input_table_urls).
        """
        if RANK < 0:
            return table_url_strs, metrics_infos, pred_instance_counts, [input_table_url]

        world_size = dist.get_world_size()  # type: ignore[possibly-missing-attribute]
        payload = (table_url_strs, metrics_infos, pred_instance_counts, input_table_url)
        # gather_object fills this in place on RANK 0 with each rank's payload tuple;
        # annotate so the post-gather element type (not None) is known to the type checker.
        gathered: list[tuple[list[str], list, list[int], str]] | None = [None] * world_size if RANK == 0 else None
        dist.gather_object(payload, gathered, dst=0)  # type: ignore[possibly-missing-attribute]

        input_table_urls: list[str] = []
        if RANK == 0:
            assert gathered is not None
            table_url_strs, metrics_infos, pred_instance_counts = [], [], []
            for rank_table_urls, rank_metrics_infos, rank_counts, _ in gathered:
                table_url_strs.extend(rank_table_urls)
                metrics_infos.extend(rank_metrics_infos)
                pred_instance_counts.extend(rank_counts)

            # Collect unique input table URLs (should all be the same in distributed validation)
            input_table_urls = list({rank_input_url for _, _, _, rank_input_url in gathered})

        return table_url_strs, metrics_infos, pred_instance_counts, input_table_urls

    def _reduce_and_rewrite_raw_tables(self, raw_table_urls, pred_instance_counts, row_counts):
        """TEMP(embeddings): reduce raw embeddings from the flushed raw tables and
        rewrite each into a table with reduced embedding columns.

        Instance embeddings: the ``..._raw`` columns are replaced by reduced
        ``predicted_instance_embedding`` / ``ground_truth_instance_embedding``
        columns. Image embeddings: the raw ``embeddings`` column is replaced by
        ``embeddings_{reducer}``, the same column name core 3LC's reduction
        produces.

        Each reducer is fitted on a uniform random sample of at most the
        configured fit sample size drawn across the raw tables (or reused from
        an earlier split of the same run via the per-run reducer registry), and
        every row/instance is projected into the fitted space with the
        reducer's transform. Tables are processed one at a time and evicted
        from the object caches when done, so peak memory is bounded by a single
        table regardless of dataset size.

        Returns the metrics infos of the rewritten tables, or None when no
        rewrite was performed (no tables, or the image-embeddings fit failed
        while instance embeddings are disabled — the raw tables are then kept).
        Goes away when core 3LC reduces these columns server-side; the raw
        columns are already tagged ``NUMBER_ROLE_NN_EMBEDDING`` for that future
        flow.
        """
        if not raw_table_urls:
            return None

        # Fit on this split's sampled embeddings, or reuse the run's cross-split
        # reducers if an earlier split already fitted them.
        instance_reducer = (
            self._fit_or_reuse_instance_reducer(raw_table_urls, pred_instance_counts)
            if self._settings.instance_embeddings_dim > 0
            else None
        )
        image_reducer = (
            self._fit_or_reuse_image_reducer(raw_table_urls, row_counts)
            if self._settings.image_embeddings_dim > 0
            else None
        )

        # Nothing to rewrite: instance embeddings are off and the image fit failed, so
        # keep the raw tables (with the raw 'embeddings' column) as they were written.
        if self._settings.instance_embeddings_dim == 0 and image_reducer is None:
            return None

        LOGGER.info(f"{TLC_COLORSTR}Reducing embeddings across {len(raw_table_urls)} metrics table(s)...")

        dst_writer = None
        for table_number, url_str in enumerate(raw_table_urls, start=1):
            LOGGER.debug(f"{TLC_COLORSTR}Reducing embeddings for metrics table {table_number}/{len(raw_table_urls)}.")
            raw_table = tlc.Table.from_url(url_str)
            if dst_writer is None:
                dst_writer = self._reduced_table_writer(raw_table, image_reducer)
            self._rewrite_raw_table(raw_table, instance_reducer, image_reducer, dst_writer)
            # Evict the processed table so only one raw table is in RAM at a time.
            ObjectRegistry._delete_object_from_caches(raw_table.url)
            del raw_table

        assert dst_writer is not None
        _, reduced_metrics_infos = dst_writer.finalize()
        LOGGER.info(f"{TLC_COLORSTR}Done reducing embeddings.")
        return reduced_metrics_infos

    def _fit_or_reuse_instance_reducer(self, raw_table_urls, pred_instance_counts):
        """TEMP(instance-embeddings): return the run's instance reducer, fitting it
        on sampled predicted embeddings when this is the first split to reduce.

        Returns None when there are no predicted instances to fit on (a warning
        is logged; the reduced columns are then written empty).
        """
        from tlc_ultralytics.utils._instance_reduce import (
            _fit_embeddings_reducer,
            _get_fitted_reducer,
            _read_raw_embedding_column,
            _set_fitted_reducer,
        )

        run_url_str = self._run.url.to_str()
        reducer = _get_fitted_reducer(run_url_str, "instance")
        if reducer is not None:
            return reducer

        sample = self._sample_embeddings_across_tables(
            raw_table_urls,
            pred_instance_counts,
            self._settings.instance_embeddings_fit_sample_size,
            lambda table: _read_raw_embedding_column(table, PREDICTED_INSTANCE_EMBEDDING_RAW)[0],
        )
        if sample is None:
            msg = (
                "No predicted instances were available to fit the instance-embeddings reducer "
                f"for this split (conf_thres={self._settings.conf_thres}); instance embeddings "
                "will be empty."
            )
            if self._settings.ground_truth_instance_embeddings:
                msg += " Ground-truth instance embeddings will be empty too."
            LOGGER.warning(f"{TLC_COLORSTR}{msg}")
            return None

        reducer_kwargs = dict(self._settings.instance_embeddings_reducer_kwargs or {})
        reducer_kwargs.pop("n_components", None)  # provided via instance_embeddings_dim
        reducer = _fit_embeddings_reducer(
            sample,
            method=self._settings.instance_embeddings_reducer,
            n_components=self._settings.instance_embeddings_dim,
            progress_callback=getattr(self._settings, "_reduction_progress_callback", None),
            label="instance",
            **reducer_kwargs,
        )
        _set_fitted_reducer(run_url_str, "instance", reducer)
        return reducer

    def _fit_or_reuse_image_reducer(self, raw_table_urls, row_counts):
        """TEMP(image-embeddings): return the run's image reducer, fitting it on
        sampled image embeddings when this is the first split to reduce.

        Returns None when the fit fails or there is nothing to fit on (a warning
        is logged; the raw ``embeddings`` column is then kept as written).
        """
        from tlc_ultralytics.utils._instance_reduce import (
            _fit_embeddings_reducer,
            _get_fitted_reducer,
            _read_image_embedding_column,
            _set_fitted_reducer,
        )

        run_url_str = self._run.url.to_str()
        reducer = _get_fitted_reducer(run_url_str, "image")
        if reducer is not None:
            return reducer

        sample = self._sample_embeddings_across_tables(
            raw_table_urls,
            row_counts,
            self._settings.image_embeddings_fit_sample_size,
            lambda table: _read_image_embedding_column(table, "embeddings"),
        )
        if sample is None:
            return None

        reducer_args = dict(self._settings.image_embeddings_reducer_args or {})
        reducer_args.pop("n_components", None)  # provided via image_embeddings_dim
        try:
            reducer = _fit_embeddings_reducer(
                sample,
                method=self._settings.image_embeddings_reducer,
                n_components=self._settings.image_embeddings_dim,
                progress_callback=getattr(self._settings, "_reduction_progress_callback", None),
                label="image",
                **reducer_args,
            )
        except Exception as exc:
            LOGGER.warning(
                f"{TLC_COLORSTR}Fitting the image-embeddings reducer failed: {exc}. The raw "
                "'embeddings' column is kept in the written metrics tables."
            )
            return None

        _set_fitted_reducer(run_url_str, "image", reducer)
        return reducer

    def _sample_embeddings_across_tables(self, raw_table_urls, per_table_counts, sample_size, read_matrix):
        """TEMP(embeddings): draw a uniform random sample of embedding vectors
        across the flushed raw tables for fitting a reducer.

        ``per_table_counts`` gives the number of candidate vectors per table
        (predicted instances for instance embeddings, rows for image
        embeddings), so the sample can be allocated across tables (multivariate
        hypergeometric — exactly uniform over all vectors) with only one table
        loaded at a time. ``read_matrix`` reads one table's candidate vectors
        as a [N, C] matrix (or None).

        Returns a [K, C] float32 matrix, or None when there are no candidates.
        """
        counts = np.asarray(per_table_counts, dtype=np.int64)
        total = int(counts.sum())
        if total == 0:
            return None

        rng = np.random.default_rng()
        per_table_sample_sizes = rng.multivariate_hypergeometric(counts, min(sample_size, total))

        parts = []
        for url_str, table_sample_size in zip(raw_table_urls, per_table_sample_sizes, strict=True):
            if table_sample_size == 0:
                continue
            table = tlc.Table.from_url(url_str)
            matrix = read_matrix(table)
            ObjectRegistry._delete_object_from_caches(table.url)
            del table
            if matrix is None:
                continue
            table_sample_size = min(int(table_sample_size), matrix.shape[0])
            indices = np.sort(rng.choice(matrix.shape[0], size=table_sample_size, replace=False))
            parts.append(matrix[indices].copy())  # copy so the full matrix can be freed

        if not parts:
            return None
        return np.concatenate(parts, axis=0)

    def _reduced_table_writer(self, raw_table, image_reducer):
        """TEMP(embeddings): build the rolling writer for the reduced tables —
        the raw table's schema with the raw embedding columns replaced by
        reduced ones."""
        n = self._settings.instance_embeddings_dim
        schema_values = dict(raw_table.rows_schema.values)
        has_pred = schema_values.pop(PREDICTED_INSTANCE_EMBEDDING_RAW, None) is not None
        has_gt = schema_values.pop(GROUND_TRUTH_INSTANCE_EMBEDDING_RAW, None) is not None
        if has_pred:
            schema_values[PREDICTED_INSTANCE_EMBEDDING] = _instance_embeddings_list_schema(
                n, display_name=f"Predicted Instance Embedding ({n}D)"
            )
        if has_gt:
            schema_values[GROUND_TRUTH_INSTANCE_EMBEDDING] = _instance_embeddings_list_schema(
                n, display_name=f"Ground Truth Instance Embedding ({n}D)"
            )

        if image_reducer is not None and "embeddings" in schema_values:
            schema_values.pop("embeddings")
            method = self._settings.image_embeddings_reducer
            schema_values[f"embeddings_{method}"] = _reduced_image_embeddings_schema(
                self._settings.image_embeddings_dim, method
            )

        return _RollingMetricsWriter(
            run_url=self._run.url,
            foreign_table_url=self.dataloader.dataset.table.url,
            schema=schema_values,
            max_buffer_bytes=self._settings.metrics_max_buffer_mb * 1024 * 1024,
        )

    def _rewrite_raw_table(self, raw_table, instance_reducer, image_reducer, dst_writer):
        """TEMP(embeddings): copy one raw table into *dst_writer* with the raw
        embedding columns replaced by reduced ones.

        Raw embeddings are read columnar (no per-row decode) and transformed in
        chunks; the remaining columns are copied row by row (RLE-encoded heavy
        fields decode to numpy and re-encode through ``add_batch``, one small
        batch at a time). Memory is bounded by this one table plus the
        destination writer's buffer.
        """
        reduced_columns, skip_columns = self._compute_reduced_columns(raw_table, instance_reducer, image_reducer)

        BATCH_SIZE = 32

        chunk: dict[str, list] = {}
        chunk_size = 0
        offset = 0

        def flush() -> None:
            nonlocal chunk, chunk_size, offset
            if chunk_size == 0:
                return
            for column_name, per_row_values in reduced_columns.items():
                chunk[column_name] = per_row_values[offset : offset + chunk_size]
            dst_writer.add_batch(chunk)
            offset += chunk_size
            chunk = {}
            chunk_size = 0

        for sample in raw_table:
            for col, val in sample.items():
                if col in skip_columns:
                    continue
                chunk.setdefault(col, []).append(val)
            chunk_size += 1
            if chunk_size >= BATCH_SIZE:
                flush()
        flush()

    def _compute_reduced_columns(self, raw_table, instance_reducer, image_reducer):
        """TEMP(embeddings): compute one raw table's reduced embedding columns.

        Returns (reduced_columns, skip_columns): the output column names mapped
        to per-row lists of reduced values, and the raw source columns to drop
        while copying the table's remaining columns.

        Reduced embeddings are written back onto rows positionally, so row
        counts are asserted — a reorder would silently misalign embeddings.
        """
        from tlc_ultralytics.utils._instance_reduce import _read_image_embedding_column, _transform_embeddings

        progress_cb = getattr(self._settings, "_reduction_progress_callback", None)
        show_bar = self._should_show_reduction_bar()
        num_rows = len(raw_table)
        reduced_columns: dict[str, list] = {}
        skip_columns: set[str] = set()

        if PREDICTED_INSTANCE_EMBEDDING_RAW in raw_table.rows_schema.values:
            pred_reduced = self._transform_raw_column(
                raw_table, PREDICTED_INSTANCE_EMBEDDING_RAW, instance_reducer, progress_cb, show_bar, label="predicted"
            )
            assert len(pred_reduced) == num_rows, (
                f"Instance embedding row count mismatch: {len(pred_reduced)} reduced entries for {num_rows} rows."
            )
            reduced_columns[PREDICTED_INSTANCE_EMBEDDING] = [arr.astype(np.float32).tolist() for arr in pred_reduced]
            skip_columns.add(PREDICTED_INSTANCE_EMBEDDING_RAW)

        if GROUND_TRUTH_INSTANCE_EMBEDDING_RAW in raw_table.rows_schema.values:
            gt_reduced = self._transform_raw_column(
                raw_table,
                GROUND_TRUTH_INSTANCE_EMBEDDING_RAW,
                instance_reducer,
                progress_cb,
                show_bar,
                label="ground-truth",
            )
            reduced_columns[GROUND_TRUTH_INSTANCE_EMBEDDING] = [arr.astype(np.float32).tolist() for arr in gt_reduced]
            skip_columns.add(GROUND_TRUTH_INSTANCE_EMBEDDING_RAW)

        if image_reducer is not None and "embeddings" in raw_table.rows_schema.values:
            matrix = _read_image_embedding_column(raw_table, "embeddings")
            if matrix is not None:
                image_reduced = _transform_embeddings(
                    [matrix],
                    image_reducer,
                    n_components=self._settings.image_embeddings_dim,
                    progress_callback=progress_cb,
                    label="image",
                    show_progress_bar=show_bar,
                )[0]
                assert len(image_reduced) == num_rows, (
                    f"Image embedding row count mismatch: {len(image_reduced)} reduced entries for {num_rows} rows."
                )
                reduced_image_column = f"embeddings_{self._settings.image_embeddings_reducer}"
                reduced_columns[reduced_image_column] = [row.tolist() for row in image_reduced]
                skip_columns.add("embeddings")

        return reduced_columns, skip_columns

    def _should_show_reduction_bar(self) -> bool:
        """Whether to draw a default tqdm bar for the transform step.

        Only when no reduction progress callback was supplied (the caller drives its own
        reporting otherwise), on the main process, and to an interactive stdout — so DDP ranks
        and non-interactive environments (logs, CI) stay quiet.
        """
        import sys

        if getattr(self._settings, "_reduction_progress_callback", None) is not None:
            return False
        return RANK in {-1, 0} and sys.stdout.isatty()

    def _transform_raw_column(self, raw_table, column_name, reducer, progress_callback, show_progress_bar, label):
        """TEMP(instance-embeddings): read one raw instance embedding column and
        project it into the reduced space, returning one [N_i, dim] array per
        table row.

        With no fitted reducer (no predicted instances anywhere in the split),
        every row gets an empty array so the reduced column is still written.
        """
        from tlc_ultralytics.utils._instance_reduce import (
            _read_raw_embedding_column,
            _transform_embeddings,
        )

        n = self._settings.instance_embeddings_dim
        matrix, per_row_counts = _read_raw_embedding_column(raw_table, column_name)

        if matrix is None or reducer is None:
            return [np.empty((0, n), dtype=np.float32) for _ in range(len(per_row_counts))]

        reduced = _transform_embeddings(
            [matrix],
            reducer,
            n_components=n,
            progress_callback=progress_callback,
            label=label,
            show_progress_bar=show_progress_bar,
        )[0]
        return np.split(reduced, np.cumsum(per_row_counts)[:-1])

    def _remove_metrics_infos_from_run(self, metrics_infos) -> None:
        """Remove metrics infos that ``MetricsTableWriter.finalize()`` auto-registered.

        As of tlc 3.x, ``finalize()`` registers its written table on the run.
        The intermediate raw metrics table is deleted after the reduced rewrite,
        so its registration must be removed to avoid a dangling reference.
        """
        removed_urls = {info["url"] for info in metrics_infos}
        remaining = [m for m in self._run.metrics if m["url"] not in removed_urls]
        if len(remaining) != len(self._run.metrics):
            self._run.update_attributes({"metrics": remaining})

    def _delete_table_on_disk(self, table_url: tlc.Url) -> None:
        """Best-effort removal of an unregistered intermediate metrics table.

        We keep the raw streaming tables around just long enough to read them
        back during ``_reduce_and_rewrite_raw_tables``; after that they're
        orphaned directories under the run that nothing references.
        """
        try:
            ObjectRegistry._delete_object_from_caches(table_url)
        except Exception:
            pass
        try:
            table_url.delete()
        except Exception as exc:
            LOGGER.warning(f"{TLC_COLORSTR}Failed to delete intermediate raw metrics table at {table_url}: {exc}")

    @execute_when_collecting
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
        table = metrics_writer.finalize()

        # Improve memory usage - don't cache the written table
        ObjectRegistry._delete_object_from_caches(table.url)

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
