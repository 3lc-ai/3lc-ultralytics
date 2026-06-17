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

    def _process_predictions(self, preds, batch):
        """Filter, scale, and build 3LC annotations for a batch of predictions."""
        results = []
        for i, pred in enumerate(preds):
            pbatch = self._prepare_batch(i, batch)
            h, w = pbatch["ori_shape"]

            filtered = self._filter_top_predictions(pred)
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
        level_features: list[torch.Tensor | None] = [None] * len(cv3)
        weak_self = weakref.ref(self)

        total_channels = 0
        for level_idx in range(len(cv3)):
            target = cv3[level_idx][hook_sub_index]
            try:
                total_channels += target[-1].conv.out_channels
            except (AttributeError, IndexError, TypeError):
                total_channels += detect_head.nc

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

        regions_list = []
        image_sizes = []
        for i, pred in enumerate(preds):
            pbatch = self._prepare_batch(i, batch)
            imgsz = pbatch["imgsz"]
            h, w = int(imgsz[0]), int(imgsz[1])
            image_sizes.append((h, w))

            source = pbatch if ground_truth else self._filter_top_predictions(pred)
            regions_list.append(self._instance_regions(source, h, w, feature_map.device))

        if self._instance_geometry_kind == "bbox":
            return _extract_instance_embeddings_bbox(feature_map, regions_list, image_sizes)
        return _extract_instance_embeddings_mask(feature_map, regions_list, image_sizes)

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

        When instance_embeddings_dim > 0, raw per-instance embeddings are
        written inline as ``predicted_instance_embedding_raw`` / ``..._raw_gt``
        and held in a small in-RAM list for fitting the reducer at end-of-pass.
        Heavy fields like segmentation masks RLE-encode at handoff via
        ``MetricsTableWriter.add_batch``, so peak memory stays bounded.
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

        # TEMP(instance-embeddings): write raw vectors inline; the end-of-pass
        # rewrite swaps them for the reduced column. This goes away when core
        # 3LC reduces variable-length embedding list columns server-side.
        if self._settings.instance_embeddings_dim > 0:
            raw_instance_embs = self._extract_instance_embeddings(preds, batch)
            batch_metrics[PREDICTED_INSTANCE_EMBEDDING_RAW] = [
                (a.tolist() if a.size else []) for a in raw_instance_embs
            ]
            self._raw_pred_emb.extend(raw_instance_embs)

            if self._settings.ground_truth_instance_embeddings:
                raw_gt_embs = self._extract_gt_instance_embeddings(preds, batch)
                batch_metrics[GROUND_TRUTH_INSTANCE_EMBEDDING_RAW] = [
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
            column_schemas[PREDICTED_INSTANCE_EMBEDDING_RAW] = _raw_instance_embeddings_schema(
                c_raw, display_name="Predicted Instance Embedding (raw)"
            )
            if self._settings.ground_truth_instance_embeddings:
                column_schemas[GROUND_TRUTH_INSTANCE_EMBEDDING_RAW] = _raw_instance_embeddings_schema(
                    c_raw, display_name="Ground Truth Instance Embedding (raw)"
                )
            self._raw_pred_emb = []
            self._raw_gt_emb = []

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

        Finalizes the streaming metrics writer. When instance_embeddings_dim > 0,
        fits a reducer on the accumulated raw embeddings and rewrites the metrics
        table, replacing the raw embedding columns with reduced ones; the raw
        table is then deleted from disk and the reduced table is registered on
        the run. In DDP mode, gathers metrics_infos and input table URLs from
        all ranks to RANK 0, which then updates the run.
        """
        # Each rank finalizes its streaming writer. As of tlc 3.x, finalize()
        # also registers the written table on the run (the later
        # run.update_metrics call below deduplicates).
        raw_table = self._metrics_writer.finalize()
        metrics_infos = self._metrics_writer.get_written_metrics_infos()
        input_table_url = self.dataloader.dataset.table.url.to_str()

        # TEMP(instance-embeddings): reduce the raw embeddings and rewrite the
        # just-finalized raw table into a reduced one, then delete the raw
        # table on disk. Under DDP the reducer is fitted once on RANK 0 from
        # all ranks' gathered embeddings (see
        # _compute_reduced_instance_embeddings), so the reduced spaces are
        # shared across ranks; each rank rewrites its own table.
        if self._settings.instance_embeddings_dim > 0:
            raw_metrics_infos = metrics_infos
            _, reduced_metrics_infos = self._fit_and_rewrite(raw_table)
            # finalize() registered the raw table on the run; remove that
            # registration and the table itself on disk so the run doesn't
            # reference a deleted table.
            self._remove_metrics_infos_from_run(raw_metrics_infos)
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
        self._raw_pred_emb = []
        self._raw_gt_emb = []

    def _reduce_raw_instance_embeddings(self, raw_pred, raw_gt):
        """TEMP(instance-embeddings): reduce raw per-image embedding lists in-process.

        Fits the configured reducer on the predicted embeddings (or reuses the
        run's cross-split reducer if an earlier split already fitted one) and
        projects ground-truth embeddings into the same space.

        Returns (pred_reduced, gt_reduced), where gt_reduced is None when GT
        embeddings are not collected.
        """
        from tlc_ultralytics.utils._instance_reduce import (
            _get_fitted_reducer,
            _reduce_instance_embeddings,
            _set_fitted_reducer,
            _transform_instance_embeddings,
        )

        n = self._settings.instance_embeddings_dim
        method = self._settings.instance_embeddings_reducer
        reducer_args = self._settings.instance_embeddings_reducer_args or {}
        progress_cb = getattr(self._settings, "_reduction_progress_callback", None)
        existing_reducer = _get_fitted_reducer(self._run.url.to_str())

        if existing_reducer is not None:
            pred_reduced = _transform_instance_embeddings(
                raw_pred,
                existing_reducer,
                n_components=n,
                progress_callback=progress_cb,
                label="predicted",
            )
            reducer = existing_reducer
        else:
            pred_reduced, reducer = _reduce_instance_embeddings(
                raw_pred,
                method=method,
                n_components=n,
                progress_callback=progress_cb,
                **reducer_args,
            )
            if reducer is not None:
                _set_fitted_reducer(self._run.url.to_str(), reducer)

        if self._settings.ground_truth_instance_embeddings and raw_gt:
            if reducer is not None:
                gt_reduced = _transform_instance_embeddings(
                    raw_gt,
                    reducer,
                    n_components=n,
                    progress_callback=progress_cb,
                    label="ground-truth",
                )
            else:
                gt_reduced = [np.empty((0, n), dtype=np.float32) for _ in raw_gt]
        else:
            gt_reduced = None

        return pred_reduced, gt_reduced

    def _compute_reduced_instance_embeddings(self):
        """TEMP(instance-embeddings): reduce this rank's accumulated raw embeddings.

        Single-process: fit/transform locally. Under DDP, raw embeddings from
        all ranks are gathered to RANK 0, which fits a single reducer (or
        reuses the cross-split one), transforms every rank's embeddings into
        the one shared space, and scatters each rank its reduced share. The
        fitted reducer stays on RANK 0 only (it is not broadcast — fitted
        PaCMAP reducers are not picklable), which is sufficient since later
        splits gather to RANK 0 again.
        """
        if RANK < 0:
            return self._reduce_raw_instance_embeddings(self._raw_pred_emb, self._raw_gt_emb)

        # DDP: gather per-rank raw embeddings to RANK 0
        world_size = dist.get_world_size()  # type: ignore[possibly-missing-attribute]
        # gather_object fills this in place on RANK 0 with each rank's
        # (raw_pred, raw_gt) payload; annotate so the post-gather element type is known.
        gathered: list[tuple[list[np.ndarray], list[np.ndarray]]] | None = (
            [None] * world_size if RANK == 0 else None
        )
        dist.gather_object((self._raw_pred_emb, self._raw_gt_emb), gathered, dst=0)  # type: ignore[possibly-missing-attribute]

        per_rank_reduced = None
        if RANK == 0:
            assert gathered is not None
            all_pred = [arr for rank_pred, _ in gathered for arr in rank_pred]
            all_gt = [arr for _, rank_gt in gathered for arr in rank_gt]
            pred_reduced_all, gt_reduced_all = self._reduce_raw_instance_embeddings(all_pred, all_gt)
            per_rank_reduced = self._split_reduced_by_rank(gathered, pred_reduced_all, gt_reduced_all)

        output = [None]
        dist.scatter_object_list(output, per_rank_reduced, src=0)  # type: ignore[possibly-missing-attribute]
        return output[0]

    @staticmethod
    def _split_reduced_by_rank(gathered, pred_reduced_all, gt_reduced_all):
        """Re-split flattened reduced per-image lists by each rank's image count.

        ``gathered`` is the list of (raw_pred, raw_gt) per-rank payloads whose
        lengths define the split points. Returns one (pred_reduced, gt_reduced)
        tuple per rank, with gt_reduced None when GT embeddings are not collected.
        """
        per_rank_reduced = []
        pred_offset = 0
        gt_offset = 0
        for rank_pred, rank_gt in gathered:
            n_pred, n_gt = len(rank_pred), len(rank_gt)
            per_rank_reduced.append(
                (
                    pred_reduced_all[pred_offset : pred_offset + n_pred],
                    gt_reduced_all[gt_offset : gt_offset + n_gt] if gt_reduced_all is not None else None,
                )
            )
            pred_offset += n_pred
            gt_offset += n_gt
        return per_rank_reduced

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
        preserved via the per-run reducer registry in ``_instance_reduce``.

        Goes away when core 3LC reduces variable-length embedding list
        columns server-side. The raw column it leaves behind is already tagged
        ``NUMBER_ROLE_NN_EMBEDDING`` for that future flow.
        """
        n = self._settings.instance_embeddings_dim
        pred_reduced, gt_reduced = self._compute_reduced_instance_embeddings()

        # Drop raw buffers — fit is done and we re-read raw values from the
        # source table during rewrite (where they're tiny pyarrow lists, not
        # numpy arrays).
        self._raw_pred_emb = []
        self._raw_gt_emb = []

        # Build destination schema: source columns minus the raw embedding
        # columns, plus the reduced ones.
        src_schema_values = dict(raw_table.rows_schema.values)
        src_schema_values.pop(PREDICTED_INSTANCE_EMBEDDING_RAW, None)
        src_schema_values.pop(GROUND_TRUTH_INSTANCE_EMBEDDING_RAW, None)
        src_schema_values[PREDICTED_INSTANCE_EMBEDDING] = _instance_embeddings_list_schema(
            n, display_name=f"Predicted Instance Embedding ({n}D)"
        )
        if gt_reduced is not None:
            src_schema_values[GROUND_TRUTH_INSTANCE_EMBEDDING] = _instance_embeddings_list_schema(
                n, display_name=f"Ground Truth Instance Embedding ({n}D)"
            )

        dst_writer = tlc.MetricsTableWriter(
            run_url=self._run.url,
            foreign_table_url=self.dataloader.dataset.table.url,
            schema=src_schema_values,
        )

        BATCH_SIZE = 32

        chunk: dict[str, list] = {}
        chunk_size = 0
        pred_offset = 0
        gt_offset = 0

        def flush() -> None:
            nonlocal chunk, chunk_size, pred_offset, gt_offset
            if chunk_size == 0:
                return
            chunk[PREDICTED_INSTANCE_EMBEDDING] = [
                arr.astype(np.float32).tolist() for arr in pred_reduced[pred_offset : pred_offset + chunk_size]
            ]
            pred_offset += chunk_size
            if gt_reduced is not None:
                chunk[GROUND_TRUTH_INSTANCE_EMBEDDING] = [
                    arr.astype(np.float32).tolist() for arr in gt_reduced[gt_offset : gt_offset + chunk_size]
                ]
                gt_offset += chunk_size
            dst_writer.add_batch(chunk)
            chunk = {}
            chunk_size = 0

        for sample in raw_table:
            for col, val in sample.items():
                if col in (PREDICTED_INSTANCE_EMBEDDING_RAW, GROUND_TRUTH_INSTANCE_EMBEDDING_RAW):
                    continue
                chunk.setdefault(col, []).append(val)
            chunk_size += 1
            if chunk_size >= BATCH_SIZE:
                flush()
        flush()

        dst_table = dst_writer.finalize()
        return dst_table, dst_writer.get_written_metrics_infos()

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

    def _delete_table_on_disk(self, table: tlc.Table) -> None:
        """Best-effort removal of an unregistered intermediate metrics table.

        We keep the raw streaming table around just long enough to read it
        back during ``_fit_and_rewrite``; after that it's an orphaned
        directory under the run that nothing references.
        """
        try:
            ObjectRegistry._delete_object_from_caches(table.url)
        except Exception:
            pass
        try:
            table.url.delete()
        except Exception as exc:
            LOGGER.warning(f"{TLC_COLORSTR}Failed to delete intermediate raw metrics table at {table.url}: {exc}")

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
