from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import tlc
from task_config import (
    INSTANCE_EMB_OVERRIDES,
    TASK2DATASET,
    TASK2LABEL_COLUMN_NAME,
    TASK2MODEL,
    TASK2PREDICTED_LABEL_COLUMN_NAME,
    skip_pacmap_on_macos,
)
from testing_helpers import (
    capture_logs,
    get_metrics_tables_from_run,
    get_run_from_settings,
)
from tlc.constants._run_status import RUN_STATUS_COMPLETED

from tlc_ultralytics import YOLO as TLCYOLO
from tlc_ultralytics import Settings
from tlc_ultralytics.constants import (
    DEFAULT_COLLECT_RUN_DESCRIPTION,
    EPOCH,
    EXAMPLE_ID,
    PER_CLASS_METRICS_STREAM_NAME,
    PREDICTED_SEGMENTATIONS,
    TRAINING_PHASE,
)


@pytest.mark.slow
@pytest.mark.parametrize("task", ["detect", "segment"])
def test_metrics_collection_only(task) -> None:
    # save_json=True would normally route detect/segment validation through COCO/LVIS JSON
    # evaluation, which reads on-disk annotation files that 3LC Tables don't have (previously
    # crashed with KeyError: 'path'). It must instead be disabled with a warning, and collection
    # must run to completion.
    overrides = {"device": "cpu", "save_json": True}
    settings = Settings(project_name=f"test_{task}_collect", run_name=f"test_{task}_collect", collect_loss=True)
    splits = ("train", "val")

    model = TLCYOLO(TASK2MODEL[task])
    with capture_logs(logging.WARNING) as log_messages:
        results_dict = model.collect(data=TASK2DATASET[task], splits=splits, settings=settings, **overrides)
    assert all(results_dict[split] for split in splits), "Metrics collection failed"

    # save_json was unsupported, so a clear warning was emitted and it was disabled for the run.
    assert any("save_json is not supported with 3LC datasets" in msg for msg in log_messages), (
        "Expected warning about save_json not being supported with 3LC datasets"
    )

    run_urls = [results_dict[split].run_url for split in splits]
    assert run_urls[0] == run_urls[1], "Expected same run URL for both splits"

    run = tlc.Run.from_url(run_urls[0])
    metrics_tables = get_metrics_tables_from_run(run)

    metrics_df = pd.concat(
        [m.to_pandas() for m in metrics_tables["default_stream"]],
        ignore_index=True,
    )
    assert "loss" not in metrics_df.columns, "Expected no loss column"
    assert run.status == RUN_STATUS_COMPLETED, "Run status not set to completed after training"
    assert run.description == DEFAULT_COLLECT_RUN_DESCRIPTION, "Description mismatch"
    assert len(metrics_tables[PER_CLASS_METRICS_STREAM_NAME]) == 2, "Expected 2 per-class metrics tables (train, val)"

    per_class_metrics_df = pd.concat(
        [m.to_pandas() for m in metrics_tables[PER_CLASS_METRICS_STREAM_NAME]],
        ignore_index=True,
    )
    assert TRAINING_PHASE not in per_class_metrics_df.columns, "Expected no training phase column"
    assert EPOCH not in per_class_metrics_df.columns, "Expected no epoch column"


@pytest.mark.slow
@skip_pacmap_on_macos
def test_embeddings_collection() -> None:
    settings = Settings(
        project_name="test_embeddings_collection_project",
        run_name="test_embeddings_collection_run",
        image_embeddings_dim=2,
    )

    overrides = {
        "batch": 8,
        "device": "cpu",
        "workers": 0,
    }

    model = TLCYOLO(TASK2MODEL["detect"])
    model.collect(data="coco128.yaml", splits=("train",), settings=settings, **overrides)

    run = get_run_from_settings(settings)
    assert len(run.metrics_tables) == 2, "Expected 2 metrics tables to be written"

    embeddings_table = next(
        (metrics_table for metrics_table in run.metrics_tables if "embeddings_pacmap" in metrics_table.columns),
        None,
    )
    assert embeddings_table is not None, "Expected a metrics table with the reduced embeddings column"

    embeddings_column_arrow = embeddings_table.get_column_as_pyarrow_array("embeddings_pacmap")
    embeddings_column_list = embeddings_column_arrow.tolist()

    assert all(len(embedding) == settings.image_embeddings_dim for embedding in embeddings_column_list), (
        "Expected embeddings to be of correct dimension"
    )


def test_collect_with_string_tables_raises() -> None:
    # Passing a string for `tables` (instead of a {split: table} mapping) should fail fast
    model = TLCYOLO(TASK2MODEL["detect"])
    with pytest.raises(TypeError, match=r"Tables must be a mapping of \{split_name: table\}"):
        model.collect(tables="some/path")


@pytest.mark.slow
@pytest.mark.parametrize("task", ["detect", "segment", "pose", "obb"])
def test_collection_alternating_empty_and_predicted_batches(task, monkeypatch) -> None:
    # The metrics writer pins each column's value type from the first value of the first batch and rejects any later
    # batch whose first value has a different type. Images without predictions get `_empty_annotation`, the rest
    # `_build_annotation`, so the two must produce the same type. Collecting one image per batch and dropping every
    # other image's predictions makes the first value of consecutive batches alternate between the two, in both
    # orders.
    from tlc.constants import RLES

    from tlc_ultralytics.engine.validator import TLCValidatorMixin

    process_predictions = TLCValidatorMixin._process_predictions
    filter_top_predictions = TLCValidatorMixin._filter_top_predictions
    num_batches = 0
    dropped = {}  # example id -> whether its predictions were dropped

    def alternating_process_predictions(self, preds, batch):
        nonlocal num_batches
        self._drop_predictions = num_batches % 2 == 0
        num_batches += 1
        dropped.update((int(example_id), self._drop_predictions) for example_id in batch["example_id"])
        return process_predictions(self, preds, batch)

    def alternating_filter_top_predictions(self, pred):
        return None if self._drop_predictions else filter_top_predictions(self, pred)

    monkeypatch.setattr(TLCValidatorMixin, "_process_predictions", alternating_process_predictions)
    monkeypatch.setattr(TLCValidatorMixin, "_filter_top_predictions", alternating_filter_top_predictions)

    settings = Settings(
        project_name=f"test_alternating_empty_batches_{task}",
        run_name=f"test_alternating_empty_batches_{task}",
        conf_thres=0.01,
    )
    model = TLCYOLO(TASK2MODEL[task])
    model.collect(data=TASK2DATASET[task], splits=("val",), settings=settings, batch=1, device="cpu", workers=0)

    assert num_batches >= 3, "Expected at least three batches, so both orders of empty and predicted batches occur"

    column = TASK2PREDICTED_LABEL_COLUMN_NAME[task].split(".")[0]
    instance_key = RLES if task == "segment" else "instances"
    tables = get_metrics_tables_from_run(get_run_from_settings(settings))["default_stream"]
    rows = [row for table in tables for row in table.table_rows]
    assert sorted(row[EXAMPLE_ID] for row in rows) == sorted(dropped), "Expected one metrics row per collected image"

    for row in rows:
        num_instances = len(row[column][instance_key])
        if dropped[row[EXAMPLE_ID]]:
            assert num_instances == 0, f"Example {row[EXAMPLE_ID]} had its predictions dropped but has instances"
        else:
            assert num_instances > 0, f"Example {row[EXAMPLE_ID]} expected predictions at conf_thres=0.01"


@pytest.mark.parametrize("task", ["detect", "pose"])
def test_prediction_index_does_not_reach_annotations(task, monkeypatch) -> None:
    # PREDICTION_INDEX is bookkeeping for the scaling step, so it must be gone from the scaled predictions the
    # task validators turn into annotations.
    import torch

    from tlc_ultralytics.detect.validator import TLCDetectionValidator
    from tlc_ultralytics.engine.validator import PREDICTION_INDEX
    from tlc_ultralytics.pose.validator import TLCPoseValidator

    validator_class = {"detect": TLCDetectionValidator, "pose": TLCPoseValidator}[task]
    pbatch = {"imgsz": [64, 64], "ori_shape": (50, 80), "ratio_pad": None}
    pred = {
        "bboxes": torch.tensor([[4.0, 4.0, 20.0, 20.0], [8.0, 8.0, 24.0, 24.0]]),
        "conf": torch.tensor([0.9, 0.1]),  # the second prediction is filtered out
        "cls": torch.zeros(2),
        "keypoints": torch.zeros(2, 1, 3),
    }

    scaled_preds = []
    monkeypatch.setattr(validator_class, "_prepare_batch", lambda self, i, batch: pbatch)
    monkeypatch.setattr(
        validator_class, "_build_annotation", lambda self, scaled, mapped_classes, h, w: scaled_preds.append(scaled)
    )

    validator = validator_class.__new__(validator_class)
    validator._settings = Settings(conf_thres=0.5)
    validator._cur_pbatches = {}
    validator._cur_filtered_preds = {}
    validator.data = {"range_to_3lc_class": {0: 0}}

    validator._process_predictions([pred], {})

    assert len(scaled_preds) == 1
    assert PREDICTION_INDEX not in scaled_preds[0], "The prediction index must not reach annotation building"
    assert len(scaled_preds[0]["conf"]) == 1, "Expected only the prediction above the confidence threshold"


@pytest.mark.slow
@pytest.mark.parametrize("task", ["detect", "segment", "pose", "obb"])
def test_gt_instance_embeddings_collection(task: str) -> None:
    """Test that both predicted and ground-truth instance embeddings are collected."""
    dim = 2
    settings = Settings(
        project_name=f"test_gt_instance_emb_{task}",
        run_name=f"test_gt_instance_emb_{task}",
        instance_embeddings_dim=dim,
        ground_truth_instance_embeddings=True,
        instance_embeddings_reducer="pca",
        label_column_name=TASK2LABEL_COLUMN_NAME[task],
    )

    model = TLCYOLO(TASK2MODEL[task])
    model.collect(data=TASK2DATASET[task], splits=("train",), settings=settings, **INSTANCE_EMB_OVERRIDES)

    run = get_run_from_settings(settings)
    metrics_tables = get_metrics_tables_from_run(run)
    default_tables = metrics_tables["default_stream"]
    assert len(default_tables) >= 1, "Expected at least one default_stream metrics table"

    metrics_df = pd.concat([m.to_pandas() for m in default_tables], ignore_index=True)

    # Both predicted and GT instance embeddings should be top-level columns
    assert "predicted_instance_embedding" in metrics_df.columns, (
        f"Expected 'predicted_instance_embedding' column for task {task}"
    )
    assert "ground_truth_instance_embedding" in metrics_df.columns, (
        f"Expected 'ground_truth_instance_embedding' column for task {task}"
    )

    # Validate predicted embeddings
    for row_embs in metrics_df["predicted_instance_embedding"]:
        assert isinstance(row_embs, (list, np.ndarray)), "Expected list of embeddings"
        for emb in row_embs:
            assert len(emb) == dim, f"Expected predicted embedding dim {dim}, got {len(emb)}"

    # Validate GT embeddings
    for row_embs in metrics_df["ground_truth_instance_embedding"]:
        assert isinstance(row_embs, (list, np.ndarray)), "Expected list of embeddings"
        for emb in row_embs:
            assert len(emb) == dim, f"Expected GT embedding dim {dim}, got {len(emb)}"

    # At least some images should have GT annotations
    gt_counts = [len(row_embs) for row_embs in metrics_df["ground_truth_instance_embedding"]]
    assert sum(gt_counts) > 0, "Expected at least some GT instance embeddings"


@pytest.mark.parametrize("reducer", ["pca", "umap", "pacmap"])
def test_instance_reducer_fit_then_transform(reducer: str) -> None:
    """Unit test: each reducer must survive a fit followed by a fresh .transform().

    Exercises ``_fit_embeddings_reducer`` and ``_transform_embeddings``
    directly on synthetic data so the test doesn't depend on a full model run or
    the size of the YOLO test dataset. This is the scenario that catches pacmap's
    ``save_tree=True`` requirement — without it the fitted reducer can't project
    instances outside the fit sample into the fitted space.
    """
    pytest.importorskip(reducer if reducer != "pca" else "sklearn")

    from tlc_ultralytics.utils._instance_reduce import (
        _fit_embeddings_reducer,
        _transform_embeddings,
    )

    rng = np.random.default_rng(0)
    sample = rng.normal(size=(200, 32)).astype(np.float32)

    try:
        # random_state is a raw constructor kwarg for all three reducers; passing it through
        # exercises that instance_embeddings_reducer_kwargs are forwarded to the constructor.
        fitted = _fit_embeddings_reducer(
            sample,
            method=reducer,
            n_components=2,
            random_state=42,
        )
    except ValueError as exc:
        # pacmap on macOS ARM currently fails during fit with a
        # broadcast/shape error from its internal KNN. Skip rather than fail —
        # the post-fit .transform() path (the save_tree=True regression guard)
        # can only be checked when fit itself works.
        pytest.skip(f"{reducer} fit failed in this environment: {exc}")

    assert fitted is not None
    # The forwarded kwarg reached the underlying reducer constructor.
    assert fitted.random_state == 42

    # Transform a disjoint batch with the fitted reducer — this crashes on
    # pacmap when save_tree=False, which is the bug the in-process reducer guards.
    new_raw = [rng.normal(size=(5, 32)).astype(np.float32) for _ in range(3)]
    projected = _transform_embeddings(new_raw, fitted, n_components=2)
    assert all(r.shape == (5, 2) for r in projected)


def test_build_rewritten_arrow_table() -> None:
    """Unit test: the rewrite drops the raw embedding columns, carries the rest through by reference and types
    the reduced columns from their schemas."""
    import pyarrow as pa

    from tlc_ultralytics.utils._table_rewrite import (
        _build_rewritten_arrow_table,
        _fixed_size_list_array,
        _nested_list_array,
    )
    from tlc_ultralytics.utils.schemas import _instance_embeddings_list_schema, _reduced_image_embeddings_schema

    source = pa.table(
        {
            "example_id": pa.array([0, 1], type=pa.int32()),
            "rles": pa.array([[b"abc"], [b"de", b"f"]]),
            "input_table_id": pa.nulls(2, type=pa.int32()),  # declared with a default, no data in the parquet
            "embeddings": pa.array([[0.0] * 4, [1.0] * 4], type=pa.list_(pa.float32(), 4)),
            "predicted_instance_embedding_raw": pa.array([[[0.0] * 4], []], type=pa.list_(pa.list_(pa.float32(), 4))),
        }
    )
    schema = {
        "example_id": tlc.schemas.ExampleIdSchema(),
        "rles": tlc.Schema(),
        "input_table_id": tlc.schemas.ForeignTableIdSchema(foreign_table_url="../table"),
        "predicted_instance_embedding": _instance_embeddings_list_schema(2),
        "embeddings_pca": _reduced_image_embeddings_schema(2, "pca"),
    }
    reduced_columns = {
        "predicted_instance_embedding": _nested_list_array(
            [np.array([[0.1, 0.2]], dtype=np.float32), np.empty((0, 2), dtype=np.float32)], 2
        ),
        "embeddings_pca": _fixed_size_list_array(np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32), 2),
    }

    rewritten = _build_rewritten_arrow_table(
        source, schema, reduced_columns, {"embeddings", "predicted_instance_embedding_raw"}
    )

    assert rewritten.column_names == [
        "example_id",
        "rles",
        "input_table_id",
        "predicted_instance_embedding",
        "embeddings_pca",
    ]
    # Pass-through columns are the source's own arrow data, not a re-encoded copy
    assert rewritten.column("rles").to_pylist() == source.column("rles").to_pylist()
    # A column with no data in the source parquet gets its schema default, as a row-by-row copy would have
    assert rewritten.column("input_table_id").to_pylist() == [0, 0]
    # The reduced columns are typed from their schemas, exactly as tlc's own writer would type them
    assert rewritten.schema.field("predicted_instance_embedding").type == pa.list_(pa.list_(pa.float32(), 2))
    assert rewritten.schema.field("embeddings_pca").type == pa.list_(pa.float32(), 2)
    assert np.allclose(rewritten.column("embeddings_pca").to_pylist(), [[0.3, 0.4], [0.5, 0.6]])
    pred_rows = rewritten.column("predicted_instance_embedding").to_pylist()
    assert [len(row) for row in pred_rows] == [1, 0]
    assert np.allclose(pred_rows[0], [[0.1, 0.2]])


def test_rewrite_default_fill_tolerates_mistyped_default() -> None:
    """Unit test: a default whose type the column cannot hold leaves the column alone rather than failing the
    rewrite, which would strand the run's raw metrics tables."""
    import pyarrow as pa

    from tlc_ultralytics.utils._table_rewrite import _filled_with_default

    column = pa.chunked_array([pa.nulls(2, type=pa.int32())])
    mistyped = tlc.schemas.Int32Schema(default_value="not an int")

    assert _filled_with_default(column, mistyped).to_pylist() == [None, None]
    assert _filled_with_default(column, tlc.schemas.Int32Schema(default_value=7)).to_pylist() == [7, 7]


def test_write_rewritten_metrics_table_refuses_empty() -> None:
    """Unit test: an empty rewrite must raise before a table url is allocated."""
    import pyarrow as pa

    from tlc_ultralytics.utils._table_rewrite import _write_rewritten_metrics_table

    run = tlc.init(project_name="test_rewrite_empty", run_name="test_rewrite_empty")
    empty = pa.table({"example_id": pa.array([], type=pa.int32())})

    with pytest.raises(ValueError, match="empty metrics table"):
        _write_rewritten_metrics_table(
            arrow_table=empty,
            schema={"example_id": tlc.schemas.ExampleIdSchema()},
            run_url=run.url,
            foreign_table_url=run.url / "dummy_table",
        )

    assert not tlc.Run.from_url(run.url).metrics, "No metrics table should have been registered on the run"


def test_rolling_metrics_writer_rolls_by_bytes() -> None:
    """Unit test: the rolling writer flushes to a new metrics table when the buffer threshold is crossed,
    and the flushed tables together hold all rows in order."""
    from tlc_ultralytics.utils._rolling_writer import _RollingMetricsWriter

    run = tlc.init(project_name="test_rolling_writer", run_name="test_rolling_writer")

    writer = _RollingMetricsWriter(
        run_url=run.url,
        foreign_table_url=run.url / "dummy_table",
        schema={"value": tlc.schemas.Float32Schema()},
        max_buffer_bytes=1,  # every batch crosses the threshold -> one table per batch
    )

    assert writer.num_flushed_tables == 0
    for i in range(3):
        writer.add_batch({"example_id": [2 * i, 2 * i + 1], "value": [0.5, 1.5]})
        assert writer.num_flushed_tables == i + 1

    table_urls, metrics_infos = writer.finalize()
    assert len(table_urls) == 3
    assert len(metrics_infos) == 3
    assert all(info["stream_name"] == "default_stream" for info in metrics_infos)

    df = pd.concat([tlc.Table.from_url(url).to_pandas() for url in table_urls], ignore_index=True)
    assert sorted(df["example_id"].tolist()) == list(range(6))

    # Each flushed table was registered on the run as it was written
    run = tlc.Run.from_url(run.url)
    registered_urls = {info["url"] for info in run.metrics}
    assert {info["url"] for info in metrics_infos} <= registered_urls

    # A finalized writer must reject further batches and repeated finalize calls
    with pytest.raises(RuntimeError):
        writer.add_batch({"example_id": [0], "value": [0.0]})
    with pytest.raises(RuntimeError):
        writer.finalize()


@pytest.mark.slow
def test_metrics_flushing_end_to_end() -> None:
    """Collect with a zero buffer threshold: metrics are flushed to multiple tables that together
    hold one row per dataset image."""
    settings = Settings(
        project_name="test_metrics_flushing",
        run_name="test_metrics_flushing",
        metrics_max_buffer_mb=0,  # flush after every batch
    )

    model = TLCYOLO(TASK2MODEL["detect"])
    model.collect(data=TASK2DATASET["detect"], splits=("train",), settings=settings, batch=2, device="cpu", workers=0)

    run = get_run_from_settings(settings)
    default_tables = get_metrics_tables_from_run(run)["default_stream"]
    assert len(default_tables) >= 2, "Expected the zero buffer threshold to flush multiple metrics tables"

    df = pd.concat([t.to_pandas() for t in default_tables], ignore_index=True)
    assert sorted(df["example_id"].tolist()) == [0, 1, 2, 3], "Flushed tables should cover every image exactly once"


@pytest.mark.slow
def test_metrics_flushing_with_instance_embeddings() -> None:
    """Collect with a zero buffer threshold and instance embeddings: every raw table is rewritten with a
    reduced embedding column, raw columns and tables are gone, and rows are preserved."""
    dim = 2
    settings = Settings(
        project_name="test_metrics_flushing_instance_emb",
        run_name="test_metrics_flushing_instance_emb",
        metrics_max_buffer_mb=0,  # flush after every batch
        instance_embeddings_dim=dim,
        instance_embeddings_reducer="pca",
        # A user-supplied n_components must be ignored in favor of instance_embeddings_dim
        instance_embeddings_reducer_kwargs={"n_components": 7},
        instance_embeddings_fit_sample_size=5,  # force the sampled-fit path
        label_column_name=TASK2LABEL_COLUMN_NAME["detect"],
    )

    model = TLCYOLO(TASK2MODEL["detect"])
    model.collect(data=TASK2DATASET["detect"], splits=("train",), settings=settings, batch=2, device="cpu", workers=0)

    run = get_run_from_settings(settings)
    default_tables = get_metrics_tables_from_run(run)["default_stream"]
    assert len(default_tables) >= 2, "Expected the zero buffer threshold to flush multiple metrics tables"

    df = pd.concat([t.to_pandas() for t in default_tables], ignore_index=True)
    assert sorted(df["example_id"].tolist()) == [0, 1, 2, 3], "Rewritten tables should cover every image exactly once"

    assert "predicted_instance_embedding" in df.columns, "Expected the reduced embedding column in every table"
    assert "predicted_instance_embedding_raw" not in df.columns, "Raw embedding columns should have been rewritten"

    total_instances = 0
    for row_embs in df["predicted_instance_embedding"]:
        for emb in row_embs:
            assert len(emb) == dim
            total_instances += 1
    assert total_instances > 0, "Expected at least some reduced instance embeddings"


@pytest.mark.slow
def test_metrics_flushing_with_image_embeddings() -> None:
    """Collect with a zero buffer threshold and image embeddings: the reducer is fitted on a sample drawn
    across the flushed tables and every table is rewritten with the reduced column in place of the raw one."""
    dim = 2
    settings = Settings(
        project_name="test_metrics_flushing_image_emb",
        run_name="test_metrics_flushing_image_emb",
        metrics_max_buffer_mb=0,  # flush after every batch
        image_embeddings_dim=dim,
        image_embeddings_reducer="pca",
        # A user-supplied n_components must be ignored in favor of image_embeddings_dim
        image_embeddings_reducer_args={"n_components": 7},
        image_embeddings_fit_sample_size=3,  # force the sampled-fit path (fewer than the 4 images)
    )

    model = TLCYOLO(TASK2MODEL["detect"])
    model.collect(data=TASK2DATASET["detect"], splits=("train",), settings=settings, batch=2, device="cpu", workers=0)

    run = get_run_from_settings(settings)
    default_tables = get_metrics_tables_from_run(run)["default_stream"]
    assert len(default_tables) >= 2, "Expected the zero buffer threshold to flush multiple metrics tables"

    df = pd.concat([t.to_pandas() for t in default_tables], ignore_index=True)
    assert sorted(df["example_id"].tolist()) == [0, 1, 2, 3], "Rewritten tables should cover every image exactly once"

    assert "embeddings_pca" in df.columns, "Expected the reduced image-embeddings column"
    assert "embeddings" not in df.columns, "The raw image-embeddings column should have been rewritten"
    for emb in df["embeddings_pca"]:
        assert len(emb) == dim


@pytest.mark.slow
def test_metrics_flushing_segment_all_embeddings() -> None:
    """The bug-report scenario: segmentation masks plus image, predicted and ground-truth instance embeddings,
    with a zero buffer threshold. Every flushed table is rewritten with all three reduced columns while the
    heavy mask column is carried through the rewrite."""
    dim = 2
    settings = Settings(
        project_name="test_metrics_flushing_seg_all",
        run_name="test_metrics_flushing_seg_all",
        metrics_max_buffer_mb=0,  # flush after every batch
        image_embeddings_dim=dim,
        image_embeddings_reducer="pca",
        instance_embeddings_dim=dim,
        instance_embeddings_reducer="pca",
        ground_truth_instance_embeddings=True,
        label_column_name=TASK2LABEL_COLUMN_NAME["segment"],
    )

    model = TLCYOLO(TASK2MODEL["segment"])
    model.collect(data=TASK2DATASET["segment"], splits=("train",), settings=settings, batch=2, device="cpu", workers=0)

    run = get_run_from_settings(settings)
    default_tables = get_metrics_tables_from_run(run)["default_stream"]
    assert len(default_tables) >= 2, "Expected the zero buffer threshold to flush multiple metrics tables"

    df = pd.concat([t.to_pandas() for t in default_tables], ignore_index=True)
    assert sorted(df["example_id"].tolist()) == [0, 1, 2, 3], "Rewritten tables should cover every image exactly once"

    # The heavy mask column survived the rewrite alongside all three reduced embedding columns
    assert "segmentations_predicted" in df.columns, "Expected the predicted segmentations column"
    for column in ("embeddings_pca", "predicted_instance_embedding", "ground_truth_instance_embedding"):
        assert column in df.columns, f"Expected reduced column '{column}'"
    for raw_column in ("embeddings", "predicted_instance_embedding_raw", "ground_truth_instance_embedding_raw"):
        assert raw_column not in df.columns, f"Raw column '{raw_column}' should have been rewritten"

    for emb in df["embeddings_pca"]:
        assert len(emb) == dim
    for row_embs in df["predicted_instance_embedding"]:
        for emb in row_embs:
            assert len(emb) == dim


def _predicted_mask_facts(tables: list[tlc.Table]) -> dict[int, tuple[int, int, int]]:
    """Per example id, the (instance count, image height, image width) of `segmentations_predicted`.

    Read off the tables' arrow data, so the RLEs are never decoded into masks - which is the whole point of the
    rewrite this is used to check.
    """
    import pyarrow.compute as pc

    facts: dict[int, tuple[int, int, int]] = {}
    for table in tables:
        arrow_table = table._to_pyarrow_table()
        masks = arrow_table.column(PREDICTED_SEGMENTATIONS).combine_chunks()
        counts = pc.fill_null(pc.list_value_length(masks.field("rles")), 0).to_pylist()
        heights = masks.field("image_height").to_pylist()
        widths = masks.field("image_width").to_pylist()
        for example_id, count, height, width in zip(
            arrow_table.column(EXAMPLE_ID).to_pylist(), counts, heights, widths, strict=True
        ):
            facts[example_id] = (count, height, width)
    return facts


@pytest.mark.slow
def test_reduce_and_rewrite_all_empty_keeps_raw_tables() -> None:
    """When the rewrite produces nothing, the raw metrics tables must stay registered on the run and on disk.

    `_reduce_and_rewrite_raw_tables` returning None is the signal for that; returning an empty list of metrics
    infos instead would make the caller deregister and delete the run's only metrics tables.
    """
    from tlc_ultralytics.engine.validator import TLCValidatorMixin

    settings = Settings(
        project_name="test_rewrite_all_empty",
        run_name="test_rewrite_all_empty",
        image_embeddings_dim=2,
        image_embeddings_reducer="pca",
    )

    model = TLCYOLO(TASK2MODEL["detect"])
    # Every raw table looks empty to the rewrite, so no reduced table is written for any of them.
    with patch.object(TLCValidatorMixin, "_rewrite_raw_table", lambda *args, **kwargs: []):
        model.collect(
            data=TASK2DATASET["detect"], splits=("train",), settings=settings, batch=2, device="cpu", workers=0
        )

    run = get_run_from_settings(settings)
    default_tables = get_metrics_tables_from_run(run)["default_stream"]
    assert default_tables, "The raw metrics tables should still be registered on the run"
    for table in default_tables:
        assert table.url.exists(), f"The raw metrics table at {table.url} should not have been deleted"

    df = pd.concat([t.to_pandas() for t in default_tables], ignore_index=True)
    assert sorted(df[EXAMPLE_ID].tolist()) == [0, 1, 2, 3]
    assert "embeddings" in df.columns, "The raw image-embeddings column should have been kept"
    assert "embeddings_pca" not in df.columns, "No reduced column should have been written"


@pytest.mark.slow
def test_metrics_rewrite_does_not_decode_masks() -> None:
    """The embedding rewrite must carry the RLE mask column through without decoding it.

    Decoding an RLE row (`SegmentationHelper.masks_from_rles`) inflates it to a dense (H, W, N) uint8 array at
    original image resolution, and re-encoding it (`rles_from_masks`) throws that away again - for a column the
    rewrite is only copying. Counting both while the rewrite runs locks the arrow-level passthrough in place.
    """
    from tlc.helpers.segmentation_helper import SegmentationHelper

    from tlc_ultralytics.engine.validator import TLCValidatorMixin

    calls = {"decode": 0, "encode": 0, "rewrites": 0}
    collected: dict[int, tuple[int, int, int]] = {}
    original_masks_from_rles = SegmentationHelper.masks_from_rles
    original_rles_from_masks = SegmentationHelper.rles_from_masks
    original_reduce = TLCValidatorMixin._reduce_and_rewrite_raw_tables
    rewriting = False

    def counting_masks_from_rles(*args, **kwargs):
        if rewriting:
            calls["decode"] += 1
        return original_masks_from_rles(*args, **kwargs)

    def counting_rles_from_masks(*args, **kwargs):
        if rewriting:
            calls["encode"] += 1
        return original_rles_from_masks(*args, **kwargs)

    def counting_reduce(self, raw_table_urls, *args, **kwargs):
        # Only mask work done by the rewrite itself counts; the pass that produced the raw tables encodes masks
        # legitimately, and reading the tables back afterwards decodes them again.
        nonlocal rewriting
        calls["rewrites"] += 1
        collected.update(_predicted_mask_facts([tlc.Table.from_url(url) for url in raw_table_urls]))
        rewriting = True
        try:
            return original_reduce(self, raw_table_urls, *args, **kwargs)
        finally:
            rewriting = False

    settings = Settings(
        project_name="test_metrics_rewrite_no_decode",
        run_name="test_metrics_rewrite_no_decode",
        metrics_max_buffer_mb=0,  # flush after every batch, so several tables are rewritten
        image_embeddings_dim=2,
        image_embeddings_reducer="pca",
        instance_embeddings_dim=2,
        instance_embeddings_reducer="pca",
        ground_truth_instance_embeddings=True,
        label_column_name=TASK2LABEL_COLUMN_NAME["segment"],
    )

    model = TLCYOLO(TASK2MODEL["segment"])
    with (
        patch.object(SegmentationHelper, "masks_from_rles", staticmethod(counting_masks_from_rles)),
        patch.object(SegmentationHelper, "rles_from_masks", staticmethod(counting_rles_from_masks)),
        patch.object(TLCValidatorMixin, "_reduce_and_rewrite_raw_tables", counting_reduce),
    ):
        model.collect(
            data=TASK2DATASET["segment"],
            splits=("train",),
            settings=settings,
            batch=2,
            device="cpu",
            workers=0,
        )

    assert calls["rewrites"] == 1, "Expected the rewrite to run once for the single collected split"
    assert calls["decode"] == 0, "The rewrite decoded RLE masks instead of copying the column through"
    assert calls["encode"] == 0, "The rewrite re-encoded masks instead of copying the column through"

    # ... and the masks came through unchanged: same instance count and same (H, W) per image
    run = get_run_from_settings(settings)
    default_tables = get_metrics_tables_from_run(run)["default_stream"]
    rewritten = _predicted_mask_facts(default_tables)
    assert sorted(rewritten) == [0, 1, 2, 3], "Rewritten tables should cover every image exactly once"
    assert rewritten == collected, "The rewritten mask column differs from what was collected"
    assert any(count > 0 for count, _, _ in rewritten.values()), "Expected at least one predicted mask to compare"


@pytest.mark.slow
def test_instance_embeddings_cross_split_shared_space() -> None:
    """Multi-split collect: train fits the reducer, val is transformed into the same space."""
    dim = 2
    settings = Settings(
        project_name="test_instance_emb_cross_split",
        run_name="test_instance_emb_cross_split",
        instance_embeddings_dim=dim,
        instance_embeddings_reducer="pca",
        label_column_name=TASK2LABEL_COLUMN_NAME["detect"],
    )

    model = TLCYOLO(TASK2MODEL["detect"])
    model.collect(data=TASK2DATASET["detect"], splits=("train", "val"), settings=settings, **INSTANCE_EMB_OVERRIDES)

    run = get_run_from_settings(settings)
    metrics_tables = get_metrics_tables_from_run(run)
    default_tables = metrics_tables["default_stream"]
    assert len(default_tables) >= 2, "Expected one metrics table per split"

    for table in default_tables:
        df = table.to_pandas()
        assert "predicted_instance_embedding" in df.columns
        for row_embs in df["predicted_instance_embedding"]:
            for emb in row_embs:
                assert len(emb) == dim

    # The run's reducers must not leak past collect()
    from tlc_ultralytics.utils._instance_reduce import _get_fitted_reducer

    assert _get_fitted_reducer(run.url.to_str(), "instance") is None
    assert _get_fitted_reducer(run.url.to_str(), "image") is None


@pytest.mark.slow
def test_instance_embeddings_explicit_layer() -> None:
    """Collect instance embeddings from an explicit neck layer instead of the cls-head default.

    Exercises the instance_embeddings_layer path (_add_feature_map_hook / _infer_layer_channels),
    which the default cls-head tests do not cover.
    """
    from tlc_ultralytics.utils.embeddings import _auto_detect_p3_layer

    dim = 2
    model = TLCYOLO(TASK2MODEL["detect"])
    # Pick a valid neck layer the same way the auto-detect default does, so the index isn't
    # hardcoded against a specific model architecture.
    layer_index = _auto_detect_p3_layer(model.model.model)

    settings = Settings(
        project_name="test_instance_emb_explicit_layer",
        run_name="test_instance_emb_explicit_layer",
        instance_embeddings_dim=dim,
        instance_embeddings_layer=layer_index,
        instance_embeddings_reducer="pca",
        label_column_name=TASK2LABEL_COLUMN_NAME["detect"],
    )

    model.collect(data=TASK2DATASET["detect"], splits=("train",), settings=settings, **INSTANCE_EMB_OVERRIDES)

    run = get_run_from_settings(settings)
    metrics_df = pd.concat(
        [m.to_pandas() for m in get_metrics_tables_from_run(run)["default_stream"]], ignore_index=True
    )
    assert "predicted_instance_embedding" in metrics_df.columns
    for row_embs in metrics_df["predicted_instance_embedding"]:
        for emb in row_embs:
            assert len(emb) == dim


@pytest.mark.slow
def test_instance_embeddings_warns_without_predictions() -> None:
    """With no predictions passing the confidence threshold, a clear warning is logged and the
    embedding columns are empty rather than raising."""
    settings = Settings(
        project_name="test_instance_emb_no_preds",
        run_name="test_instance_emb_no_preds",
        instance_embeddings_dim=2,
        instance_embeddings_reducer="pca",
        conf_thres=1.0,  # confidences are strictly < 1.0, so nothing passes the filter
        label_column_name=TASK2LABEL_COLUMN_NAME["detect"],
    )
    model = TLCYOLO(TASK2MODEL["detect"])

    with patch("tlc_ultralytics.engine.validator.LOGGER") as mock_logger:
        model.collect(data=TASK2DATASET["detect"], splits=("train",), settings=settings, **INSTANCE_EMB_OVERRIDES)

    warnings = [str(call.args[0]) for call in mock_logger.warning.call_args_list if call.args]
    assert any("No predicted instances were available" in w for w in warnings), warnings
