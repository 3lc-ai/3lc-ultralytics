"""TEMP(instance-embeddings): in-process reducer for variable-length per-instance embeddings.

3LC's native reducer (``Run.reduce_embeddings_by_foreign_table_url``) operates on
fixed-size-list-per-row columns. Per-instance embeddings are variable-length
(N detections per image, N varies per row), so this module fits the configured
reducer on a bounded sample of raw instance embeddings read back from the written
metrics tables, and transforms per-image embeddings into the fitted space.

Delete this file and route reduction through the native 3LC API once upstream
supports variable-length embedding list columns. All symbols here are private
(leading underscore) — do not depend on them across releases.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from ultralytics.utils import LOGGER

from tlc_ultralytics.constants import TLC_COLORSTR

if TYPE_CHECKING:
    import tlc

# Fitted reducers shared across one run's validation passes, keyed by
# (run URL, kind) where kind is "instance" or "image". The first split to fit
# (train, by collect()'s split ordering) registers its reducer here; later
# splits in the same run transform into that space. Cleared by collect() and
# the trainer once the run is done. Under DDP only RANK 0 fits, so only RANK 0
# populates its registry. Dies with this module when native reduction lands.
_fitted_reducers: dict[tuple[str, str], object] = {}


def _get_fitted_reducer(run_url: str, kind: str) -> object | None:
    """Return the reducer of the given kind fitted earlier in this run, if any."""
    return _fitted_reducers.get((run_url, kind))


def _set_fitted_reducer(run_url: str, kind: str, reducer: object) -> None:
    """Register a fitted reducer of the given kind for reuse by later splits in the same run."""
    _fitted_reducers[(run_url, kind)] = reducer


def _clear_fitted_reducers(run_url: str) -> None:
    """Drop all of the run's fitted reducers (no-op if none was fitted)."""
    for key in [key for key in _fitted_reducers if key[0] == run_url]:
        _fitted_reducers.pop(key, None)


def _fit_embeddings_reducer(
    sample: np.ndarray,
    method: str,
    n_components: int,
    progress_callback: object | None = None,
    label: str = "instance",
    **reducer_args,
) -> object:
    """Fit the configured reducer on a matrix of sampled raw embeddings.

    Args:
        sample: [K, C] matrix of raw embeddings to fit on
        method: 'pacmap', 'umap', or 'pca'
        n_components: target dimensionality (2 or 3)
        progress_callback: Optional callable(phase, current, total) for progress reporting.
        label: Human-readable label for log messages (e.g. "instance", "image").

    Returns:
        The fitted reducer object, ready for `_transform_embeddings`.
    """
    LOGGER.info(
        TLC_COLORSTR + f"Fitting {method} {label}-embeddings reducer ({n_components}D) on {len(sample)} samples..."
    )

    if progress_callback:
        progress_callback("fit", 0, len(sample))

    if method == "pacmap":
        import pacmap

        # PaCMAP needs save_tree=True for the fitted reducer to support .transform()
        # on new data. All reduced values are produced via .transform() (the fit runs
        # on a sample), so default it on (but let callers override via reducer_args).
        reducer_args.setdefault("save_tree", True)
        reducer = pacmap.PaCMAP(n_components=n_components, **reducer_args)
        reducer.fit(sample)
    elif method == "umap":
        import umap  # ty: ignore[unresolved-import]

        reducer = umap.UMAP(n_components=n_components, **reducer_args)
        reducer.fit(sample)
    elif method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components, **reducer_args)
        reducer.fit(sample)
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    if progress_callback:
        progress_callback("fit", len(sample), len(sample))

    return reducer


def _read_raw_embedding_column(table: tlc.Table, column_name: str) -> tuple[np.ndarray | None, np.ndarray]:
    """Read a variable-length raw-embedding list column as a flat matrix plus per-row instance counts.

    Reads the column directly as pyarrow data, avoiding per-row sample-type decoding and Python-object
    materialization of the (potentially large) float lists.

    Args:
        table: the metrics table to read from (fully loaded into memory by the read; callers should evict it
            from the object caches when done to bound memory across many tables)
        column_name: name of the raw embedding column, a list of fixed-size float vectors per row

    Returns:
        Tuple of (matrix, per_row_counts) where matrix is a [total_instances, C] float32 array, or None when the
        column holds no instances, and per_row_counts has one instance count per table row.
    """
    column = table.get_column_as_pyarrow_array(column_name)
    per_row_counts = pc.list_value_length(column).to_numpy(zero_copy_only=False).astype(np.int64)

    total_instances = int(per_row_counts.sum())
    if total_instances == 0:
        return None, per_row_counts

    per_instance = column.flatten()  # one entry per instance, honoring row offsets
    if isinstance(per_instance, pa.ChunkedArray):
        per_instance = per_instance.combine_chunks()
    flat_values = per_instance.flatten().to_numpy(zero_copy_only=False)

    channels = flat_values.size // total_instances
    matrix = flat_values.astype(np.float32, copy=False).reshape(total_instances, channels)
    return matrix, per_row_counts


def _read_image_embedding_column(table: tlc.Table, column_name: str) -> np.ndarray | None:
    """Read a fixed-size per-row embedding column as a [n_rows, C] float32 matrix.

    The image-embedding counterpart to `_read_raw_embedding_column`: one fixed-size vector per row instead
    of a variable-length list of vectors. Returns None when the table has no rows.
    """
    column = table.get_column_as_pyarrow_array(column_name)
    n_rows = len(column)
    if n_rows == 0:
        return None

    flat = column.flatten()
    if isinstance(flat, pa.ChunkedArray):
        flat = flat.combine_chunks()
    flat_values = flat.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    return flat_values.reshape(n_rows, flat_values.size // n_rows)


_TRANSFORM_BATCH_SIZE = 5000


def _transform_embeddings(
    raw_embeddings_per_image: list[np.ndarray],
    reducer: object,
    n_components: int,
    progress_callback: object | None = None,
    label: str = "instance",
) -> list[np.ndarray]:
    """Transform raw embeddings using an already-fitted reducer.

    Projects new data (e.g. ground-truth or image embeddings) into the same
    embedding space as the data the reducer was fitted on. Processes in batches
    of 5000 for progress reporting.

    Args:
        raw_embeddings_per_image: list of [N_i, C] arrays
        reducer: A fitted PaCMAP, UMAP, or PCA reducer object
        n_components: target dimensionality (must match the reducer)
        progress_callback: Optional callable(phase, current, total) for progress reporting.
        label: Human-readable label for log messages (e.g. "predicted", "ground-truth", "image").

    Returns:
        list of [N_i, n_components] arrays (or empty arrays for entries with no rows)
    """
    counts = [emb.shape[0] for emb in raw_embeddings_per_image]
    total_instances = sum(counts)

    if total_instances == 0:
        return [np.empty((0, n_components), dtype=np.float32) for _ in raw_embeddings_per_image]

    all_embeddings = np.concatenate([emb for emb in raw_embeddings_per_image if emb.shape[0] > 0], axis=0)

    LOGGER.info(TLC_COLORSTR + f"Transforming {total_instances} {label} embeddings to {n_components}D...")

    if progress_callback:
        progress_callback("transform", 0, total_instances)

    if total_instances > _TRANSFORM_BATCH_SIZE:
        reduced_parts = []
        for start in range(0, total_instances, _TRANSFORM_BATCH_SIZE):
            end = min(start + _TRANSFORM_BATCH_SIZE, total_instances)
            chunk = all_embeddings[start:end]
            reduced_parts.append(reducer.transform(chunk).astype(np.float32))
            if progress_callback:
                progress_callback("transform", end, total_instances)
        reduced = np.concatenate(reduced_parts, axis=0)
    else:
        reduced = reducer.transform(all_embeddings).astype(np.float32)
        if progress_callback:
            progress_callback("transform", total_instances, total_instances)

    result = []
    idx = 0
    for count in counts:
        if count > 0:
            result.append(reduced[idx : idx + count])
            idx += count
        else:
            result.append(np.empty((0, n_components), dtype=np.float32))

    return result
