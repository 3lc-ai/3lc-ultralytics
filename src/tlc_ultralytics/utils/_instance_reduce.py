"""TEMP(instance-embeddings): in-process reducer for variable-length per-instance embeddings.

3LC's native reducer (``Run.reduce_embeddings_by_foreign_table_url``) operates on
fixed-size-list-per-row columns. Per-instance embeddings are variable-length
(N detections per image, N varies per row), so this module flattens across the
batch, fits/transforms with the configured reducer, and re-splits by per-image
instance counts.

Delete this file and route reduction through the native 3LC API once upstream
supports variable-length embedding list columns. All symbols here are private
(leading underscore) — do not depend on them across releases.
"""

from __future__ import annotations

import numpy as np
from ultralytics.utils import LOGGER

from tlc_ultralytics.constants import TLC_COLORSTR

# Fitted reducers shared across one run's validation passes, keyed by run URL.
# The first split to fit (train, by collect()'s split ordering) registers its
# reducer here; later splits in the same run transform into that space. Cleared
# by collect() once the run is done. Under DDP only RANK 0 fits, so only RANK 0
# populates its registry. Dies with this module when native reduction lands.
_fitted_reducers: dict[str, object] = {}


def _get_fitted_reducer(run_url: str) -> object | None:
    """Return the reducer fitted earlier in this run, if any."""
    return _fitted_reducers.get(run_url)


def _set_fitted_reducer(run_url: str, reducer: object) -> None:
    """Register a fitted reducer for reuse by later splits in the same run."""
    _fitted_reducers[run_url] = reducer


def _clear_fitted_reducer(run_url: str) -> None:
    """Drop the run's fitted reducer (no-op if none was fitted)."""
    _fitted_reducers.pop(run_url, None)


def _reduce_instance_embeddings(
    raw_embeddings_per_image: list[np.ndarray],
    method: str,
    n_components: int,
    progress_callback: object | None = None,
    **reducer_args,
) -> tuple[list[np.ndarray], object | None]:
    """Flatten all instance embeddings, reduce, map back to per-image lists.

    Args:
        raw_embeddings_per_image: list of [N_i, C] arrays
        method: 'pacmap', 'umap', or 'pca'
        n_components: target dimensionality (2 or 3)
        progress_callback: Optional callable(phase, current, total) for progress reporting.

    Returns:
        Tuple of (reduced per-image lists, fitted reducer object).
        The reducer can be passed to _transform_instance_embeddings for projecting
        additional data (e.g. ground-truth embeddings) into the same space.
    """
    counts = [emb.shape[0] for emb in raw_embeddings_per_image]
    total_instances = sum(counts)

    if total_instances == 0:
        return [np.empty((0, n_components), dtype=np.float32) for _ in raw_embeddings_per_image], None

    all_embeddings = np.concatenate([emb for emb in raw_embeddings_per_image if emb.shape[0] > 0], axis=0)

    LOGGER.info(TLC_COLORSTR + f"Reducing {total_instances} instance embeddings to {n_components}D with {method}...")

    if progress_callback:
        progress_callback("fit", 0, total_instances)

    if method == "pacmap":
        import pacmap

        # PaCMAP needs save_tree=True for the fitted reducer to support .transform()
        # on new data. We rely on that for GT embeddings and cross-split projection,
        # so default it on (but let callers override via reducer_args).
        reducer_args.setdefault("save_tree", True)
        reducer = pacmap.PaCMAP(n_components=n_components, **reducer_args)
        reduced = reducer.fit_transform(all_embeddings)
    elif method == "umap":
        import umap  # ty: ignore[unresolved-import]

        reducer = umap.UMAP(n_components=n_components, **reducer_args)
        reduced = reducer.fit_transform(all_embeddings)
    elif method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=n_components, **reducer_args)
        reduced = reducer.fit_transform(all_embeddings)
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    reduced = reduced.astype(np.float32)

    if progress_callback:
        progress_callback("fit", total_instances, total_instances)

    result = []
    idx = 0
    for count in counts:
        if count > 0:
            result.append(reduced[idx : idx + count])
            idx += count
        else:
            result.append(np.empty((0, n_components), dtype=np.float32))

    return result, reducer


_TRANSFORM_BATCH_SIZE = 5000


def _transform_instance_embeddings(
    raw_embeddings_per_image: list[np.ndarray],
    reducer: object,
    n_components: int,
    progress_callback: object | None = None,
    label: str = "instance",
) -> list[np.ndarray]:
    """Transform instance embeddings using an already-fitted reducer.

    Projects new data (e.g. ground-truth embeddings) into the same embedding
    space as the data the reducer was fitted on. Processes in batches of 5000
    for progress reporting.

    Args:
        raw_embeddings_per_image: list of [N_i, C] arrays
        reducer: A fitted PaCMAP, UMAP, or PCA reducer object
        n_components: target dimensionality (must match the reducer)
        progress_callback: Optional callable(phase, current, total) for progress reporting.
        label: Human-readable label for log messages (e.g. "predicted", "ground-truth").

    Returns:
        list of [N_i, n_components] arrays (or empty arrays for images with no instances)
    """
    counts = [emb.shape[0] for emb in raw_embeddings_per_image]
    total_instances = sum(counts)

    if total_instances == 0:
        return [np.empty((0, n_components), dtype=np.float32) for _ in raw_embeddings_per_image]

    all_embeddings = np.concatenate([emb for emb in raw_embeddings_per_image if emb.shape[0] > 0], axis=0)

    LOGGER.info(TLC_COLORSTR + f"Transforming {total_instances} {label} instance embeddings to {n_components}D...")

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
