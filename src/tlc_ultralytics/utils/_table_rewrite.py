"""TEMP(embeddings): arrow-level rewrite of a flushed raw metrics table into a reduced one.

The end-of-pass embedding reduction replaces two or three columns of a just-written metrics table and carries
everything else through unchanged. Reading the raw table row by row and feeding the rows back to a
`tlc.MetricsTableWriter` decodes and re-encodes every pass-through column on the way: for segmentation metrics the
`segmentations_predicted` column expands from RLE into dense `(H, W, N)` uint8 masks at original image resolution
only to be RLE-encoded again by the very next `add_batch` — tens of gigabytes of intermediates for columns nobody
is changing.

This module copies the pass-through columns at the arrow level instead. The raw table's row-form `pyarrow.Table` is
taken as it is on disk, the raw embedding columns are dropped from it, the reduced ones are appended as
properly-typed arrow arrays, and the result replaces the writer's own record-batch buffer so that everything else
about a metrics table (url allocation, schema resolution and serialization, the table json sidecar, row count, run
registration and the metrics infos `_post_validation` consumes) is still produced by `tlc.MetricsTableWriter`. The
private-attribute injection is needed because `TableWriter` only ingests values through `add_batch`, which always
runs the write pipeline even for values that came straight out of another 3LC table and are already in row form.

Delete this module along with the rest of the TEMP(embeddings) reduction once core 3LC reduces embedding columns
server-side. All symbols here are private (leading underscore) — do not depend on them across releases.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import tlc
from tlc._core.object_registry import ObjectRegistry
from tlc.helpers.schema_helper import SchemaHelper

if TYPE_CHECKING:
    from collections.abc import Mapping


def _fixed_size_list_array(values: np.ndarray, dim: int) -> pa.FixedSizeListArray:
    """Build a `fixed_size_list<float: dim>` array of `len(values) // dim` rows straight from *values*' buffer."""
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), dim)


def _nested_list_array(rows: list[np.ndarray], dim: int) -> pa.ListArray:
    """Build a `list<fixed_size_list<float: dim>>` array from one `(N_i, dim)` float array per row."""
    counts = [len(row) for row in rows]
    flat = np.concatenate(rows) if any(counts) else np.empty(0, dtype=np.float32)
    offsets = np.zeros(len(rows) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts)
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), _fixed_size_list_array(flat, dim))


def _filled_with_default(
    column: pa.Array | pa.ChunkedArray, column_schema: tlc.Schema | None
) -> pa.Array | pa.ChunkedArray:
    """Replace *column*'s nulls with the default value its schema declares, so that the rewritten table reads back
    the way a row-by-row copy through the sample view would have written it. Nested columns, columns without a
    declared default and defaults of a type the column cannot hold are deliberately left as nulls — which is what
    the source table stores for them anyway, and losing the already-written raw tables to a mistyped default in a
    user-supplied schema is not worth it.
    """
    if column.null_count == 0 or column_schema is None or pa.types.is_nested(column.type):
        return column

    # The same two places `tlc.TableWriter.finalize` looks when deciding which declared columns survive.
    default = column_schema.default_value
    if default is None and column_schema.value is not None:
        default = column_schema.value._default_value
    if default is None:
        return column

    try:
        return pc.fill_null(column, pa.scalar(default, type=column.type))
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
        return column


def _build_rewritten_arrow_table(
    source: pa.Table,
    schema: Mapping[str, tlc.Schema],
    reduced_columns: Mapping[str, pa.Array],
    drop_columns: set[str],
) -> pa.Table:
    """Build the reduced table's arrow data: *source* minus *drop_columns*, plus *reduced_columns* appended.

    Pass-through columns are taken from *source* by reference — no copy, no decode. This includes columns whose
    values are urls relative to their own table (image paths, `input_table_id`); they resolve identically in the
    rewritten table only because every metrics table, raw and reduced alike, is a direct child of the run url.
    """
    columns: list[pa.Array | pa.ChunkedArray] = []
    names: list[str] = []

    for name in source.column_names:
        if name in drop_columns:
            continue
        columns.append(_filled_with_default(source.column(name), schema.get(name)))
        names.append(name)

    for name, array in reduced_columns.items():
        # Type each reduced column exactly as `tlc.TableWriter` would have from the same schema.
        target_type = SchemaHelper.to_pyarrow_datatype(schema[name])
        columns.append(array if array.type == target_type else array.cast(target_type))
        names.append(name)

    return pa.table(columns, names=names)


def _write_rewritten_metrics_table(
    *,
    arrow_table: pa.Table,
    schema: Mapping[str, tlc.Schema],
    run_url: tlc.Url,
    foreign_table_url: tlc.Url,
    stream_name: str = "default_stream",
) -> list[Mapping[str, Any]]:
    """Write *arrow_table* to the run as a metrics table with the given column schemas, and return its metrics
    infos with urls relative to the run url.

    The record batches are handed to a `tlc.MetricsTableWriter` directly instead of through `add_batch`, so the
    already-row-form values are written as they are. Everything the writer does around the buffer — url
    allocation, the pruned rows schema on the table json, the row count, registering the table on the run — runs
    unchanged, so the result is indistinguishable from a table written batch by batch.

    Raises ValueError when *arrow_table* has no rows, before a url is allocated.
    """
    if arrow_table.num_rows == 0:
        msg = "Refusing to write an empty metrics table; callers must skip raw tables with no rows."
        raise ValueError(msg)

    writer = tlc.MetricsTableWriter(
        run_url=run_url,
        foreign_table_url=foreign_table_url,
        # The writer's pipeline mutates the schemas it is given; one dict is reused across tables.
        schema=copy.deepcopy(dict(schema)),
        stream_name=stream_name,
    )

    # `finalize()` reads the record-batch buffer and the row count off the writer, and nothing else that
    # `add_batch` would have set. Guard those two by name so that a rename in a future 3lc surfaces here rather
    # than as a silently zero-row table.
    missing = [name for name in ("buffer", "row_count") if not hasattr(writer, name)]
    if missing:
        msg = (
            f"tlc.MetricsTableWriter no longer has {', '.join(missing)}; the arrow-level metrics table rewrite in "
            "tlc_ultralytics.utils._table_rewrite needs updating for this version of 3lc."
        )
        raise RuntimeError(msg)

    # to_batches() slices the columns without copying their buffers.
    writer.buffer.extend(arrow_table.to_batches())
    writer.row_count = arrow_table.num_rows

    table = writer.finalize()
    metrics_infos = list(writer.get_written_metrics_infos())

    # Drop the written table from RAM the way `_RollingMetricsWriter` does for the tables it flushes.
    ObjectRegistry._delete_object_from_caches(table.url)

    return metrics_infos
