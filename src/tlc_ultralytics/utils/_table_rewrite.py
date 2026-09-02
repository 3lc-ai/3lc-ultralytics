"""TEMP(embeddings): arrow-level rewrite of a flushed raw metrics table into a reduced one.

The end-of-pass embedding reduction replaces two or three columns of a just-written metrics table and carries
everything else through unchanged. Reading the raw table row by row and feeding the rows back to a
`tlc.MetricsTableWriter` decodes and re-encodes every pass-through column on the way: for segmentation metrics the
`predicted_segmentations` column expands from RLE into dense `(H, W, N)` uint8 masks at original image resolution
(`SegmentationMasks.from_row` -> `pycocotools.decode`) only to be RLE-encoded again by the very next `add_batch` —
tens of gigabytes of intermediates for columns nobody is changing.

This module copies the pass-through columns at the arrow level instead. The raw table's row-form `pyarrow.Table` is
taken as it is on disk, the raw embedding columns are dropped from it, the reduced ones are appended as
properly-typed arrow arrays, and the result replaces the writer's own record-batch buffer so that everything else
about a metrics table (url allocation, schema resolution and serialization, the table json sidecar, row count, run
registration and the metrics infos `_post_validation` consumes) is still produced by `tlc.MetricsTableWriter`.

tlc-core gap: `TableWriter` only ingests python/columnar batches through `add_batch`, which always runs the write
pipeline (`Schema.to_row` per leaf, externalization, relativization) even for values that came straight out of
another 3LC table and are already in row form. A public "write these record batches / this arrow table as a table
with this schema" entry point would remove the private-attribute injection in `_write_rewritten_metrics_table`.

Delete this module along with the rest of the TEMP(embeddings) reduction once core 3LC reduces embedding columns
server-side. All symbols here are private (leading underscore) — do not depend on them across releases.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc
import tlc
from tlc.helpers.schema_helper import SchemaHelper

if TYPE_CHECKING:
    from collections.abc import Mapping


def _raw_table_arrow(table: tlc.Table) -> pa.Table:
    """Return one metrics table's rows as a row-form `pyarrow.Table`, without any sample-type decoding.

    This is the table exactly as it sits in its parquet file (RLE mask blobs stay blobs, embeddings stay float
    lists), with the columns ordered and typed by the table's `rows_schema` — the same order and values a row
    iteration over the table would hand to a writer, minus the decode/re-encode round trip.

    tlc-core gap: `Table._to_pyarrow_table` is private. `Table.get_column_as_pyarrow_array` is the public columnar
    accessor, but it re-reads the whole parquet file per column, so building a full table from it would read the
    file once per column.
    """
    return table._to_pyarrow_table()


def _default_value(column_schema: tlc.Schema) -> Any:
    """The value *column_schema* supplies for rows that have none, or None when it declares none.

    Looks in the same two places `tlc.TableWriter.finalize` does when deciding which declared columns survive into
    a written table's schema without data behind them.
    """
    if column_schema.default_value is not None:
        return column_schema.default_value
    return column_schema.value._default_value if column_schema.value is not None else None


def _filled_with_default(
    column: pa.Array | pa.ChunkedArray, column_schema: tlc.Schema | None
) -> pa.Array | pa.ChunkedArray:
    """Replace *column*'s nulls with its schema's default value, if it has both nulls and a default.

    Only plain (non-nested) columns are filled — a default value for a list or struct column is not something the
    metrics writer produces, and such a column's nulls are left as they are.
    """
    if column.null_count == 0 or column_schema is None or pa.types.is_nested(column.type):
        return column

    default = _default_value(column_schema)
    if default is None:
        return column

    return pc.fill_null(column, pa.scalar(default, type=column.type))


def _build_rewritten_arrow_table(
    source: pa.Table,
    schema: Mapping[str, tlc.Schema],
    reduced_columns: Mapping[str, list],
    drop_columns: set[str],
) -> pa.Table:
    """Build the reduced table's arrow data: *source* minus *drop_columns*, plus *reduced_columns*.

    Pass-through columns are taken from *source* by reference — no copy, no decode. Each reduced column is built
    with the arrow type its entry in *schema* implies, which is the same type `tlc.TableWriter` would have derived
    for it, so the written parquet is typed identically either way. Column order matches a row-by-row copy: the
    source's order with the dropped columns removed, then the reduced columns appended.

    Columns the source's parquet has no data for at all (`input_table_id`, for one) surface in its row view as
    all-null, where a row-by-row copy would have seen the schema's default value instead. Those are filled with
    the default so the reduced table holds what a row-by-row copy would have written, values and column order
    both.
    """
    columns: list[pa.Array | pa.ChunkedArray] = []
    names: list[str] = []

    for name in source.column_names:
        if name in drop_columns:
            continue
        columns.append(_filled_with_default(source.column(name), schema.get(name)))
        names.append(name)

    for name, values in reduced_columns.items():
        columns.append(pa.array(values, type=SchemaHelper.to_pyarrow_datatype(schema[name])))
        names.append(name)

    return pa.table(columns, names=names)


def _write_rewritten_metrics_table(
    *,
    arrow_table: pa.Table,
    schema: Mapping[str, tlc.Schema],
    run_url: tlc.Url,
    foreign_table_url: tlc.Url,
    stream_name: str = "default_stream",
) -> tuple[tlc.Url, list[Mapping[str, Any]]]:
    """Write *arrow_table* to the run as a metrics table with the given column schemas.

    The record batches are handed to a `tlc.MetricsTableWriter` directly instead of through `add_batch`, so the
    already-row-form values are written as they are. Everything the writer does around the buffer — url
    allocation, the pruned rows schema on the table json, the row count, registering the table on the run — runs
    unchanged, so the result is indistinguishable from a table written batch by batch.

    Returns the written table's url and its metrics infos, urls relative to the run url.
    """
    writer = tlc.MetricsTableWriter(
        run_url=run_url,
        foreign_table_url=foreign_table_url,
        # Deep-copied for the same reason `_RollingMetricsWriter` does it: the writer's pipeline mutates the
        # schemas it is given, and the caller reuses one schema dict across every rewritten table.
        schema=copy.deepcopy(dict(schema)),
        stream_name=stream_name,
    )

    # Fail loudly rather than write a table with the wrong row count if a tlc release renames the writer state
    # the injection below stands in for.
    missing = [
        name
        for name in ("buffer", "row_count", "_pyarrow_schema", "_pyarrow_schema_ready")
        if not hasattr(writer, name)
    ]
    if missing:
        msg = (
            f"tlc.MetricsTableWriter no longer has {', '.join(missing)}; the arrow-level metrics table rewrite in "
            "tlc_ultralytics.utils._table_rewrite needs updating for this version of 3lc."
        )
        raise RuntimeError(msg)

    # Stand in for what add_batch would have set up: the pyarrow schema it derives from the resolved 3LC schema,
    # the column signature it validates later batches against, and the buffer of record batches finalize() turns
    # into the parquet file. to_batches() slices the columns without copying their buffers.
    writer._pyarrow_schema = arrow_table.schema
    writer._pyarrow_schema_ready = True
    writer._column_signature = set(arrow_table.column_names)
    writer.buffer.extend(arrow_table.to_batches())
    writer.row_count = arrow_table.num_rows

    table = writer.finalize()
    return table.url, list(writer.get_written_metrics_infos())
