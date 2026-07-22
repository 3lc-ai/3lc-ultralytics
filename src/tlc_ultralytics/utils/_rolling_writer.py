"""Byte-bounded rolling wrapper around `tlc.MetricsTableWriter`.

`tlc.MetricsTableWriter` buffers every added batch in RAM and only writes them out at `finalize()`, so a single
writer's peak host memory grows with the total size of the pass — unbounded for large datasets with heavy per-row
metrics such as RLE segmentation masks or raw instance embeddings. `_RollingMetricsWriter` bounds that growth by
tracking the byte size of the buffered record batches and finalizing + replacing the inner writer whenever the buffer
crosses a threshold. The resulting tables share a column signature and stream name, so the 3LC Dashboard joins them
back together, and the `example_id` column identifies the rows.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import tlc
from tlc._core.object_registry import ObjectRegistry

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping


class _RollingMetricsWriter:
    """Write metrics batches to a run, rolling to a new metrics table when buffered bytes cross a threshold."""

    def __init__(
        self,
        *,
        run_url: tlc.Url,
        foreign_table_url: tlc.Url,
        schema: dict[str, tlc.Schema],
        max_buffer_bytes: int,
        stream_name: str = "default_stream",
    ) -> None:
        """Initialize a rolling metrics writer.

        :param run_url: The url of the run to write metrics tables for.
        :param foreign_table_url: The url of the table the metrics rows refer to.
        :param schema: Column schemas, deep-copied for each inner writer so writers never share schema objects.
        :param max_buffer_bytes: Flush the buffered rows to a table and start a new one when the buffered pyarrow
            record batches exceed this many bytes. 0 flushes after every batch.
        :param stream_name: The metrics stream name shared by all written tables.
        """
        self._run_url = run_url
        self._foreign_table_url = foreign_table_url
        self._schema = schema
        self._max_buffer_bytes = max_buffer_bytes
        self._stream_name = stream_name

        self._writer: tlc.MetricsTableWriter | None = None
        self._buffered_bytes = 0
        self._flushed_urls: list[tlc.Url] = []
        self._flushed_infos: list[Mapping[str, Any]] = []
        self._finalized = False

    @property
    def num_flushed_tables(self) -> int:
        """The number of tables flushed so far; also the index of the table the next batch is written to."""
        return len(self._flushed_urls)

    def add_batch(self, batch: MutableMapping[str, Any]) -> None:
        """Add a batch of metrics rows, rolling to a new table if the buffer threshold is crossed."""
        if self._finalized:
            raise RuntimeError("Cannot add batches to a finalized _RollingMetricsWriter.")

        if self._writer is None:
            self._writer = self._new_writer()

        buffer_length_before = len(self._writer.buffer)
        self._writer.add_batch(batch)
        self._buffered_bytes += sum(rb.nbytes for rb in self._writer.buffer[buffer_length_before:])

        if self._buffered_bytes >= self._max_buffer_bytes:
            self._roll()

    def finalize(self) -> tuple[list[tlc.Url], list[Mapping[str, Any]]]:
        """Flush any remaining buffered rows and return (table urls, metrics infos) for all written tables.

        The metrics infos have urls relative to the run url, as returned by
        `MetricsTableWriter.get_written_metrics_infos`.
        """
        if self._finalized:
            raise RuntimeError("finalize() has already been called on this _RollingMetricsWriter.")
        self._finalized = True

        if self._writer is not None:
            if self._writer.row_count > 0:
                self._roll()
            else:
                self._writer.clear()
                self._writer = None

        return self._flushed_urls, self._flushed_infos

    def _new_writer(self) -> tlc.MetricsTableWriter:
        # Deep-copy the column schemas so schema mutations made by one writer's pipeline (defaults merging,
        # schema resolution) cannot leak into the next writer's.
        return tlc.MetricsTableWriter(
            run_url=self._run_url,
            foreign_table_url=self._foreign_table_url,
            schema=copy.deepcopy(self._schema),
            stream_name=self._stream_name,
        )

    def _roll(self) -> None:
        """Finalize the current inner writer and arrange for a fresh one on the next batch."""
        assert self._writer is not None
        table = self._writer.finalize()
        self._flushed_infos.extend(self._writer.get_written_metrics_infos())
        self._flushed_urls.append(table.url)

        # Drop the table object and evict it from the object caches so flushed data doesn't linger in RAM.
        ObjectRegistry._delete_object_from_caches(table.url)
        self._writer = None
        self._buffered_bytes = 0
