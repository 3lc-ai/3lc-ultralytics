from __future__ import annotations

import tlc
from ultralytics.utils import LOGGER

from tlc_ultralytics.constants import TLC_COLORSTR


def reduce_embeddings(
    run: tlc.Run,
    method: str,
    n_components: int,
    foreign_table_url: tlc.Url | None = None,
    reducer_args: dict | None = None,
):
    """Reduce image embeddings by a foreign table URL."""
    if foreign_table_url is None:
        foreign_table_url = tlc.Url(tlc.active_run().constants["inputs"][0]["input_table_url"]).to_absolute(
            tlc.active_run().url
        )

    LOGGER.info(
        TLC_COLORSTR + f"Reducing image embeddings to {n_components}D with {method}, this may take a few minutes..."
    )
    run.reduce_embeddings_by_foreign_table_url(
        foreign_table_url=foreign_table_url,
        method=method,
        n_components=n_components,
        **(reducer_args or {}),
    )
