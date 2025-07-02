from __future__ import annotations

from typing import Literal

import tlc
import torch

from tlc_ultralytics.settings import Settings


def create_sampler(
    table: tlc.Table,
    mode: Literal["train", "val"],
    settings: Settings,
    distributed: bool = False,
) -> torch.utils.data.Sampler | None:
    """Get the sampler for the dataset.

    :param table: The table to get the sampler for.
    :param mode: The mode of the sampler.
    :param settings: The settings for the run.
    :param distributed: Whether training is distributed.
    :returns: The sampler for the dataset.
    """
    sampler = None

    if mode == "train":
        if settings.sampling_weights or settings.exclude_zero_weight_training:
            if distributed:
                raise NotImplementedError("Distributed training and using 3LC weights is not yet supported.")

            # No need to exclude zero weights if there is no weights column
            exclude_zero_weights = False if table.weights_column_name is None else settings.exclude_zero_weight_training

            try:
                sampler = table.create_sampler(
                    exclude_zero_weights=exclude_zero_weights,
                    weighted=settings.sampling_weights,
                    shuffle=True,
                )
            except Exception as e:
                raise ValueError(f"Error creating sampler for table {table.url}") from e

    elif mode == "val":
        if distributed:
            raise NotImplementedError("Distributed validation and exclusion by weight is not yet supported.")

        # Exclude zero weight is handled in the dataset for validation
        return None
    return sampler
