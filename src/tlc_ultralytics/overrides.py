from typing import Optional

import ultralytics
from torch.utils.data import Sampler
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.data.build import build_dataloader as build_dataloader_ultralytics


def build_dataloader(*args, **kwargs):
    sampler: Optional[Sampler] = kwargs.pop("sampler", None)

    class _InfiniteDataLoaderWithSampler(InfiniteDataLoader):
        """This class is a temporary patch to the InfiniteDataLoader that allows for a custom sampler to be set.

        It is necessary because the `build_dataloader` function in Ultralytics does not allow for a custom sampler to be
        set.
        """

        def __init__(self, *args, **kwargs):
            nonlocal sampler

            provided_sampler = kwargs.pop("sampler", None)
            if provided_sampler and sampler:
                msg = "Cannot patch InfiniteDataLoader when a sampler is already set by Ultralytics. "
                msg += "The 3LC integration does not yet support distributed training with custom weights."
                raise ValueError(msg)

            sampler = provided_sampler or sampler

            shuffle = kwargs.pop("shuffle", True) and sampler is None

            super().__init__(*args, sampler=sampler, shuffle=shuffle, **kwargs)

    # Temporarily patch the InfiniteDataLoader with one that sets the sampler
    ultralytics.data.build.InfiniteDataLoader = _InfiniteDataLoaderWithSampler

    dataloader = build_dataloader_ultralytics(*args, **kwargs)

    # Restore the original InfiniteDataLoader
    ultralytics.data.build.InfiniteDataLoader = InfiniteDataLoader

    return dataloader
