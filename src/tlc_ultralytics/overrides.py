"""This file contains overrides for classes and methods in ultralytics where patching is required.

This file is therefore under AGPL-3.0 License by Ultralytics.

"""

import os
import torch

from torch.utils.data import distributed

from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.utils import PIN_MEMORY
from ultralytics.utils import RANK


def build_dataloader(dataset, batch, workers, shuffle=True, rank=-1, sampler=None):
    """
    Create and return an InfiniteDataLoader or DataLoader for training or validation.

    Args:
        dataset (Dataset): Dataset to load data from.
        batch (int): Batch size for the dataloader.
        workers (int): Number of worker threads for loading data.
        shuffle (bool): Whether to shuffle the dataset.
        rank (int): Process rank in distributed training. -1 for single-GPU training.
        sampler (Sampler, optional): Sampler for the dataset.

    Returns:
        (InfiniteDataLoader): A dataloader that can be used for training or validation.
    """
    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min(os.cpu_count() // max(nd, 1), workers)  # number of workers
    if sampler is None:
        sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )
