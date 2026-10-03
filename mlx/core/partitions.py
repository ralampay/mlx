"""Seeded row partitions shared by vector training and fitted controls."""
from __future__ import annotations

import hashlib
import json

from mlx.core.exceptions import MLXUserError


def partition_rows(count, val_ratio, seed, *, generator=None):
    import torch
    if count < 2 or not 0 < val_ratio < 1:
        raise MLXUserError("Partition requires at least two rows and 0 < val-ratio < 1.")
    generator = generator if generator is not None else torch.Generator().manual_seed(seed)
    indices = torch.randperm(count, generator=generator).tolist()
    size = min(count - 1, max(1, round(count * val_ratio)))
    return indices[size:], indices[:size]


def partition_hash(training, validation):
    return hashlib.sha256(json.dumps([list(training), list(validation)]).encode()).hexdigest()
