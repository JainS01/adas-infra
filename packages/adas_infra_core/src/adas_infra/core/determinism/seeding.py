"""Global seeding for reproducible training runs."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic_cudnn: bool = True) -> None:
    """Seed Python random, NumPy, and PyTorch (CPU + CUDA) with *seed*.

    Sets PYTHONHASHSEED for subprocess reproducibility.
    CuDNN deterministic mode trades performance for bit-exact results.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker initialiser — unique seed per worker to avoid correlation."""
    worker_seed = (
        int(np.random.get_state()[1][0]) + worker_id  # type: ignore[index]
    ) % (2**31)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
