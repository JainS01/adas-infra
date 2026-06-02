"""Learning-rate schedulers."""

from __future__ import annotations

import torch.optim as optim


def build_scheduler(
    optimizer: optim.Optimizer,
    max_steps: int,
    warmup_steps: int = 10,
) -> optim.lr_scheduler.LRScheduler:
    """Cosine annealing with a short linear warmup.

    Simple but effective for the judge path.  For production runs with a
    large step budget, replace with OneCycleLR or a custom cosine-restarts.
    """
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, max_steps - warmup_steps)
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )
