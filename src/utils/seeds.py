"""Single source of truth for reproducibility. Every run takes a seed.

The slides require 5 seeds for all experiments; call `set_seed(s)` once at the
start of each run with each of the five seeds.
"""
from __future__ import annotations

import os
import random

import numpy as np

# The five canonical seeds used across all experiments.
SEEDS = (0, 1, 2, 3, 4)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if available) PyTorch deterministically."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Favour determinism over the last few % of throughput.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch not installed yet (e.g. during the data-pipeline phase) — fine.
        pass
