"""Shared pytest fixtures for HCAI project tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.base import Context


@pytest.fixture
def small_splits() -> pd.DataFrame:
    """Synthetic splits DataFrame with 20 sessions across 2 modules/presentations."""
    rng = np.random.RandomState(42)
    records = []
    for i in range(20):
        # 10 sessions for AAA, 10 sessions for BBB
        module = "AAA" if i < 10 else "BBB"
        presentation = "2013J"
        seq_len = rng.randint(5, 11)
        
        # Generate sequence of item indices between 1 and 20 (0 is padding)
        seq = rng.randint(1, 21, size=seq_len).tolist()
        
        records.append({
            "id_student": 1000 + i,
            "code_module": module,
            "code_presentation": presentation,
            "seq": seq,
            "seq_len": len(seq),
            "gender": rng.choice(["M", "F"]),
            "age_band": rng.choice(["0-35", "35-55"]),
            "age_binary": "0-35" if rng.random() > 0.5 else "35+",
            "disability": rng.choice(["N", "Y"]),
            "imd_band": rng.choice(["0-10%", "50-60%"]),
            "imd_binary": rng.choice(["advantaged", "disadvantaged"]),
            "imd_unknown": False,
            "test_target": seq[-1],
            "test_input": seq[:-1],
            "val_target": seq[-2],
            "val_input": seq[:-2],
            "train_history": seq[:-2],
        })
    return pd.DataFrame(records)


@pytest.fixture
def context_aaa() -> Context:
    """Mock context for module AAA, presentation 2013J."""
    return Context(code_module="AAA", code_presentation="2013J", id_student=1000)
