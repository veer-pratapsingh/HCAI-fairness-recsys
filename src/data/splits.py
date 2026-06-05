"""Phase 1.4: leave-last-out split (Decision D5).

For each session sequence `seq = [i0, i1, ..., i_{n-1}]` (n >= 3):
  - test_target  = i_{n-1}          (the final visit we try to predict)
  - test_input   = i0 ... i_{n-2}   (history available at test time)
  - val_target   = i_{n-2}
  - val_input    = i0 ... i_{n-3}
  - train_history= i0 ... i_{n-3}   (what models learn next-item transitions from)

The split is deterministic and seed-independent: only model initialisation and
batch shuffling vary across the 5 seeds, never the data split.
"""
from __future__ import annotations

import pandas as pd

from src.utils import paths


def make_splits(sequences: pd.DataFrame) -> pd.DataFrame:
    """Attach leave-last-out split columns to the sequences frame."""
    df = sequences.copy()
    seq = df["seq"]
    df["test_target"] = seq.map(lambda s: s[-1])
    df["test_input"] = seq.map(lambda s: s[:-1])
    df["val_target"] = seq.map(lambda s: s[-2])
    df["val_input"] = seq.map(lambda s: s[:-2])
    df["train_history"] = seq.map(lambda s: s[:-2])
    return df


def load_sequences() -> pd.DataFrame:
    """Load the sequences parquet produced by build_sequences.py."""
    return pd.read_parquet(paths.SEQUENCES_PARQUET)


def load_splits(write: bool = True) -> pd.DataFrame:
    """Load sequences and attach split columns, optionally caching to parquet."""
    df = make_splits(load_sequences())
    if write:
        paths.ensure_dirs()
        df.to_parquet(paths.SPLITS_PARQUET, index=False)
    return df


if __name__ == "__main__":
    out = load_splits(write=True)
    print(f"Built splits for {len(out):,} sessions -> {paths.SPLITS_PARQUET}")
