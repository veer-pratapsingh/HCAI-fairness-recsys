"""Phase 1.5: sanity-check the processed data before modelling.

Prints group sizes per protected attribute, sequence-length distribution, and
vocab size. Run this right after build_sequences.py and EYEBALL the group sizes:
any tiny group will make its fairness metrics noisy, and you want to know now.

    python -m src.data.eda_summary
"""
from __future__ import annotations

import pandas as pd

from src.utils import paths

GROUP_COLS = ["gender", "age_band", "age_binary", "disability", "imd_binary"]


def main() -> None:
    seq = pd.read_parquet(paths.SEQUENCES_PARQUET)
    vocab = pd.read_parquet(paths.ITEM_VOCAB_PARQUET)

    print("=" * 60)
    print(f"Sessions : {len(seq):,}")
    print(f"Students : {seq['id_student'].nunique():,}")
    print(f"Items    : {len(vocab):,}")
    print("=" * 60)

    print("\nSequence length:")
    desc = seq["seq_len"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99])
    print(desc.to_string())

    print("\nProtected group sizes:")
    for col in GROUP_COLS:
        print(f"\n  [{col}]")
        counts = seq[col].value_counts(dropna=False)
        pct = 100 * counts / len(seq)
        for value, n in counts.items():
            flag = "  <-- SMALL" if pct[value] < 5 else ""
            print(f"    {str(value):<16} {n:>8,}  ({pct[value]:5.1f}%){flag}")

    n_unknown = int(seq["imd_unknown"].sum())
    print(f"\n  imd_band missing (unknown group): {n_unknown:,} "
          f"({100 * n_unknown / len(seq):.1f}%)")

    print("\nTop activity types in vocab:")
    print(vocab["activity_type"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
