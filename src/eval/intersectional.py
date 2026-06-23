"""Intersectional fairness analysis.

Computes Recall@K, NDCG@K and MRR for intersection groups (e.g.,
disability=Y AND imd=disadvantaged) and reports the max–min gap between
the best and worst intersection groups within each attribute pair.

This goes beyond the single-attribute per-group accuracy in ``fairness.py``
by revealing disparities that only surface at the intersection of two
protected characteristics — a core HCAI requirement.
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd

# The four protected-attribute columns used throughout the project.
_ATTR_COLUMNS = ["gender", "imd_binary", "disability", "age_band"]


def intersectional_analysis(
    per_session: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    """Compute Recall@K for all 2-way intersections of protected attributes.

    Parameters
    ----------
    per_session : pd.DataFrame
        Per-session evaluation frame **with group columns already merged**
        (i.e. after ``fairness.attach_groups``).  Must contain at least:
        ``hit``, ``ndcg``, ``rr`` and the four group columns in
        ``_ATTR_COLUMNS``.
    k : int
        List length that was used when computing hit/ndcg/rr (used only for
        labelling the output columns).

    Returns
    -------
    pd.DataFrame
        One row per (attr1-value, attr2-value) combination for every pair,
        plus a ``__GAP(max-min)__`` summary row per pair.

        Columns: ``attr1``, ``val1``, ``attr2``, ``val2``, ``n_sessions``,
        ``recall``, ``ndcg``, ``mrr``.
    """
    pairs = list(combinations(_ATTR_COLUMNS, 2))
    all_rows: list[dict] = []

    for attr1, attr2 in pairs:
        # Skip if the required columns are missing (defensive).
        if attr1 not in per_session.columns or attr2 not in per_session.columns:
            continue

        grp = per_session.groupby([attr1, attr2], observed=True)
        agg = grp.agg(
            n_sessions=("hit", "size"),
            recall=("hit", "mean"),
            ndcg=("ndcg", "mean"),
            mrr=("rr", "mean"),
        ).reset_index()

        for _, row in agg.iterrows():
            all_rows.append({
                "attr1": attr1,
                "val1": row[attr1],
                "attr2": attr2,
                "val2": row[attr2],
                "n_sessions": int(row["n_sessions"]),
                "recall": float(row["recall"]),
                "ndcg": float(row["ndcg"]),
                "mrr": float(row["mrr"]),
            })

        # Gap row: max – min across this pair's subgroups.
        if len(agg) > 0:
            all_rows.append({
                "attr1": attr1,
                "val1": "__GAP(max-min)__",
                "attr2": attr2,
                "val2": "__GAP(max-min)__",
                "n_sessions": int(agg["n_sessions"].sum()),
                "recall": float(agg["recall"].max() - agg["recall"].min()),
                "ndcg": float(agg["ndcg"].max() - agg["ndcg"].min()),
                "mrr": float(agg["mrr"].max() - agg["mrr"].min()),
            })

    return pd.DataFrame(all_rows)
