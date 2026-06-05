"""Phase 3.1 + evaluation harness: accuracy metrics and a model runner.

Metrics (single relevant target per session, leave-last-out):
  - Recall@K : 1 if the true next item is in the top-K, else 0 (a.k.a. HR@K)
  - NDCG@K   : 1 / log2(rank+1) if hit within K, else 0
  - MRR@K    : 1 / rank if hit within K, else 0

`evaluate_model` runs any Recommender over the test split and returns both the
aggregate metrics and a per-session DataFrame (hit flag + rank). The per-session
frame is what the fairness evaluator (Phase 3.2) consumes, so accuracy and
fairness always come from exactly the same predictions.
"""
from __future__ import annotations

from math import log2

import pandas as pd
from tqdm import tqdm

from src.models.base import Context, Recommender


def rank_of_target(ranked: list[int], target: int) -> int | None:
    """1-based rank of target in the ranked list, or None if absent."""
    for i, item in enumerate(ranked, start=1):
        if item == target:
            return i
    return None


def evaluate_model(
    model: Recommender,
    test_df: pd.DataFrame,
    k: int = 10,
    show_progress: bool = True,
    keep_topk: bool = False,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Run `model` over the test split.

    Uses the `test_input` history and `test_target` columns from splits.parquet.
    Returns (aggregate_metrics, per_session_df).

    If `keep_topk` is True, the per-session frame additionally carries the full
    ranked `topk` list and the `target` — the fairness evaluator (Phase 3.2)
    needs these so it scores the very same predictions.
    """
    records = []
    rows = test_df.itertuples(index=False)
    if show_progress:
        rows = tqdm(rows, total=len(test_df), desc=f"eval:{model.name}")

    for row in rows:
        ctx = Context(row.code_module, row.code_presentation)
        ranked = model.recommend(list(row.test_input), k, ctx)
        rank = rank_of_target(ranked, row.test_target)
        hit = rank is not None and rank <= k
        rec = {
            "id_student": row.id_student,
            "code_module": row.code_module,
            "code_presentation": row.code_presentation,
            "rank": rank if rank is not None else 0,
            "hit": int(hit),
            "ndcg": (1.0 / log2(rank + 1)) if hit else 0.0,
            "rr": (1.0 / rank) if hit else 0.0,
        }
        if keep_topk:
            rec["target"] = row.test_target
            rec["topk"] = ranked
        records.append(rec)

    per_session = pd.DataFrame(records)
    metrics = {
        f"Recall@{k}": float(per_session["hit"].mean()),
        f"NDCG@{k}": float(per_session["ndcg"].mean()),
        "MRR": float(per_session["rr"].mean()),
        "n_sessions": int(len(per_session)),
    }
    return metrics, per_session
