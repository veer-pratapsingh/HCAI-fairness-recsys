"""Phase 3.1 + evaluation harness: accuracy metrics and a model runner.

Metrics (single relevant target per session, leave-last-out):
  - Recall@K : 1 if the true next item is in the top-K, else 0 (a.k.a. HR@K)
  - NDCG@K   : 1 / log2(rank+1) if hit within K, else 0
  - MRR@K    : 1 / rank if hit within K, else 0

Extended metrics (opt-in via ``extended_metrics=True``):
  - Coverage      : fraction of unique items recommended vs total items (per
                    presentation, then averaged).
  - Diversity_ILD : Intra-List Diversity – average distinct activity_types in
                    top-k list divided by k.
  - Novelty       : mean inverse popularity (-log2) of recommended items.

`evaluate_model` runs any Recommender over the test split and returns both the
aggregate metrics and a per-session DataFrame (hit flag + rank). The per-session
frame is what the fairness evaluator (Phase 3.2) consumes, so accuracy and
fairness always come from exactly the same predictions.
"""
from __future__ import annotations

from collections import Counter
from math import log2

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.models.base import Context, Recommender
from src.utils import paths


def rank_of_target(ranked: list[int], target: int) -> int | None:
    """1-based rank of target in the ranked list, or None if absent."""
    for i, item in enumerate(ranked, start=1):
        if item == target:
            return i
    return None


# ---------------------------------------------------------------------------
# Extended-metric helpers
# ---------------------------------------------------------------------------

def _item_frequencies(splits_df: pd.DataFrame) -> dict[int, int]:
    """Count how often each item appears in training histories.

    Returns {item_idx: count}.
    """
    freq: Counter = Counter()
    for hist in splits_df["train_history"]:
        freq.update(hist)
    return dict(freq)


def _load_item_activity_map() -> dict[int, str]:
    """Load item_vocab.parquet and return {item_idx: activity_type}."""
    vocab = pd.read_parquet(paths.ITEM_VOCAB_PARQUET)
    return dict(zip(vocab["item_idx"].tolist(), vocab["activity_type"].tolist()))


def _compute_extended_metrics(
    per_session: pd.DataFrame,
    splits_df: pd.DataFrame | None,
    k: int,
) -> dict[str, float]:
    """Compute Coverage, Diversity_ILD and Novelty from per-session top-k lists.

    Requires ``per_session`` to carry the ``topk`` column (list of item indices)
    and ``code_module``/``code_presentation`` columns.

    Parameters
    ----------
    per_session : pd.DataFrame
        Per-session evaluation frame with ``topk`` column.
    splits_df : pd.DataFrame | None
        Full splits frame for computing item frequencies.  If *None*, Novelty
        is set to NaN.
    k : int
        Length of recommendation list.

    Returns
    -------
    dict with keys ``Coverage``, ``Diversity_ILD``, ``Novelty``.
    """
    # --- 1. Coverage: per-presentation fraction of unique recommended items ---
    total_items_by_pres: dict[tuple[str, str], set[int]] = {}
    rec_items_by_pres: dict[tuple[str, str], set[int]] = {}
    for row in per_session.itertuples(index=False):
        key = (row.code_module, row.code_presentation)
        rec_items_by_pres.setdefault(key, set()).update(row.topk)

    # Total items per presentation from splits_df (train_history union)
    if splits_df is not None:
        for mod, pres, hist in zip(
            splits_df["code_module"],
            splits_df["code_presentation"],
            splits_df["train_history"],
        ):
            total_items_by_pres.setdefault((mod, pres), set()).update(hist)

    coverages = []
    for key, rec_set in rec_items_by_pres.items():
        total_set = total_items_by_pres.get(key)
        if total_set and len(total_set) > 0:
            coverages.append(len(rec_set) / len(total_set))
    coverage = float(np.mean(coverages)) if coverages else float("nan")

    # --- 2. Diversity_ILD: avg distinct activity_types / k ---
    try:
        act_map = _load_item_activity_map()
    except FileNotFoundError:
        act_map = {}

    if act_map:
        div_scores = []
        for topk in per_session["topk"]:
            types = {act_map.get(item, "unknown") for item in topk}
            div_scores.append(len(types) / max(k, 1))
        diversity = float(np.mean(div_scores))
    else:
        diversity = float("nan")

    # --- 3. Novelty: mean -log2(popularity) of recommended items ---
    if splits_df is not None:
        freq = _item_frequencies(splits_df)
        total_interactions = sum(freq.values())
        if total_interactions > 0:
            nov_scores = []
            for topk in per_session["topk"]:
                for item in topk:
                    pop = freq.get(item, 1) / total_interactions
                    nov_scores.append(-log2(max(pop, 1e-12)))
            novelty = float(np.mean(nov_scores)) if nov_scores else 0.0
        else:
            novelty = float("nan")
    else:
        novelty = float("nan")

    return {"Coverage": coverage, "Diversity_ILD": diversity, "Novelty": novelty}


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Recommender,
    test_df: pd.DataFrame,
    k: int = 10,
    show_progress: bool = True,
    keep_topk: bool = False,
    extended_metrics: bool = False,
    splits_df: pd.DataFrame | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Run ``model`` over the test split.

    Uses the ``test_input`` history and ``test_target`` columns from
    splits.parquet.  Returns ``(aggregate_metrics, per_session_df)``.

    Parameters
    ----------
    model : Recommender
        Any fitted recommender.
    test_df : pd.DataFrame
        Test-split frame with ``test_input`` and ``test_target`` columns.
    k : int
        Length of the recommendation list.
    show_progress : bool
        Show a tqdm progress bar.
    keep_topk : bool
        If *True*, the per-session frame additionally carries the full ranked
        ``topk`` list and the ``target``.
    extended_metrics : bool
        If *True*, also compute Coverage, Diversity_ILD and Novelty.  This
        forces ``keep_topk=True`` internally (the top-k lists are needed for
        extended metrics).
    splits_df : pd.DataFrame | None
        Full splits frame, needed for computing item popularity (Novelty) and
        Coverage.  Only used when ``extended_metrics=True``.

    Returns
    -------
    (metrics_dict, per_session_df)
    """
    # Extended metrics require the top-k lists to be stored.
    _keep_topk = keep_topk or extended_metrics

    records = []
    rows = test_df.itertuples(index=False)
    if show_progress:
        rows = tqdm(rows, total=len(test_df), desc=f"eval:{model.name}")

    for row in rows:
        ctx = Context(row.code_module, row.code_presentation, id_student=row.id_student)
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
        if _keep_topk:
            rec["target"] = row.test_target
            rec["topk"] = ranked
        records.append(rec)

    per_session = pd.DataFrame(records)
    metrics: dict[str, float] = {
        f"Recall@{k}": float(per_session["hit"].mean()),
        f"NDCG@{k}": float(per_session["ndcg"].mean()),
        "MRR": float(per_session["rr"].mean()),
        "n_sessions": int(len(per_session)),
    }

    if extended_metrics:
        ext = _compute_extended_metrics(per_session, splits_df, k)
        metrics.update(ext)

    # If caller did not request keep_topk but we forced it for extended
    # metrics, drop the helper columns so the output shape stays compatible.
    if extended_metrics and not keep_topk:
        per_session = per_session.drop(columns=["target", "topk"], errors="ignore")

    return metrics, per_session
