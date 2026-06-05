"""Phase 2.1: Popularity baseline.

Recommends the most frequently visited activities within the session's course
presentation, excluding what the student has already seen. This is the floor
every other model must beat, and it doubles as the candidate generator for the
LLM recommender (Phase 2.4).

Popularity is counted from TRAIN history only (the `train_history` column), so
the baseline never peeks at validation/test targets.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from src.models.base import Context, Recommender


class PopularityRecommender(Recommender):
    name = "popularity"

    def __init__(self) -> None:
        # (module, presentation) -> list of item_idx sorted by descending frequency
        self._ranked: dict[tuple[str, str], list[int]] = {}

    def fit(self, splits_df: pd.DataFrame) -> "PopularityRecommender":
        counters: dict[tuple[str, str], Counter] = {}
        for module, presentation, hist in zip(
            splits_df["code_module"],
            splits_df["code_presentation"],
            splits_df["train_history"],
        ):
            key = (module, presentation)
            counters.setdefault(key, Counter()).update(hist)
        # Sort by count desc, then item_idx asc for deterministic ties.
        self._ranked = {
            key: [item for item, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))]
            for key, c in counters.items()
        }
        return self

    def recommend(self, history: list[int], k: int, context: Context) -> list[int]:
        seen = set(history)
        out: list[int] = []
        for item in self._ranked.get(context.key, []):
            if item not in seen:
                out.append(item)
                if len(out) == k:
                    break
        return out
