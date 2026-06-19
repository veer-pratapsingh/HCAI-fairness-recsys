"""Phase 4.2: Fix 2 - Post-hoc group-calibrated reranking.

After any base recommender (e.g. SASRec) produces a ranked candidate list, this
wrapper re-scores each item using a convex combination of:

    score(item, student) = (1 - alpha) * base_score(item)
                         +  alpha   * group_affinity(group, item)

where:
  - `base_score` is the item's rank position from the base model (inverted so
    rank 1 = highest score = 1.0), keeping the model's learned preference signal.
  - `group_affinity(g, i)` is the normalised training-set frequency of item i
    among sessions in group g - a proxy for "how relevant has this item historically
    been for students like this one?"
  - alpha in [0, 1] is the fairness strength knob:
        alpha = 0  ->  identical to the base model  (Fix 2 baseline point)
        alpha = 1  ->  pure group-affinity ranking  (maximum fairness nudge)

Why this helps: if disadvantaged-IMD students have historically engaged with
different activity types (e.g. more `resource` and less `quiz`), their group
affinity scores will promote those items, narrowing the per-group recall gap
without retraining.

The reranker works with any Recommender and adds group awareness post-hoc via the
`id_student` field on Context (populated by the evaluation harness since Phase 4).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.models.base import Context, Recommender

_CANDIDATE_MULT = 5   # fetch k * mult candidates from base model for reranking pool


class RerankingRecommender(Recommender):
    """Wraps any base recommender with group-calibrated score fusion."""

    def __init__(
        self,
        base: Recommender,
        alpha: float = 0.3,
        fair_attr: str = "imd_binary",
        candidate_mult: int = _CANDIDATE_MULT,
    ) -> None:
        self.base = base
        self.alpha = alpha
        self.fair_attr = fair_attr
        self.candidate_mult = candidate_mult

        # Built in fit():
        # student_id (int) -> group value string
        self._student_group: dict[int, str] = {}
        # group value -> {item_idx: normalised_affinity}
        self._group_affinity: dict[str, dict[int, float]] = {}

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"{self.base.name}_rerank_a{self.alpha}"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, splits_df: pd.DataFrame) -> "RerankingRecommender":
        # Fit base model first.
        self.base.fit(splits_df)

        # Build student -> group lookup.
        if self.fair_attr in splits_df.columns:
            self._student_group = dict(
                zip(splits_df["id_student"].tolist(),
                    splits_df[self.fair_attr].tolist())
            )
        else:
            self._student_group = {}

        # Build group -> item frequency from training histories.
        group_counters: dict[str, Counter] = defaultdict(Counter)
        for student_id, hist in zip(
            splits_df["id_student"], splits_df["train_history"]
        ):
            g = self._student_group.get(int(student_id))
            if g is None:
                continue
            group_counters[g].update(hist)

        # Normalise counters to [0, 1].
        self._group_affinity = {}
        for g, counter in group_counters.items():
            total = max(sum(counter.values()), 1)
            self._group_affinity[g] = {
                item: count / total for item, count in counter.items()
            }

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend(self, history: list[int], k: int, context: Context) -> list[int]:
        # Fetch extended candidate list from base model.
        pool_size = min(k * self.candidate_mult, 500)
        candidates = self.base.recommend(history, pool_size, context)

        if not candidates or self.alpha == 0.0:
            return candidates[:k]

        # Look up student group from context.
        student_id = context.id_student
        group = self._student_group.get(student_id) if student_id is not None else None
        affinity = self._group_affinity.get(group, {})

        # Base scores: rank 1 -> 1.0, rank N -> 1/N (inverted rank).
        n = len(candidates)
        base_scores = np.array([1.0 / (rank + 1) for rank in range(n)])

        # Group affinity scores for each candidate (0 if unseen in group training data).
        aff_scores = np.array([affinity.get(item, 0.0) for item in candidates])

        # Normalise affinity to [0, 1] for stable mixing.
        aff_max = aff_scores.max()
        if aff_max > 0:
            aff_scores /= aff_max

        # Combined score.
        combined = (1.0 - self.alpha) * base_scores + self.alpha * aff_scores

        # Rerank by combined score (descending).
        order = np.argsort(-combined)
        return [candidates[i] for i in order[:k]]
