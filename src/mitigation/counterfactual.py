"""Counterfactual fairness reranker.

For each student, computes recommendations using both the student's actual
group affinity AND the counterfactual group's affinity, then averages the
scores.  This ensures the recommendation does not depend on group membership.

Conceptually, for a student in group A::

    score(item) = base_score(item)
                + 0.5 · affinity_A(item)      ← actual world
                + 0.5 · affinity_B(item)      ← counterfactual world

If the base model's ranking already captures personal taste and the two
affinity terms differ, averaging them smooths out the group-specific bias
while preserving the base model's relevance signal.

Post-processing approach — wraps any base recommender.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.data.protected import PROTECTED_ATTRS
from src.models.base import Context, Recommender

_CANDIDATE_MULT = 5  # fetch k * mult candidates for reranking pool


class CounterfactualReranker(Recommender):
    """Post-hoc counterfactual fairness wrapper for any recommender.

    Parameters
    ----------
    base : Recommender
        The underlying trained recommender whose candidate list is reranked.
    fair_attr : str
        Name of the protected attribute column (e.g. ``"imd_binary"``).
    candidate_mult : int
        Multiplier on *k* to determine how many candidates to fetch from the
        base model before reranking.

    Notes
    -----
    The reranker uses ``PROTECTED_ATTRS`` from :mod:`src.data.protected` to
    determine which two values constitute the privileged/unprivileged split
    and to identify each student's counterfactual group.
    """

    def __init__(
        self,
        base: Recommender,
        fair_attr: str = "imd_binary",
        candidate_mult: int = _CANDIDATE_MULT,
    ) -> None:
        self.base = base
        self.fair_attr = fair_attr
        self.candidate_mult = candidate_mult

        # Built in fit():
        self._student_group: dict[int, str] = {}
        self._group_affinity: dict[str, dict[int, float]] = {}

        # Resolve privileged/unprivileged values from PROTECTED_ATTRS
        self._priv_val: str | None = None
        self._unpriv_val: str | None = None
        for _attr, cfg in PROTECTED_ATTRS.items():
            if cfg["column"] == self.fair_attr:
                self._priv_val = cfg["privileged"]
                self._unpriv_val = cfg["unprivileged"]
                break

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"{self.base.name}_counterfactual"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, splits_df: pd.DataFrame) -> "CounterfactualReranker":
        """Fit the base model and build group-affinity lookup tables.

        Steps:
        1. Delegate to ``self.base.fit()``.
        2. Build student → group mapping.
        3. Build per-group item affinity (normalised frequency of each item
           in that group's training histories).
        """
        self.base.fit(splits_df)

        # Student → group lookup
        if self.fair_attr in splits_df.columns:
            self._student_group = dict(
                zip(
                    splits_df["id_student"].tolist(),
                    splits_df[self.fair_attr].tolist(),
                )
            )
        else:
            self._student_group = {}

        # Group → item frequency from training histories
        group_counters: dict[str, Counter] = defaultdict(Counter)
        for student_id, hist in zip(
            splits_df["id_student"], splits_df["train_history"]
        ):
            g = self._student_group.get(int(student_id))
            if g is None:
                continue
            group_counters[g].update(hist)

        # Normalise to [0, 1]
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
    def _counterfactual_group(self, group: str | None) -> str | None:
        """Return the 'other' group value for counterfactual reasoning.

        If the student is privileged, return unprivileged and vice versa.
        Returns ``None`` if the group is unknown or does not match either
        expected value.
        """
        if group == self._priv_val:
            return self._unpriv_val
        if group == self._unpriv_val:
            return self._priv_val
        return None

    def recommend(self, history: list[int], k: int, context: Context) -> list[int]:
        """Recommend by averaging actual and counterfactual group affinities.

        Steps:
        1. Fetch an extended candidate list from the base model.
        2. Compute ``base_score`` (inverse rank) for each candidate.
        3. Look up the student's actual group affinity scores.
        4. Look up the counterfactual group's affinity scores.
        5. Combine: ``combined = base_score + 0.5 * (actual_aff + counter_aff)``.
        6. Return the top-*k* items by combined score.
        """
        pool_size = min(k * self.candidate_mult, 500)
        candidates = self.base.recommend(history, pool_size, context)

        if not candidates:
            return []

        # Identify groups
        student_id = context.id_student
        group = self._student_group.get(student_id) if student_id is not None else None
        counter_group = self._counterfactual_group(group)

        actual_aff = self._group_affinity.get(group, {})
        counter_aff = self._group_affinity.get(counter_group, {}) if counter_group else {}

        # If neither group is known, fall back to the base ranking
        if not actual_aff and not counter_aff:
            return candidates[:k]

        n = len(candidates)
        base_scores = np.array([1.0 / (rank + 1) for rank in range(n)])

        actual_scores = np.array([actual_aff.get(item, 0.0) for item in candidates])
        counter_scores = np.array([counter_aff.get(item, 0.0) for item in candidates])

        # Normalise affinity vectors to [0, 1] for stable mixing
        for arr in (actual_scores, counter_scores):
            mx = arr.max()
            if mx > 0:
                arr /= mx

        # Combined score: base_score + 0.5 * (actual + counterfactual)
        combined = base_scores + 0.5 * (actual_scores + counter_scores)

        order = np.argsort(-combined)
        return [candidates[i] for i in order[:k]]
