"""Activity-type calibrated reranker.

Ensures the distribution of recommended activity types matches each group's
historical engagement distribution, using a greedy selection algorithm.

The intuition: if disadvantaged-IMD students historically engaged 40 % with
``resource`` activities, 30 % with ``quiz``, and 20 % with ``forumng``, their
recommendations should roughly mirror that mix rather than skewing toward
the privileged group's dominant activity types.

Algorithm (MMR-style greedy):
    Start with an empty result list.  At each step, pick the candidate that
    maximises::

        score(item) = (1 − λ_cal) · base_relevance(item)
                    +  λ_cal      · calibration_gain(item)

    where ``calibration_gain`` is the reduction in KL-divergence between the
    current result's activity-type distribution and the group's target
    distribution when adding ``item``.

Post-processing approach — wraps any base recommender.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.models.base import Context, Recommender
from src.utils import paths

_CANDIDATE_MULT = 5  # fetch k * mult candidates for reranking pool
_EPS = 1e-10  # smoothing constant for KL computation


class CalibratedReranker(Recommender):
    """Post-hoc calibrated reranker ensuring activity-type distribution parity.

    Parameters
    ----------
    base : Recommender
        The underlying trained recommender whose candidate list is reranked.
    fair_attr : str
        Name of the protected-attribute column (e.g. ``"imd_binary"``).
    lambda_cal : float
        Trade-off between base relevance (0.0) and calibration (1.0).
    candidate_mult : int
        Multiplier on *k* to determine how many candidates to fetch from the
        base model before the greedy selection.
    """

    def __init__(
        self,
        base: Recommender,
        fair_attr: str = "imd_binary",
        lambda_cal: float = 0.5,
        candidate_mult: int = _CANDIDATE_MULT,
    ) -> None:
        self.base = base
        self.fair_attr = fair_attr
        self.lambda_cal = lambda_cal
        self.candidate_mult = candidate_mult

        # Built in fit():
        self._student_group: dict[int, str] = {}
        # group_value → {activity_type: proportion}
        self._target_dist: dict[str, dict[str, float]] = {}
        # item_idx → activity_type
        self._item_activity: dict[int, str] = {}

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"{self.base.name}_calibrated_lam{self.lambda_cal}"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, splits_df: pd.DataFrame) -> "CalibratedReranker":
        """Fit the base model and build calibration targets.

        Steps:
        1. Delegate to ``self.base.fit()``.
        2. Build student → group mapping.
        3. Load ``item_vocab.parquet`` for item_idx → activity_type.
        4. Compute ``target_distribution``: for each group, the proportion of
           each activity type in training histories.
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

        # Item → activity type from the vocabulary
        vocab = pd.read_parquet(paths.ITEM_VOCAB_PARQUET)
        self._item_activity = dict(
            zip(vocab["item_idx"].tolist(), vocab["activity_type"].tolist())
        )

        # Per-group activity-type distribution from training histories
        group_activity_counts: dict[str, Counter] = defaultdict(Counter)
        for student_id, hist in zip(
            splits_df["id_student"], splits_df["train_history"]
        ):
            g = self._student_group.get(int(student_id))
            if g is None:
                continue
            for item_idx in hist:
                act = self._item_activity.get(item_idx)
                if act is not None:
                    group_activity_counts[g][act] += 1

        # Normalise to probability distributions
        self._target_dist = {}
        for g, counter in group_activity_counts.items():
            total = max(sum(counter.values()), 1)
            self._target_dist[g] = {
                act: count / total for act, count in counter.items()
            }

        return self

    # ------------------------------------------------------------------
    # KL-divergence helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _kl_divergence(
        current_dist: dict[str, float],
        target_dist: dict[str, float],
    ) -> float:
        """Compute KL(current || target) with Laplace smoothing.

        Both inputs are (possibly un-normalised) dicts mapping activity types
        to proportions.  Missing keys in either distribution are treated as
        having probability ``_EPS``.
        """
        all_keys = set(current_dist) | set(target_dist)
        if not all_keys:
            return 0.0

        kl = 0.0
        for key in all_keys:
            p = current_dist.get(key, _EPS)
            q = target_dist.get(key, _EPS)
            # Clamp both to avoid log(0)
            p = max(p, _EPS)
            q = max(q, _EPS)
            kl += p * np.log(p / q)
        return float(kl)

    def _distribution_of(self, activity_counts: Counter, total: int) -> dict[str, float]:
        """Convert raw counts to a normalised distribution."""
        if total == 0:
            return {}
        return {act: cnt / total for act, cnt in activity_counts.items()}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def recommend(self, history: list[int], k: int, context: Context) -> list[int]:
        """Recommend with greedy calibration-aware selection.

        Steps:
        1. Fetch an extended candidate list from the base model.
        2. For each of *k* slots, pick the candidate maximising:
           ``(1 − λ) · base_relevance + λ · calibration_gain``
           where calibration_gain = ``KL_before − KL_after`` for adding the
           candidate to the result set.
        3. Return the top-*k* items.
        """
        pool_size = min(k * self.candidate_mult, 500)
        candidates = self.base.recommend(history, pool_size, context)

        if not candidates:
            return []

        # Determine group and target distribution
        student_id = context.id_student
        group = self._student_group.get(student_id) if student_id is not None else None
        target = self._target_dist.get(group, {})

        # If no calibration target or lambda is 0, fall back to base ranking
        if not target or self.lambda_cal == 0.0:
            return candidates[:k]

        # Base relevance scores: inverse rank, normalised to [0, 1]
        n = len(candidates)
        base_scores = np.array([1.0 / (rank + 1) for rank in range(n)])
        base_max = base_scores[0]
        if base_max > 0:
            base_scores /= base_max

        # Greedy MMR-style selection
        selected: list[int] = []
        selected_activity_counts: Counter = Counter()
        remaining = set(range(n))  # indices into candidates

        for _ in range(min(k, n)):
            best_idx = -1
            best_score = -np.inf

            n_selected = len(selected)
            current_dist = self._distribution_of(selected_activity_counts, n_selected)
            kl_before = self._kl_divergence(current_dist, target) if n_selected > 0 else 0.0

            for idx in remaining:
                item = candidates[idx]
                act = self._item_activity.get(item, "__unknown__")

                # Compute KL if we add this item
                trial_counts = selected_activity_counts.copy()
                trial_counts[act] += 1
                trial_dist = self._distribution_of(trial_counts, n_selected + 1)
                kl_after = self._kl_divergence(trial_dist, target)

                # Calibration gain: how much KL *decreases* (higher = better)
                # For the first item, kl_before is 0, so we use −kl_after as gain
                cal_gain = kl_before - kl_after if n_selected > 0 else -kl_after

                # Normalise cal_gain for stable mixing (cap between -1 and 1)
                cal_gain = max(min(cal_gain, 1.0), -1.0)

                combined = (1.0 - self.lambda_cal) * base_scores[idx] + self.lambda_cal * cal_gain

                if combined > best_score:
                    best_score = combined
                    best_idx = idx

            if best_idx < 0:
                break

            selected.append(candidates[best_idx])
            act = self._item_activity.get(candidates[best_idx], "__unknown__")
            selected_activity_counts[act] += 1
            remaining.discard(best_idx)

        return selected
