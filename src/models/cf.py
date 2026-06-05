"""Phase 2.2: Collaborative Filtering baseline.

"Recommends what similar students did." Matrix-factorization CF on the implicit
session x item matrix built from training history.

Implementation note (for the paper): the slides specify LightFM (WARP loss).
LightFM does not build on Python 3.11 (a known setup incompatibility), so we use
the `implicit` library instead. We default to ALS (Alternating Least Squares)
matrix factorisation, the standard implicit-feedback CF method, because it
supports recomputing a user's vector at inference time from arbitrary history
(`recalculate_user`) — exactly what our leave-last-out protocol needs. This is a
ceteris-paribus-preserving substitution: same task, same data, same evaluation.

Each session is treated as a "user". At inference we recompute the user vector
on the fly from the provided history (`recalculate_user=True`), so the model
never needs the held-out tail of the sequence, and we restrict candidates to the
session's own presentation (Decision D1).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.models.base import Context, Recommender


class CFRecommender(Recommender):
    name = "cf"

    def __init__(self, loss: str = "als", factors: int = 64, iterations: int = 20,
                 regularization: float = 0.01, seed: int = 0) -> None:
        self.loss = loss
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.seed = seed
        self._model = None
        self._n_items = 0
        self._valid_items: dict[tuple[str, str], np.ndarray] = {}

    def _build_model(self):
        if self.loss == "als":
            from implicit.als import AlternatingLeastSquares
            return AlternatingLeastSquares(
                factors=self.factors, regularization=self.regularization,
                iterations=self.iterations, random_state=self.seed,
            )
        from implicit.bpr import BayesianPersonalizedRanking
        return BayesianPersonalizedRanking(
            factors=self.factors, regularization=self.regularization,
            iterations=self.iterations, random_state=self.seed,
        )

    def fit(self, splits_df: pd.DataFrame) -> "CFRecommender":
        # Column space = item indices 1..V (0 is padding, left as an empty column).
        self._n_items = int(splits_df["train_history"].map(
            lambda h: max(h) if len(h) else 0).max()) + 1

        rows, cols, vals = [], [], []
        valid: dict[tuple[str, str], set] = {}
        for r, (module, pres, hist) in enumerate(zip(
            splits_df["code_module"], splits_df["code_presentation"], splits_df["train_history"],
        )):
            for item, count in Counter(hist).items():
                rows.append(r)
                cols.append(item)
                vals.append(float(count))
            valid.setdefault((module, pres), set()).update(hist)

        user_items = sp.csr_matrix(
            (vals, (rows, cols)),
            shape=(len(splits_df), self._n_items),
            dtype=np.float32,
        )
        self._valid_items = {k: np.fromiter(v, dtype=np.int64) for k, v in valid.items()}

        self._model = self._build_model()
        self._model.fit(user_items, show_progress=False)
        return self

    def recommend(self, history: list[int], k: int, context: Context) -> list[int]:
        candidates = self._valid_items.get(context.key)
        if candidates is None or len(candidates) == 0 or not history:
            return []
        # One-row implicit matrix for this session's history.
        counts = Counter(history)
        row = sp.csr_matrix(
            (np.array(list(counts.values()), dtype=np.float32),
             (np.zeros(len(counts), dtype=np.int64), np.array(list(counts.keys()), dtype=np.int64))),
            shape=(1, self._n_items), dtype=np.float32,
        )
        n = min(k, len(candidates))
        ids, _ = self._model.recommend(
            0, row, N=n, items=candidates,
            filter_already_liked_items=True, recalculate_user=True,
        )
        return [int(i) for i in ids]
