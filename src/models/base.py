"""Shared recommender interface so every model is trained and evaluated identically.

All recommenders implement:
    fit(splits_df)                       -> learn from training history
    recommend(history, k, context)       -> ranked list of item indices (best first)

`context` carries side information a model may need (the session's module +
presentation), since OULAD VLE sites are presentation-specific (Decision D1):
a recommender must only ever return items that exist in the session's presentation.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Context:
    """Per-session side information passed to recommend()."""
    code_module: str
    code_presentation: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.code_module, self.code_presentation)


class Recommender:
    """Abstract base. Subclasses override fit() and recommend()."""

    name: str = "base"

    def fit(self, splits_df: pd.DataFrame) -> "Recommender":
        raise NotImplementedError

    def recommend(self, history: list[int], k: int, context: Context) -> list[int]:
        """Return up to k item indices, ranked best-first, excluding `history`."""
        raise NotImplementedError
