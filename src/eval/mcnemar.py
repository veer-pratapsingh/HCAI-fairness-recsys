"""Session-level significance testing via McNemar's test.

Why this exists
---------------
The multi-seed Wilcoxon signed-rank test we ran across 5 seeds has a hard
mathematical floor: with N=5 paired samples the smallest achievable two-sided
p-value is 1 / 2^(N-1) = 1/16 = 0.0625. It can NEVER reach p < 0.05, so it
cannot, even in principle, demonstrate significance. Reporting p = 0.0625 as
evidence of significance is therefore incorrect.

McNemar's test operates on the ~28,761 *paired per-session hit flags* instead of
5 seed-level means. Two models are evaluated on exactly the same test sessions,
so for each session we know whether model A hit, model B hit, both, or neither.
McNemar's test asks whether the disagreement counts (A-hit-B-miss vs
A-miss-B-hit) are balanced. With tens of thousands of paired sessions this has
ample power, and it is the standard significance test for paired classifier /
recommender comparisons on a shared test set.

Usage
-----
    from src.eval.mcnemar import mcnemar_from_hits, mcnemar_report

    # hits_a, hits_b are 0/1 arrays aligned by session (same order / same sessions)
    result = mcnemar_from_hits(hits_a, hits_b)
    print(result["p_value"], result["n_discordant"])

    # or drive it straight from two per_session frames sharing a session key:
    mcnemar_report(per_session_a, per_session_b, label_a="cf", label_b="sasrec")
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

_SESSION_KEYS = ["id_student", "code_module", "code_presentation"]


@dataclass
class McNemarResult:
    label_a: str
    label_b: str
    n_pairs: int
    a_only: int          # sessions A hit but B missed  (b = n01 or n10 depending on convention)
    b_only: int          # sessions B hit but A missed
    both: int
    neither: int
    n_discordant: int
    statistic: float
    p_value: float
    recall_a: float
    recall_b: float

    def as_row(self) -> dict:
        return asdict(self)


def _mcnemar_pvalue(n01: int, n10: int) -> tuple[float, float]:
    """Return (statistic, two-sided p-value).

    Uses the exact binomial test when the number of discordant pairs is small
    (< 25), otherwise the chi-square approximation with continuity correction.
    Falls back to a pure-Python binomial if scipy is unavailable.
    """
    n_disc = n01 + n10
    if n_disc == 0:
        return 0.0, 1.0

    # Exact binomial for small discordant counts.
    if n_disc < 25:
        try:
            from scipy.stats import binomtest
            p = binomtest(min(n01, n10), n_disc, 0.5, alternative="two-sided").pvalue
            return float(min(n01, n10)), float(p)
        except Exception:
            # Pure-python exact two-sided binomial.
            from math import comb
            k = min(n01, n10)
            tail = sum(comb(n_disc, i) for i in range(0, k + 1)) / (2 ** n_disc)
            return float(k), float(min(1.0, 2 * tail))

    # Chi-square approximation with continuity correction.
    stat = (abs(n01 - n10) - 1.0) ** 2 / n_disc
    try:
        from scipy.stats import chi2
        p = float(chi2.sf(stat, df=1))
    except Exception:
        # Survival function of chi-square with 1 dof = erfc(sqrt(stat/2)).
        from math import erfc, sqrt
        p = float(erfc(sqrt(stat / 2.0)))
    return float(stat), p


def mcnemar_from_hits(
    hits_a: np.ndarray,
    hits_b: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
) -> McNemarResult:
    """Compute McNemar's test from two aligned 0/1 hit arrays."""
    a = np.asarray(hits_a).astype(int)
    b = np.asarray(hits_b).astype(int)
    if a.shape != b.shape:
        raise ValueError(f"hit arrays must align: {a.shape} vs {b.shape}")

    both = int(np.sum((a == 1) & (b == 1)))
    neither = int(np.sum((a == 0) & (b == 0)))
    a_only = int(np.sum((a == 1) & (b == 0)))   # A hit, B miss
    b_only = int(np.sum((a == 0) & (b == 1)))   # B hit, A miss

    stat, p = _mcnemar_pvalue(a_only, b_only)

    return McNemarResult(
        label_a=label_a,
        label_b=label_b,
        n_pairs=int(a.shape[0]),
        a_only=a_only,
        b_only=b_only,
        both=both,
        neither=neither,
        n_discordant=a_only + b_only,
        statistic=stat,
        p_value=p,
        recall_a=float(a.mean()),
        recall_b=float(b.mean()),
    )


def _align_on_sessions(
    per_session_a: pd.DataFrame,
    per_session_b: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Inner-join two per-session frames on the session key, return aligned hits."""
    keys = [k for k in _SESSION_KEYS if k in per_session_a.columns and k in per_session_b.columns]
    if not keys:
        raise ValueError("no shared session-key columns to align on")

    a = per_session_a[keys + ["hit"]].rename(columns={"hit": "hit_a"})
    b = per_session_b[keys + ["hit"]].rename(columns={"hit": "hit_b"})
    merged = a.merge(b, on=keys, how="inner")
    if merged.empty:
        raise ValueError("no overlapping sessions between the two frames")
    return merged["hit_a"].to_numpy(), merged["hit_b"].to_numpy()


def mcnemar_report(
    per_session_a: pd.DataFrame,
    per_session_b: pd.DataFrame,
    label_a: str = "A",
    label_b: str = "B",
) -> McNemarResult:
    """Align two per-session frames on their session key and run McNemar."""
    hits_a, hits_b = _align_on_sessions(per_session_a, per_session_b)
    return mcnemar_from_hits(hits_a, hits_b, label_a=label_a, label_b=label_b)


def mcnemar_table(results: list[McNemarResult]) -> pd.DataFrame:
    """Collect several McNemarResult rows into a tidy DataFrame for the paper."""
    return pd.DataFrame([r.as_row() for r in results])