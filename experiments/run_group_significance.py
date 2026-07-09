"""Group-gap significance, bootstrap CIs, and multiple-comparison correction.

This addresses the reviewer's central criticism: the headline fairness finding
(the per-group recall gaps, e.g. IMD 4.66% vs 3.45%) was reported without any
significance test, confidence intervals, or the multiple-comparison correction
the paper itself recommends.

For each model and each binary protected attribute, this script:
  1. Computes the per-group Recall@10 and the gap (disadvantaged - advantaged).
  2. Bootstraps a 95% CI for that gap (resampling sessions with replacement).
  3. Runs a McNemar-style paired test... NO -- groups are different students, so
     the two groups are INDEPENDENT samples, not paired. We therefore use a
     two-proportion test (the recall gap is a difference of two hit-rates on
     disjoint session sets). We report:
        - the gap
        - bootstrap 95% CI of the gap
        - a two-sided p-value (bootstrap-based; also a normal-approx z-test)
  4. Applies Holm-Bonferroni correction across all (model x attribute) gap tests
     and marks which survive at alpha = 0.05.

Reads per-session prediction files already saved by the audit runs. It looks for
files matching results/*_per_session.csv OR reconstructs per-session hits from the
saved per_group files is NOT possible (those are aggregated), so we re-evaluate
the three headline models on the full test set to get per-session hits. This is
cheap for popularity/CF; SASRec trains once.

Output: results/group_gap_significance.csv

Usage:
    python -m experiments.run_group_significance
    python -m experiments.run_group_significance --seed 0 --n_boot 2000
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from src.data.splits import load_splits
from src.eval.accuracy import evaluate_model
from src.eval.fairness import attach_groups, load_sequences
from src.utils import paths
from src.utils.seeds import set_seed

K = 10

# Binary protected attributes and which value is the "disadvantaged"/focus group.
# gap is computed as recall(focus) - recall(other).
ATTRS = {
    "gender": ("F", "M"),            # focus F vs M (label only; direction is descriptive)
    "age_binary": ("35+", "0-35"),   # focus older vs younger
    "disability": ("Y", "N"),        # focus disabled vs not
    "imd_binary": ("disadvantaged", "advantaged"),
}


@dataclass
class GapResult:
    model: str
    attribute: str
    group_focus: str
    group_other: str
    n_focus: int
    n_other: int
    recall_focus: float
    recall_other: float
    gap: float
    ci_low: float
    ci_high: float
    p_value: float
    p_holm: float = np.nan
    significant_holm: bool = False

    def row(self):
        return asdict(self)


def _bootstrap_gap_ci(hits_focus, hits_other, n_boot, rng):
    """Bootstrap 95% CI for recall(focus) - recall(other)."""
    boot = np.empty(n_boot)
    nf, no = len(hits_focus), len(hits_other)
    for b in range(n_boot):
        bf = hits_focus[rng.integers(0, nf, nf)].mean()
        bo = hits_other[rng.integers(0, no, no)].mean()
        boot[b] = bf - bo
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # Two-sided bootstrap p-value: proportion of resamples on the far side of 0.
    if boot.mean() >= 0:
        p = 2 * np.mean(boot <= 0)
    else:
        p = 2 * np.mean(boot >= 0)
    p = float(min(1.0, p))
    return float(lo), float(hi), p


def _two_proportion_z_p(hits_focus, hits_other):
    """Normal-approx two-proportion z-test p-value (independent groups)."""
    from math import erfc, sqrt
    x1, n1 = hits_focus.sum(), len(hits_focus)
    x2, n2 = hits_other.sum(), len(hits_other)
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return float(erfc(abs(z) / sqrt(2)))


def _holm(pvals):
    """Holm-Bonferroni: return corrected p-values in original order.

    Sort ascending; the rank-i smallest p is multiplied by (m - i); corrected
    values are enforced monotone non-decreasing along that order and capped at 1.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    corrected = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running_max = max(running_max, val)
        corrected[idx] = running_max
    return corrected


def _build(name, seed):
    if name == "popularity":
        from src.models.popularity import PopularityRecommender
        return PopularityRecommender()
    if name == "cf":
        from src.models.cf import CFRecommender
        return CFRecommender(seed=seed)
    if name == "sasrec":
        from src.models.sasrec import SASRecRecommender
        return SASRecRecommender(seed=seed)
    raise SystemExit(f"unknown model {name}")


def _per_session_hits(name, splits, sequences, seed):
    set_seed(seed)
    model = _build(name, seed)
    model.fit(splits)
    _, per_session = evaluate_model(model, splits, k=K, keep_topk=False)
    return attach_groups(per_session, sequences)


def main(seed=0, n_boot=2000, models=("popularity", "cf", "sasrec")):
    paths.ensure_dirs()
    rng = np.random.default_rng(seed)
    splits = load_splits(write=False)
    sequences = load_sequences()

    results = []
    for name in models:
        print(f"[group-sig] evaluating {name} (seed {seed}) ...", flush=True)
        ps = _per_session_hits(name, splits, sequences, seed)

        for attr, (focus, other) in ATTRS.items():
            if attr not in ps.columns:
                continue
            hf = ps.loc[ps[attr] == focus, "hit"].to_numpy().astype(float)
            ho = ps.loc[ps[attr] == other, "hit"].to_numpy().astype(float)
            if len(hf) == 0 or len(ho) == 0:
                continue

            gap = hf.mean() - ho.mean()
            lo, hi, p_boot = _bootstrap_gap_ci(hf, ho, n_boot, rng)
            p_z = _two_proportion_z_p(hf, ho)
            # Use the more conservative of the two p-values.
            p = max(p_boot, p_z)

            results.append(GapResult(
                model=name, attribute=attr,
                group_focus=str(focus), group_other=str(other),
                n_focus=len(hf), n_other=len(ho),
                recall_focus=float(hf.mean()), recall_other=float(ho.mean()),
                gap=float(gap), ci_low=lo, ci_high=hi, p_value=float(p),
            ))
            print(f"  {name} {attr}: gap={gap:+.4f} "
                  f"CI[{lo:+.4f},{hi:+.4f}] p={p:.3f}")

    # Holm correction across ALL gap tests.
    pvals = np.array([r.p_value for r in results])
    p_holm = _holm(pvals)
    for r, ph in zip(results, p_holm):
        r.p_holm = float(ph)
        r.significant_holm = bool(ph < 0.05)

    df = pd.DataFrame([r.row() for r in results])
    out = paths.RESULTS_DIR / "group_gap_significance.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(df.to_string(index=False))

    n_sig = int(df["significant_holm"].sum())
    print(f"\n{n_sig} of {len(df)} group gaps significant after Holm correction (alpha=0.05).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_boot", type=int, default=2000)
    args = parser.parse_args()
    main(seed=args.seed, n_boot=args.n_boot)