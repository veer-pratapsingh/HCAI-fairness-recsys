"""Produce the session-level McNemar significance table for the headline claims.

The multi-seed Wilcoxon test cannot go below p=0.0625 at N=5 (see
src/eval/mcnemar.py for the full explanation), so it cannot demonstrate
significance. This script evaluates the models to be compared on the SAME test
sessions (same seed, same split), collects per-session hit flags, and runs
McNemar's test over the ~28,761 paired sessions -- the statistically correct way
to test paired recommender comparisons on a shared test set.

Comparisons produced (all at seed 0 on the full test set):
    - cf            vs sasrec               (RQ1 headline: is CF really better?)
    - sasrec        vs sasrec_fair_lam1.0   (RQ2: does fair-loss change accuracy?)
    - sasrec        vs sasrec_rerank_a0.7   (RQ2: does reranking change accuracy?)

Output: results/mcnemar_significance.csv

Usage:
    python -m experiments.run_mcnemar
    python -m experiments.run_mcnemar --seed 0
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.data.splits import load_splits
from src.eval.accuracy import evaluate_model
from src.eval.mcnemar import mcnemar_report, mcnemar_table, McNemarResult
from src.utils import paths
from src.utils.seeds import set_seed

K = 10


def _fit_eval(model, splits) -> pd.DataFrame:
    """Fit on full training data, evaluate on the full test set, return per-session."""
    model.fit(splits)
    _, per_session = evaluate_model(model, splits, k=K, keep_topk=False)
    return per_session


def _build(name: str, seed: int):
    """Construct a model by name. Mirrors experiments/run_audit.py builders."""
    if name == "cf":
        from src.models.cf import CFRecommender
        return CFRecommender(seed=seed)
    if name == "sasrec":
        from src.models.sasrec import SASRecRecommender
        return SASRecRecommender(seed=seed)
    if name == "sasrec_fair_lam1.0":
        from src.mitigation.fair_loss import FairSASRecRecommender
        return FairSASRecRecommender(lam=1.0, seed=seed)
    if name == "sasrec_rerank_a0.7":
        from src.models.sasrec import SASRecRecommender
        from src.mitigation.rerank import RerankingRecommender
        return RerankingRecommender(base=SASRecRecommender(seed=seed), alpha=0.7)
    raise SystemExit(f"Unknown model '{name}'")


# (label_a, label_b) pairs to compare
COMPARISONS = [
    ("cf", "sasrec"),
    ("sasrec", "sasrec_fair_lam1.0"),
    ("sasrec", "sasrec_rerank_a0.7"),
]


def main(seed: int = 0) -> None:
    paths.ensure_dirs()
    set_seed(seed)
    splits = load_splits(write=False)

    # Cache per-session frames so a model used in two comparisons is only run once.
    cache: dict[str, pd.DataFrame] = {}

    def get_per_session(name: str) -> pd.DataFrame:
        if name not in cache:
            print(f"[mcnemar] evaluating {name} (seed {seed}) ...", flush=True)
            set_seed(seed)  # reset before each model for identical sampling
            cache[name] = _fit_eval(_build(name, seed), splits)
        return cache[name]

    results: list[McNemarResult] = []
    for a, b in COMPARISONS:
        psa = get_per_session(a)
        psb = get_per_session(b)
        res = mcnemar_report(psa, psb, label_a=a, label_b=b)
        results.append(res)
        print(
            f"[mcnemar] {a} vs {b}: "
            f"recall_a={res.recall_a:.4f} recall_b={res.recall_b:.4f} "
            f"discordant={res.n_discordant} "
            f"({a}-only={res.a_only}, {b}-only={res.b_only}) "
            f"p={res.p_value:.3e}"
        )

    table = mcnemar_table(results)
    out = paths.RESULTS_DIR / "mcnemar_significance.csv"
    table.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed)