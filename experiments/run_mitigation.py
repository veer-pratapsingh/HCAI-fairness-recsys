"""Phase 4: Mitigation sweep - RQ2 trade-off curves.

Runs both mitigation strategies over their hyperparameter ranges and produces
the accuracy-vs-fairness trade-off data that answers RQ2.

Fix 1 - Fair training (FairSASRec):
    Sweeps lam in LAM_GRID on the SASRec training loss.
    Each lam is one point on the trade-off curve.

Fix 2 - Post-hoc reranking:
    Wraps a pre-trained SASRec base with the group-calibrated reranker.
    Sweeps alpha in ALPHA_GRID (no retraining needed, so this is fast).

Both fixes use the same evaluation harness (run_audit) so numbers are comparable.

Usage:
    python -m experiments.run_mitigation --fix 1 --seed 0
    python -m experiments.run_mitigation --fix 2 --seed 0
    python -m experiments.run_mitigation --fix all --seed 0
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.data.splits import load_splits
from src.eval.accuracy import evaluate_model
from src.eval.fairness import attach_groups, group_fairness_metrics, per_group_accuracy
from src.utils import paths
from src.utils.seeds import set_seed

K = 10
N_NEG = 100

# RQ2 sweep grids (lam=0 and alpha=0 reproduce plain SASRec as the baseline point).
LAM_GRID = [0.0, 0.1, 0.5, 1.0, 2.0]
ALPHA_GRID = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]


def _audit(model, splits, sequences, seed):
    """Run the standard accuracy + fairness audit and return (metrics, grp, fair)."""
    metrics, per_session = evaluate_model(model, splits, k=K, keep_topk=True)
    per_session = attach_groups(per_session, sequences)
    grp = per_group_accuracy(per_session, k=K)
    fair = group_fairness_metrics(per_session, splits["test_input"], sequences,
                                  k=K, n_neg=N_NEG, seed=seed)
    return metrics, grp, fair


def _save(name, seed, metrics, grp, fair):
    grp.to_csv(paths.RESULTS_DIR / f"{name}_per_group.csv", index=False)
    fair.to_csv(paths.RESULTS_DIR / f"{name}_fairness.csv", index=False)
    pd.DataFrame([{"model": name, "seed": seed, **metrics}]).to_csv(
        paths.RESULTS_DIR / f"{name}_metrics.csv", index=False
    )
    print(f"  saved results/{name}_metrics/per_group/fairness.csv")


def run_fix1(seed: int) -> None:
    """Sweep lam for FairSASRec (Fix 1: fair training loss)."""
    from src.mitigation.fair_loss import FairSASRecRecommender

    splits = load_splits(write=False)
    sequences = pd.read_parquet(paths.SEQUENCES_PARQUET)
    # Group columns (gender, imd_binary, etc.) are already in splits.parquet.

    print(f"\n{'='*60}")
    print(f"Fix 1 - Fair training (FairSASRec)  seed={seed}")
    print(f"lam grid: {LAM_GRID}")
    print(f"{'='*60}")

    for lam in LAM_GRID:
        set_seed(seed)
        model = FairSASRecRecommender(lam=lam, seed=seed)
        print(f"\n--- lam={lam} | training ---")
        model.fit(splits)
        metrics, grp, fair = _audit(model, splits, sequences, seed)

        print(f"  Recall@10={metrics[f'Recall@{K}']:.4f}  "
              f"NDCG@10={metrics[f'NDCG@{K}']:.4f}  MRR={metrics['MRR']:.4f}")
        imd_gap = grp.loc[
            (grp["attribute"] == "imd_binary") & (grp["group_value"] == "__GAP(max-min)__"),
            "recall"
        ]
        if len(imd_gap):
            print(f"  IMD recall gap: {float(imd_gap.iloc[0]):.4f}")

        _save(model.name, seed, metrics, grp, fair)


def run_fix2(seed: int) -> None:
    """Sweep alpha for post-hoc reranking (Fix 2). Trains SASRec once, reranks N times."""
    from src.models.sasrec import SASRecRecommender
    from src.mitigation.rerank import RerankingRecommender

    splits = load_splits(write=False)
    sequences = pd.read_parquet(paths.SEQUENCES_PARQUET)
    # Group columns already in splits.parquet.

    print(f"\n{'='*60}")
    print(f"Fix 2 - Post-hoc reranking  seed={seed}")
    print(f"alpha grid: {ALPHA_GRID}")
    print(f"{'='*60}")

    # Train base SASRec once.
    set_seed(seed)
    print("\n--- Training base SASRec (shared across all alpha) ---")
    base = SASRecRecommender(seed=seed)
    base.fit(splits)

    for alpha in ALPHA_GRID:
        model = RerankingRecommender(base=base, alpha=alpha)
        model.fit(splits)   # only builds affinity tables; base is already trained
        metrics, grp, fair = _audit(model, splits, sequences, seed)

        print(f"\n--- alpha={alpha} ---")
        print(f"  Recall@10={metrics[f'Recall@{K}']:.4f}  "
              f"NDCG@10={metrics[f'NDCG@{K}']:.4f}  MRR={metrics['MRR']:.4f}")
        imd_gap = grp.loc[
            (grp["attribute"] == "imd_binary") & (grp["group_value"] == "__GAP(max-min)__"),
            "recall"
        ]
        if len(imd_gap):
            print(f"  IMD recall gap: {float(imd_gap.iloc[0]):.4f}")

        _save(model.name, seed, metrics, grp, fair)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", choices=["1", "2", "all"], default="all",
                        help="Which fix to run (1=fair loss, 2=reranking, all=both)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    paths.ensure_dirs()

    if args.fix in ("1", "all"):
        run_fix1(args.seed)
    if args.fix in ("2", "all"):
        run_fix2(args.seed)
