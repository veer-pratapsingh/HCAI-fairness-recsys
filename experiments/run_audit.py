"""Full audit of one recommender: accuracy + fairness, from identical predictions.

    python -m experiments.run_audit            # default: popularity
    python -m experiments.run_audit --model popularity

This is the reusable template: every later model (CF / SASRec / LLM) plugs into
the same harness, so RQ1 and RQ3 numbers are always produced the same way.
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.data.splits import load_splits
from src.eval.accuracy import evaluate_model
from src.eval.fairness import (
    attach_groups,
    group_fairness_metrics,
    per_group_accuracy,
)
from src.models.cf import CFRecommender
from src.models.popularity import PopularityRecommender
from src.utils import paths
from src.utils.seeds import set_seed

K = 10
N_NEG = 100

MODELS = ["popularity", "cf", "sasrec"]


def build_model(name: str, seed: int):
    if name == "popularity":
        return PopularityRecommender()
    if name == "cf":
        return CFRecommender(seed=seed)
    if name == "sasrec":
        from src.models.sasrec import SASRecRecommender
        return SASRecRecommender(seed=seed)
    raise SystemExit(f"Unknown model '{name}'. Available: {MODELS}")


def main(model_name: str, seed: int) -> None:
    paths.ensure_dirs()
    set_seed(seed)

    splits = load_splits(write=False)
    sequences = pd.read_parquet(paths.SEQUENCES_PARQUET)

    # --- predictions (one pass, reused for both accuracy and fairness) ---
    model = build_model(model_name, seed).fit(splits)
    metrics, per_session = evaluate_model(model, splits, k=K, keep_topk=True)
    per_session = attach_groups(per_session, sequences)

    print(f"\n=== {model.name} | accuracy (overall) ===")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    # --- View A: per-group accuracy + gap (RQ3) ---
    grp = per_group_accuracy(per_session, k=K)
    print(f"\n=== {model.name} | per-group Recall@{K} (RQ3) ===")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(grp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- View B: SPD / EOD / AOD per binary attribute ---
    fair = group_fairness_metrics(
        per_session, splits["test_input"], sequences, k=K, n_neg=N_NEG, seed=seed
    )
    print(f"\n=== {model.name} | SPD / EOD / AOD (ideal=0) ===")
    print(fair.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- persist ---
    grp.to_csv(paths.RESULTS_DIR / f"{model.name}_per_group.csv", index=False)
    fair.to_csv(paths.RESULTS_DIR / f"{model.name}_fairness.csv", index=False)
    pd.DataFrame([{"model": model.name, "seed": seed, **metrics}]).to_csv(
        paths.RESULTS_DIR / f"{model.name}_metrics.csv", index=False
    )
    print(f"\nWrote results/{model.name}_{{metrics,per_group,fairness}}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="popularity", choices=MODELS)
    parser.add_argument("--seed", type=int, default=0)
    main(*vars(parser.parse_args()).values())


