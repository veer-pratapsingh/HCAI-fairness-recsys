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

# For LLM runs the full test set (~28k sessions) takes ~11 h and is expensive.
# Decision D8: evaluate on a fixed random sample for a fair, affordable comparison.
LLM_DEFAULT_N = 500

MODELS = ["popularity", "cf", "sasrec", "llm"]

# Set DEEPSEEK_API_KEY in your environment (never commit the raw key).
import os as _os
DEEPSEEK_API_KEY = _os.environ.get("DEEPSEEK_API_KEY", "")


def build_model(name: str, seed: int):
    if name == "popularity":
        return PopularityRecommender()
    if name == "cf":
        return CFRecommender(seed=seed)
    if name == "sasrec":
        from src.models.sasrec import SASRecRecommender
        return SASRecRecommender(seed=seed)
    if name == "llm":
        from src.models.llm import LLMRecommender
        return LLMRecommender(api_key=DEEPSEEK_API_KEY, seed=seed)
    raise SystemExit(f"Unknown model '{name}'. Available: {MODELS}")


def main(model_name: str, seed: int, n_sessions: int | None = None) -> None:
    paths.ensure_dirs()
    set_seed(seed)

    splits = load_splits(write=False)
    sequences = pd.read_parquet(paths.SEQUENCES_PARQUET)

    # For the LLM model, sample a fixed subset of the test set (Decision D8).
    test_splits = splits
    if model_name == "llm":
        cap = n_sessions if n_sessions is not None else LLM_DEFAULT_N
        test_splits = splits.sample(n=min(cap, len(splits)), random_state=seed).reset_index(drop=True)
        print(f"[LLM] Evaluating on {len(test_splits)} sampled sessions (of {len(splits)} total)")

    # --- predictions (one pass, reused for both accuracy and fairness) ---
    model = build_model(model_name, seed).fit(splits)  # always fit on full training data
    metrics, per_session = evaluate_model(model, test_splits, k=K, keep_topk=True)
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
        per_session, test_splits["test_input"], sequences, k=K, n_neg=N_NEG, seed=seed
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
    parser.add_argument(
        "--n_sessions", type=int, default=None,
        help="For LLM: number of test sessions to evaluate (default 500). Ignored for other models."
    )
    args = parser.parse_args()
    main(args.model, args.seed, args.n_sessions)


