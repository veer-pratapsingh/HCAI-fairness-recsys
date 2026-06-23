"""Run the NVIDIA Nemotron LLM recommender across multiple seeds.

Usage:
    python -m experiments.run_nvidia_llm --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from src.data.splits import load_splits
from src.eval.accuracy import evaluate_model
from src.eval.fairness import (
    attach_groups,
    group_fairness_metrics,
    per_group_accuracy,
)
from src.models.llm_nvidia import NvidiaLLMRecommender
from src.utils import paths
from src.utils.seeds import set_seed

K = 10
N_NEG = 100
LLM_DEFAULT_N = 500

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")


def audit_model(model, splits, sequences, seed):
    """Evaluate accuracy and fairness, return flattened metrics dictionary."""
    metrics, per_session = evaluate_model(model, splits, k=K, show_progress=True, keep_topk=True)
    per_session = attach_groups(per_session, sequences)

    grp = per_group_accuracy(per_session, k=K)
    gaps = {}
    for attr in ["gender", "age_band", "disability", "imd_binary"]:
        val = grp.loc[
            (grp["attribute"] == attr) & (grp["group_value"] == "__GAP(max-min)__"),
            "recall"
        ]
        if len(val):
            gaps[f"{attr}_recall_gap"] = float(val.iloc[0])

    fair = group_fairness_metrics(
        per_session, splits["test_input"], sequences, k=K, n_neg=N_NEG, seed=seed
    )
    fair_metrics = {}
    for _, row in fair.iterrows():
        attr = row["attribute"]
        fair_metrics[f"{attr}_SPD"] = float(row["SPD"])
        fair_metrics[f"{attr}_EOD"] = float(row["EOD"])
        fair_metrics[f"{attr}_AOD"] = float(row["AOD"])

    res = {
        "model": model.name,
        "seed": seed,
        "Recall@10": metrics[f"Recall@{K}"],
        "NDCG@10": metrics[f"NDCG@{K}"],
        "MRR": metrics["MRR"],
    }
    res.update(gaps)
    res.update(fair_metrics)
    return res, grp, fair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--n_sessions", type=int, default=LLM_DEFAULT_N)
    args = parser.parse_args()

    if not NVIDIA_API_KEY:
        print("ERROR: NVIDIA_API_KEY not set. Set it via environment variable.")
        return

    paths.ensure_dirs()

    run_seeds = [int(s) for s in args.seeds.split(",")]
    splits = load_splits(write=False)
    sequences = pd.read_parquet(paths.SEQUENCES_PARQUET)

    raw_results = []
    output_csv = paths.RESULTS_DIR / "nvidia_llm_runs.csv"

    # Resume from existing results
    if output_csv.exists():
        try:
            existing_df = pd.read_csv(output_csv)
            raw_results = existing_df.to_dict(orient="records")
            print(f"Loaded {len(raw_results)} existing NVIDIA runs from {output_csv}")
        except Exception:
            pass

    for seed in run_seeds:
        if any(r["model"] == "llm_nvidia" and r["seed"] == seed for r in raw_results):
            print(f"[Skip] llm_nvidia seed {seed} already ran")
            continue

        print(f"\n===== NVIDIA Nemotron LLM | Seed {seed} =====", flush=True)
        set_seed(seed)

        model = NvidiaLLMRecommender(api_key=NVIDIA_API_KEY, seed=seed)
        model.fit(splits)

        # Sample test sessions (Decision D8)
        test_splits = splits.sample(
            n=min(args.n_sessions, len(splits)), random_state=seed
        ).reset_index(drop=True)

        print(f"  Evaluating on {len(test_splits)} sampled sessions...", flush=True)
        t0 = time.time()
        res, grp, fair = audit_model(model, test_splits, sequences, seed)
        res["run_time_sec"] = time.time() - t0

        print(f"  Recall@10={res['Recall@10']:.4f}  NDCG@10={res['NDCG@10']:.4f}  MRR={res['MRR']:.4f}")
        print(f"  Time: {res['run_time_sec']:.1f}s")

        # Save per-seed results
        grp.to_csv(paths.RESULTS_DIR / f"llm_nvidia_s{seed}_per_group.csv", index=False)
        fair.to_csv(paths.RESULTS_DIR / f"llm_nvidia_s{seed}_fairness.csv", index=False)

        raw_results.append(res)
        pd.DataFrame(raw_results).to_csv(output_csv, index=False)
        print(f"  Saved to {output_csv}")

    # Print summary
    df = pd.DataFrame(raw_results)
    print("\n===== NVIDIA Nemotron LLM Summary =====")
    for metric in ["Recall@10", "NDCG@10", "MRR"]:
        vals = df[metric].to_numpy()
        print(f"  {metric}: {np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}")

    print("\nNVIDIA Nemotron LLM experiment complete.", flush=True)


if __name__ == "__main__":
    main()
