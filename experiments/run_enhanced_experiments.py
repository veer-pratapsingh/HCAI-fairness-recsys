"""Master runner for enhanced models and post-processing mitigations.

Trains and evaluates:
  - Feature-Rich SASRec (sasrec_enhanced)
  - Adversarial SASRec (sasrec_adv_lamX)
  - Counterfactual Reranker (sasrec_counterfactual)
  - Calibrated Reranker (sasrec_calibrated)

Evaluates on accuracy, extended metrics (Coverage, Diversity, Novelty), and
demographic fairness metrics.
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
from src.eval.intersectional import intersectional_analysis
from src.utils import paths
from src.utils.seeds import set_seed, SEEDS

K = 10
N_NEG = 100

# Sweep configurations
LAMBDA_ADV_GRID = [0.01, 0.05, 0.1, 0.5]
LAMBDA_CAL_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]


def audit_model(model, splits, sequences, seed, quick=False):
    """Evaluate model on accuracy, extended metrics, and fairness."""
    print(f"  Evaluating {model.name} for seed {seed} ...", flush=True)
    eval_splits = splits
    if quick:
        eval_splits = splits.sample(n=min(10, len(splits)), random_state=seed).reset_index(drop=True)
    # Enable extended metrics computation
    metrics, per_session = evaluate_model(
        model, eval_splits, k=K, show_progress=False, keep_topk=True,
        extended_metrics=True, splits_df=splits
    )
    per_session = attach_groups(per_session, sequences)

    # Compute per-group accuracy gaps (View A)
    grp = per_group_accuracy(per_session, k=K)
    gaps = {}
    for attr in ["gender", "age_band", "disability", "imd_binary"]:
        val = grp.loc[
            (grp["attribute"] == attr) & (grp["group_value"] == "__GAP(max-min)__"),
            "recall"
        ]
        if len(val):
            gaps[f"{attr}_recall_gap"] = float(val.iloc[0])

    # Compute AIF360-style group fairness metrics (View B)
    fair = group_fairness_metrics(
        per_session, eval_splits["test_input"], sequences, k=K, n_neg=N_NEG, seed=seed
    )
    fair_metrics = {}
    for _, row in fair.iterrows():
        attr = row["attribute"]
        fair_metrics[f"{attr}_SPD"] = float(row["SPD"])
        fair_metrics[f"{attr}_EOD"] = float(row["EOD"])
        fair_metrics[f"{attr}_AOD"] = float(row["AOD"])

    # Combine metrics into results dict
    res = {
        "model": model.name,
        "seed": seed,
        "Recall@10": metrics[f"Recall@{K}"],
        "NDCG@10": metrics[f"NDCG@{K}"],
        "MRR": metrics["MRR"],
        "Coverage": metrics.get("Coverage", float("nan")),
        "Diversity_ILD": metrics.get("Diversity_ILD", float("nan")),
        "Novelty": metrics.get("Novelty", float("nan")),
    }
    res.update(gaps)
    res.update(fair_metrics)
    return res, grp, fair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--quick", action="store_true", help="Run 1 epoch and 1 seed for testing")
    args = parser.parse_args()

    paths.ensure_dirs()
    run_seeds = [int(s) for s in args.seeds.split(",")]
    if args.quick:
        run_seeds = [0]
        print("Running in QUICK mode: 1 seed, 1 epoch per model.")

    splits = load_splits(write=False)
    sequences = pd.read_parquet(paths.SEQUENCES_PARQUET)

    raw_results = []
    output_csv = paths.RESULTS_DIR / "raw_experiment_runs.csv"

    if output_csv.exists():
        try:
            existing_df = pd.read_csv(output_csv)
            raw_results = existing_df.to_dict(orient="records")
            print(f"Loaded {len(raw_results)} existing runs from {output_csv}")
        except Exception as e:
            print(f"Error loading existing results: {e}. Starting fresh.")

    def run_and_log(model, splits_df, seed):
        # Skip if already run
        if any(r["model"] == model.name and r["seed"] == seed for r in raw_results):
            print(f"  [Skip] {model.name} for seed {seed}")
            return
        
        t0 = time.time()
        model.fit(splits_df)
        res, grp, fair = audit_model(model, splits_df, sequences, seed, quick=args.quick)
        res["run_time_sec"] = time.time() - t0
        raw_results.append(res)
        
        # Save incrementally
        pd.DataFrame(raw_results).to_csv(output_csv, index=False)
        print(f"  Recall@10={res['Recall@10']:.4f}  IMD_EOD={res.get('imd_binary_EOD', 0.0):.4f}")
        
        # Save detailed logs per seed
        paths.ensure_dirs()
        grp.to_csv(paths.RESULTS_DIR / f"{model.name}_s{seed}_per_group.csv", index=False)
        fair.to_csv(paths.RESULTS_DIR / f"{model.name}_s{seed}_fairness.csv", index=False)

    for seed in run_seeds:
        print(f"\n==================== ENHANCED MODEL SEED {seed} ====================", flush=True)

        # First we need a base SASRec model for rerankers
        from src.models.sasrec import SASRecRecommender
        set_seed(seed)
        sasrec_kwargs = {
            "seed": seed,
            "verbose": False,
        }
        if args.quick:
            sasrec_kwargs["epochs"] = 1
            sasrec_kwargs["val_sample"] = 5

        print(f"  Fitting base SASRec for post-processing wrappers...", flush=True)
        sasrec_base = SASRecRecommender(**sasrec_kwargs)
        sasrec_base.fit(splits)

        # 1. Feature-Rich SASRec
        from src.models.sasrec_enhanced import EnhancedSASRecRecommender
        set_seed(seed)
        enhanced_kwargs = {
            "seed": seed,
            "verbose": False,
            "augment": "none",
        }
        if args.quick:
            enhanced_kwargs["epochs"] = 1
            enhanced_kwargs["val_sample"] = 5
        
        enhanced_model = EnhancedSASRecRecommender(**enhanced_kwargs)
        run_and_log(enhanced_model, splits, seed)

        # 2. Adversarial SASRec (Sweeping lambda_adv)
        from src.mitigation.adversarial import AdversarialSASRecRecommender
        for lam in (LAMBDA_ADV_GRID if not args.quick else [0.05]):
            set_seed(seed)
            adv_kwargs = {
                "lambda_adv": lam,
                "seed": seed,
                "verbose": False,
            }
            if args.quick:
                adv_kwargs["epochs"] = 1
                adv_kwargs["val_sample"] = 5
            
            adv_model = AdversarialSASRecRecommender(**adv_kwargs)
            run_and_log(adv_model, splits, seed)

        # 3. Counterfactual Reranker (Post-hoc)
        from src.mitigation.counterfactual import CounterfactualReranker
        set_seed(seed)
        cf_reranker = CounterfactualReranker(base=sasrec_base)
        run_and_log(cf_reranker, splits, seed)

        # 4. Calibrated Reranker (Post-hoc)
        from src.mitigation.calibration import CalibratedReranker
        for lam_cal in (LAMBDA_CAL_GRID if not args.quick else [0.5]):
            set_seed(seed)
            # Create a name suffix inside model instance for tracking
            cal_reranker = CalibratedReranker(base=sasrec_base, lambda_cal=lam_cal)
            run_and_log(cal_reranker, splits, seed)

    print("\nEnhanced models experiment run complete.", flush=True)


if __name__ == "__main__":
    main()
