"""Run hybrid fairness mitigation experiments (Fair SASRec + post-hoc Reranking).

Sweeps:
- lam=0.1, alpha=0.1
- lam=0.1, alpha=0.3
across all 5 seeds.
"""
from __future__ import annotations

import os
import sys
import time
import pandas as pd

# Add root directory to path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.splits import load_splits
from src.utils import paths
from src.utils.seeds import set_seed, SEEDS
from src.mitigation.fair_loss import FairSASRecRecommender
from src.mitigation.rerank import RerankingRecommender
from experiments.run_all_experiments import audit_model, compute_statistics


def run_hybrid_experiments():
    paths.ensure_dirs()
    
    # Load data splits and sequences
    splits = load_splits(write=False)
    sequences = pd.read_parquet(paths.SEQUENCES_PARQUET)
    
    output_csv = paths.RESULTS_DIR / "raw_experiment_runs.csv"
    raw_results = []
    
    # Load existing runs to allow resuming
    if output_csv.exists():
        try:
            existing_df = pd.read_csv(output_csv)
            raw_results = existing_df.to_dict(orient="records")
            print(f"Loaded {len(raw_results)} existing runs from {output_csv}")
        except Exception as e:
            print(f"Error reading existing raw results: {e}. Starting fresh.")
            
    # We sweep lam=0.1 and alpha in [0.1, 0.3]
    lam = 0.1
    alphas = [0.1, 0.3]
    
    for seed in SEEDS:
        print(f"\n==================== SEED {seed} ====================", flush=True)
        
        # Check if both configurations already exists for this seed
        configs_to_run = []
        for alpha in alphas:
            model_name = f"sasrec_hybrid_lam{lam}_alpha{alpha}"
            already_run = any(r["model"] == model_name and r["seed"] == seed for r in raw_results)
            if not already_run:
                configs_to_run.append(alpha)
            else:
                print(f"  [Skip] {model_name} for seed {seed} (already ran)")
                
        if not configs_to_run:
            continue
            
        # Set seed before training
        set_seed(seed)
        
        print(f"  [Train] FairSASRecRecommender(lam={lam}) for seed {seed} ...", flush=True)
        t_start = time.time()
        
        # Initialize and train base FairSASRecRecommender
        base_model = FairSASRecRecommender(lam=lam, seed=seed)
        base_model.fit(splits)
        
        # Monkey-patch the base model's fit method to be a no-op
        # so RerankingRecommender.fit() doesn't retrain it
        base_model.fit = lambda splits_df: base_model
        
        for alpha in configs_to_run:
            model_name = f"sasrec_hybrid_lam{lam}_alpha{alpha}"
            print(f"  [Run] {model_name} for seed {seed} ...", flush=True)
            
            t0 = time.time()
            # Wrap in RerankingRecommender
            hybrid_model = RerankingRecommender(base=base_model, alpha=alpha)
            hybrid_model.fit(splits)
            
            # Audit the model using the same metrics
            res = audit_model(hybrid_model, splits, sequences, seed)
            
            # Override model name to follow hybrid convention
            res["model"] = model_name
            res["run_time_sec"] = time.time() - t0 + (time.time() - t_start if alpha == configs_to_run[0] else 0)
            
            raw_results.append(res)
            
            # Save incrementally
            pd.DataFrame(raw_results).to_csv(output_csv, index=False)
            print(f"  [Saved] {model_name} for seed {seed} in {time.time() - t0:.1f}s")

    print("\nAll hybrid runs completed. Updating summary statistics...", flush=True)
    df_all = pd.read_csv(output_csv)
    compute_statistics(df_all)
    print("Summary statistics updated successfully.")


if __name__ == "__main__":
    run_hybrid_experiments()
