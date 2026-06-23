"""Run the LLM recommender with CF-based retrieval on seed 0.

Usage:
    python -m experiments.run_cf_retrieval_llm
"""
from __future__ import annotations

import os
import time
import pandas as pd

from src.data.splits import load_splits
from src.eval.accuracy import evaluate_model
from src.eval.fairness import (
    attach_groups,
    group_fairness_metrics,
    per_group_accuracy,
)
from src.models.cf import CFRecommender
from src.models.llm import LLMRecommender
from src.utils import paths
from src.utils.seeds import set_seed

K = 10
N_NEG = 100
LLM_DEFAULT_N = 500

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

def main():
    if not DEEPSEEK_API_KEY:
        print("ERROR: DEEPSEEK_API_KEY environment variable not set.")
        return
        
    paths.ensure_dirs()
    set_seed(0)
    
    splits = load_splits(write=False)
    sequences = pd.read_parquet(paths.SEQUENCES_PARQUET)
    
    # Sample seed 0 test splits
    test_splits = splits.sample(n=min(LLM_DEFAULT_N, len(splits)), random_state=0).reset_index(drop=True)
    print(f"Evaluating LLM with CF-based retrieval on {len(test_splits)} sessions...")
    
    # Initialize the CF retriever
    cf_retriever = CFRecommender(seed=0)
    
    # Initialize and fit the model with CF retriever and use_rank_hint=True (since rank hint stabilizes reranking)
    model = LLMRecommender(
        api_key=DEEPSEEK_API_KEY, 
        seed=0, 
        use_rank_hint=True,
        retriever=cf_retriever
    )
    model.fit(splits)
    
    t0 = time.time()
    metrics, per_session = evaluate_model(model, test_splits, k=K, keep_topk=True)
    duration = time.time() - t0
    
    per_session = attach_groups(per_session, sequences)
    
    print(f"\n=== llm_cfretrieval | accuracy (overall) ===")
    for key, val in metrics.items():
        print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")
    print(f"  Time taken: {duration:.1f}s")
    
    # View A: per-group accuracy + gap
    grp = per_group_accuracy(per_session, k=K)
    
    # View B: SPD / EOD / AOD per binary attribute
    fair = group_fairness_metrics(
        per_session, test_splits["test_input"], sequences, k=K, n_neg=N_NEG, seed=0
    )
    
    # Save results to dedicated cfretrieval files
    grp.to_csv(paths.RESULTS_DIR / "llm_cfretrieval_seed0_per_group.csv", index=False)
    fair.to_csv(paths.RESULTS_DIR / "llm_cfretrieval_seed0_fairness.csv", index=False)
    pd.DataFrame([{"model": "llm_cfretrieval", "seed": 0, **metrics, "run_time_sec": duration}]).to_csv(
        paths.RESULTS_DIR / "llm_cfretrieval_seed0_metrics.csv", index=False
    )
    print("\nSaved results to results/llm_cfretrieval_seed0_*.csv")

if __name__ == '__main__':
    main()
