"""Phase 5: Multi-seed experiment runner and statistical aggregator.

Runs all models (Popularity, CF, SASRec, LLM) and mitigations (Fair loss, post-hoc reranking)
across the 5 canonical seeds (0, 1, 2, 3, 4).
Performs statistical aggregation (mean, std, confidence intervals) and paired
Wilcoxon signed-rank tests to answer RQ1 and RQ2.

Saves:
  - results/raw_experiment_runs.csv: metrics for every model and seed.
  - results/summary_table.csv: statistical summaries (mean, std, CI, Wilcoxon p-value).
"""
from __future__ import annotations

import argparse
import os
import time
import numpy as np
import pandas as pd
import scipy.stats as stats

from src.data.splits import load_splits
from src.eval.accuracy import evaluate_model
from src.eval.fairness import (
    attach_groups,
    group_fairness_metrics,
    per_group_accuracy,
)
from src.utils import paths
from src.utils.seeds import set_seed, SEEDS

K = 10
N_NEG = 100
LLM_DEFAULT_N = 500

MODELS = ["popularity", "cf", "sasrec", "llm"]
LAM_GRID = [0.1, 0.5, 1.0, 2.0]
ALPHA_GRID = [0.1, 0.3, 0.5, 0.7, 1.0]

# Retrieve API key
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4",
                        help="Comma-separated list of seeds to run (default: 0,1,2,3,4)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test run: runs only popularity and CF for 2 seeds")
    parser.add_argument("--n_sessions", type=int, default=LLM_DEFAULT_N,
                        help="Number of sessions to evaluate for LLM (default: 500)")
    return parser.parse_args()


def audit_model(model, splits, sequences, seed):
    """Evaluate accuracy and fairness, return flattened metrics dictionary."""
    metrics, per_session = evaluate_model(
        model, splits, k=K, show_progress=False, keep_topk=True,
        extended_metrics=True, splits_df=splits
    )
    per_session = attach_groups(per_session, sequences)
    
    # View A: Gaps
    grp = per_group_accuracy(per_session, k=K)
    gaps = {}
    for attr in ["gender", "age_band", "disability", "imd_binary"]:
        val = grp.loc[
            (grp["attribute"] == attr) & (grp["group_value"] == "__GAP(max-min)__"),
            "recall"
        ]
        if len(val):
            gaps[f"{attr}_recall_gap"] = float(val.iloc[0])

    # View B: AIF360-style SPD/EOD/AOD
    fair = group_fairness_metrics(
        per_session, splits["test_input"], sequences, k=K, n_neg=N_NEG, seed=seed
    )
    fair_metrics = {}
    for _, row in fair.iterrows():
        attr = row["attribute"]
        fair_metrics[f"{attr}_SPD"] = float(row["SPD"])
        fair_metrics[f"{attr}_EOD"] = float(row["EOD"])
        fair_metrics[f"{attr}_AOD"] = float(row["AOD"])

    # Combine all
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
    return res


def run_experiment_grid(run_seeds, quick_mode, n_sessions_llm):
    """Run the experiments and save raw logs."""
    from src.models.popularity import PopularityRecommender
    from src.models.cf import CFRecommender
    
    # Load data splits
    splits = load_splits(write=False)
    sequences = pd.read_parquet(paths.SEQUENCES_PARQUET)
    
    raw_results = []
    output_csv = paths.RESULTS_DIR / "raw_experiment_runs.csv"
    
    # If the file already exists, load it to allow resuming
    if output_csv.exists():
        try:
            existing_df = pd.read_csv(output_csv)
            raw_results = existing_df.to_dict(orient="records")
            print(f"Loaded {len(raw_results)} existing runs from {output_csv}")
        except Exception as e:
            print(f"Error reading existing raw results: {e}. Starting fresh.")

    def run_and_log(model, splits_df, seed):
        # Check if already run
        for r in raw_results:
            if r["model"] == model.name and r["seed"] == seed:
                print(f"  [Skip] {model.name} for seed {seed} (already ran)")
                return r

        print(f"  [Run] {model.name} for seed {seed} ...", flush=True)
        t0 = time.time()
        model.fit(splits_df)
        res = audit_model(model, splits_df, sequences, seed)
        res["run_time_sec"] = time.time() - t0
        raw_results.append(res)
        
        # Save incrementally
        pd.DataFrame(raw_results).to_csv(output_csv, index=False)
        return res

    for seed in run_seeds:
        print(f"\n==================== SEED {seed} ====================", flush=True)
        
        # 1. Popularity
        set_seed(seed)
        pop_model = PopularityRecommender()
        run_and_log(pop_model, splits, seed)

        # 2. Collaborative Filtering (ALS)
        set_seed(seed)
        cf_model = CFRecommender(seed=seed)
        run_and_log(cf_model, splits, seed)

        if quick_mode:
            continue

        # 3. Base SASRec (needed as base for reranking and acts as lam=0.0 / alpha=0.0)
        from src.models.sasrec import SASRecRecommender
        set_seed(seed)
        sasrec_base = SASRecRecommender(seed=seed)
        # We need to save reference to trained base to reuse it for post-hoc reranking
        print(f"  [Run] sasrec (base) for seed {seed} ...", flush=True)
        t0 = time.time()
        sasrec_base.fit(splits)
        res_base = audit_model(sasrec_base, splits, sequences, seed)
        res_base["run_time_sec"] = time.time() - t0
        
        # Log base sasrec
        if not any(r["model"] == "sasrec" and r["seed"] == seed for r in raw_results):
            raw_results.append(res_base)
            pd.DataFrame(raw_results).to_csv(output_csv, index=False)

        # Log duplicate keys for sweeps at 0.0 value
        # Fair loss lam=0.0 is identical to base SASRec
        res_lam0 = res_base.copy()
        res_lam0["model"] = "sasrec_fair_lam0.0"
        if not any(r["model"] == "sasrec_fair_lam0.0" and r["seed"] == seed for r in raw_results):
            raw_results.append(res_lam0)
            pd.DataFrame(raw_results).to_csv(output_csv, index=False)

        # Rerank alpha=0.0 is identical to base SASRec
        res_a0 = res_base.copy()
        res_a0["model"] = "sasrec_rerank_a0.0"
        if not any(r["model"] == "sasrec_rerank_a0.0" and r["seed"] == seed for r in raw_results):
            raw_results.append(res_a0)
            pd.DataFrame(raw_results).to_csv(output_csv, index=False)

        # 4. Fair Loss SASRec (Sweeping lam)
        from src.mitigation.fair_loss import FairSASRecRecommender
        for lam in LAM_GRID:
            set_seed(seed)
            fair_model = FairSASRecRecommender(lam=lam, seed=seed)
            run_and_log(fair_model, splits, seed)

        # 5. Reranked SASRec (Sweeping alpha - very fast since base is fitted)
        from src.mitigation.rerank import RerankingRecommender
        for alpha in ALPHA_GRID:
            set_seed(seed)
            rerank_model = RerankingRecommender(base=sasrec_base, alpha=alpha)
            # fit just builds affinity tables
            run_and_log(rerank_model, splits, seed)

        # 6. LLM Recommender (GPT/DeepSeek)
        if not DEEPSEEK_API_KEY:
            print("  [Warning] DEEPSEEK_API_KEY not set. Skipping live LLM evaluations for this seed.")
            # For seed 0, check if we have llm results already committed in results/llm_metrics.csv
            # or if we can populate it from results/llm_*.csv.
            if seed == 0:
                print("  Attempting to load committed seed 0 LLM results...")
                try:
                    llm_m = pd.read_csv(paths.RESULTS_DIR / "llm_metrics.csv")
                    llm_f = pd.read_csv(paths.RESULTS_DIR / "llm_fairness.csv")
                    llm_g = pd.read_csv(paths.RESULTS_DIR / "llm_per_group.csv")
                    
                    llm_res = {
                        "model": "llm",
                        "seed": 0,
                        "Recall@10": float(llm_m.iloc[0]["Recall@10"]),
                        "NDCG@10": float(llm_m.iloc[0]["NDCG@10"]),
                        "MRR": float(llm_m.iloc[0]["MRR"]),
                    }
                    # Gaps
                    for attr in ["gender", "age_band", "disability", "imd_binary"]:
                        val = llm_g.loc[
                            (llm_g["attribute"] == attr) & (llm_g["group_value"] == "__GAP(max-min)__"),
                            "recall"
                        ]
                        if len(val):
                            llm_res[f"{attr}_recall_gap"] = float(val.iloc[0])
                    # Fairness
                    for _, row in llm_f.iterrows():
                        attr = row["attribute"]
                        llm_res[f"{attr}_SPD"] = float(row["SPD"])
                        llm_res[f"{attr}_EOD"] = float(row["EOD"])
                        llm_res[f"{attr}_AOD"] = float(row["AOD"])
                    
                    if not any(r["model"] == "llm" and r["seed"] == 0 for r in raw_results):
                        raw_results.append(llm_res)
                        pd.DataFrame(raw_results).to_csv(output_csv, index=False)
                        print("  Successfully loaded committed seed 0 LLM results.")
                except Exception as e:
                    print(f"  Could not load committed LLM results: {e}")
        else:
            from src.models.llm import LLMRecommender
            set_seed(seed)
            llm_model = LLMRecommender(api_key=DEEPSEEK_API_KEY, seed=seed)
            # For LLM, we sample the evaluation splits (Decision D8)
            test_splits = splits.sample(n=min(n_sessions_llm, len(splits)), random_state=seed).reset_index(drop=True)
            run_and_log(llm_model, test_splits, seed)

    print(f"\nCompleted raw runs! Output saved to {output_csv}", flush=True)
    return pd.DataFrame(raw_results)


def compute_statistics(df):
    """Compute mean, std, 95% Confidence Intervals, and Wilcoxon tests."""
    print("\nComputing statistical summaries...", flush=True)
    models = df["model"].unique()
    
    summary_records = []
    metrics_to_agg = ["Recall@10", "NDCG@10", "MRR", "Coverage", "Diversity_ILD", "Novelty", "imd_binary_recall_gap", "imd_SPD", "imd_EOD", "imd_AOD"]
    
    # 1. Standard aggregations (mean, std, 95% CI)
    for model_name in models:
        model_df = df[df["model"] == model_name]
        n_seeds = len(model_df)
        
        record = {"model": model_name, "n_seeds": n_seeds}
        
        for metric in metrics_to_agg:
            if metric not in model_df.columns:
                continue
            vals = model_df[metric].dropna().to_numpy()
            if len(vals) == 0:
                continue
                
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            
            # 95% CI using t-distribution
            if len(vals) > 1:
                se = std / np.sqrt(len(vals))
                ci_margin = stats.t.ppf(0.975, df=len(vals)-1) * se
                ci_lower = mean - ci_margin
                ci_upper = mean + ci_margin
            else:
                ci_lower = mean
                ci_upper = mean
                
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
            record[f"{metric}_ci_lower"] = ci_lower
            record[f"{metric}_ci_upper"] = ci_upper
            
        summary_records.append(record)
        
    summary_df = pd.DataFrame(summary_records)
    
    # 2. Paired Wilcoxon Signed-Rank tests
    # Wilcoxon requires matched pairs across seeds. We can only compute it if we have multiple seeds.
    print("\nRunning paired Wilcoxon signed-rank significance tests...")
    
    def run_wilcoxon(model_a, model_b, metric):
        a_df = df[df["model"] == model_a].sort_values("seed")
        b_df = df[df["model"] == model_b].sort_values("seed")
        
        # Align seeds
        merged = pd.merge(a_df, b_df, on="seed", suffixes=("_a", "_b"))
        a_vals = merged[f"{metric}_a"].to_numpy()
        b_vals = merged[f"{metric}_b"].to_numpy()
        
        if len(a_vals) < 3:
            print(f"  [Wilcoxon Skip] {model_a} vs {model_b} on {metric}: Insufficient/mismatched data")
            return None, None
            
        # check if differences are all zero
        diff = a_vals - b_vals
        if np.all(diff == 0):
            return 0.0, 1.0
            
        stat, pval = stats.wilcoxon(a_vals, b_vals)
        return float(stat), float(pval)

    # RQ1: CF vs SASRec (does sequence order beat matrix factorization?)
    cf_vs_sas_stat, cf_vs_sas_p = run_wilcoxon("cf", "sasrec", "Recall@10")
    if cf_vs_sas_p is not None:
        print(f"  RQ1: CF vs SASRec Recall@10 p-value = {cf_vs_sas_p:.4f} (stat={cf_vs_sas_stat:.2f})")

    # RQ2: SASRec vs FairSASRec (lam=1.0)
    sas_vs_fair_stat, sas_vs_fair_p = run_wilcoxon("sasrec", "sasrec_fair_lam1.0", "Recall@10")
    sas_vs_fair_eod_stat, sas_vs_fair_eod_p = run_wilcoxon("sasrec", "sasrec_fair_lam1.0", "imd_EOD")
    if sas_vs_fair_p is not None:
        print(f"  RQ2 (Fair loss lam=1.0): SASRec vs FairSASRec Recall@10 p-value = {sas_vs_fair_p:.4f}")
        print(f"  RQ2 (Fair loss lam=1.0): SASRec vs FairSASRec IMD EOD p-value = {sas_vs_fair_eod_p:.4f}")

    # RQ2: SASRec vs Reranked SASRec (alpha=0.7)
    sas_vs_rerank_stat, sas_vs_rerank_p = run_wilcoxon("sasrec", "sasrec_rerank_a0.7", "Recall@10")
    sas_vs_rerank_eod_stat, sas_vs_rerank_eod_p = run_wilcoxon("sasrec", "sasrec_rerank_a0.7", "imd_EOD")
    if sas_vs_rerank_p is not None:
        print(f"  RQ2 (Rerank alpha=0.7): SASRec vs Reranked Recall@10 p-value = {sas_vs_rerank_p:.4f}")
        print(f"  RQ2 (Rerank alpha=0.7): SASRec vs Reranked IMD EOD p-value = {sas_vs_rerank_eod_p:.4f}")

    # Attach p-values as properties to the summary df or save as a separate metadata file
    stats_data = {
        "comparison": ["cf_vs_sasrec", "sasrec_vs_fair_lam1.0", "sasrec_vs_fair_lam1.0_eod", "sasrec_vs_rerank_a0.7", "sasrec_vs_rerank_a0.7_eod"],
        "metric": ["Recall@10", "Recall@10", "imd_EOD", "Recall@10", "imd_EOD"],
        "wilcoxon_stat": [cf_vs_sas_stat, sas_vs_fair_stat, sas_vs_fair_eod_stat, sas_vs_rerank_stat, sas_vs_rerank_eod_stat],
        "p_value": [cf_vs_sas_p, sas_vs_fair_p, sas_vs_fair_eod_p, sas_vs_rerank_p, sas_vs_rerank_eod_p]
    }
    stats_df = pd.DataFrame(stats_data)
    
    summary_csv = paths.RESULTS_DIR / "summary_table.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved statistical summaries to {summary_csv}")
    
    stats_csv = paths.RESULTS_DIR / "wilcoxon_significance.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"Saved significance tests to {stats_csv}")


def main():
    args = parse_args()
    paths.ensure_dirs()
    
    if args.quick:
        run_seeds = [0, 1]
        print(f"Running in QUICK mode on seeds {run_seeds} for baselines only.")
        df = run_experiment_grid(run_seeds, quick_mode=True, n_sessions_llm=args.n_sessions)
    else:
        run_seeds = [int(s) for s in args.seeds.split(",")]
        print(f"Running experiments on seeds: {run_seeds}")
        df = run_experiment_grid(run_seeds, quick_mode=False, n_sessions_llm=args.n_sessions)
        
    compute_statistics(df)
    print("\nPhase 5 execution complete.", flush=True)


if __name__ == "__main__":
    main()
