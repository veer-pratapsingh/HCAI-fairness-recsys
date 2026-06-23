"""Plotting tradeoff curves including the new hybrid mitigation points."""
from __future__ import annotations

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Add root directory to path to allow absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.plot_results import load_model_stats, RESULTS_DIR


def plot_tradeoff_hybrid(agg: pd.DataFrame, ext_llms: dict):
    """Plot Accuracy vs Fairness Tradeoff including Hybrid points."""
    # Set modern plotting style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.edgecolor': '#cccccc',
        'axes.linewidth': 0.8,
    })

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 1. Plot Base baselines (Popularity, CF, SASRec)
    baselines = {
        'popularity': ('Popularity', 'tab:gray', '*'),
        'cf': ('CF (ALS)', 'tab:orange', 'o'),
        'sasrec': ('SASRec Base', 'tab:red', 's'),
    }
    
    for m_id, (label, color, marker) in baselines.items():
        row = agg[agg['model'] == m_id]
        if len(row):
            x = row['Recall@10_mean'].values[0]
            y = row['imd_EOD_mean'].values[0]
            x_err = row['Recall@10_std'].values[0]
            y_err = row['imd_EOD_std'].values[0]
            ax.errorbar(
                x, y, xerr=x_err, yerr=y_err,
                fmt=marker, color=color, markersize=10, label=label,
                capsize=3, elinewidth=1, markeredgecolor='black', zorder=5
            )

    # 2. Plot FairLoss sweep (Fix 1)
    fl_lams = [0.0, 0.1, 0.5, 1.0, 2.0]
    fl_x, fl_y = [], []
    for l in fl_lams:
        row = agg[agg['model'] == f"sasrec_fair_lam{l}"]
        if len(row):
            fl_x.append(row['Recall@10_mean'].values[0])
            fl_y.append(row['imd_EOD_mean'].values[0])
            
    if fl_x:
        ax.plot(fl_x, fl_y, 'o--', color='darkblue', linewidth=1.5, alpha=0.8, label='Fix 1: Fair Loss Sweep')
        for i, l in enumerate(fl_lams):
            if i < len(fl_x):
                # Offset slightly differently for lambda=0.1 to avoid overlapping hybrid label
                xy_text = (0, 7) if l != 0.1 else (-15, 5)
                ax.annotate(f"λ={l}", (fl_x[i], fl_y[i]), textcoords="offset points", xytext=xy_text, ha='center', fontsize=9, color='darkblue')

    # 3. Plot Reranking sweep (Fix 2)
    rr_alphas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    rr_x, rr_y = [], []
    for a in rr_alphas:
        row = agg[agg['model'] == f"sasrec_rerank_a{a}"]
        if len(row):
            rr_x.append(row['Recall@10_mean'].values[0])
            rr_y.append(row['imd_EOD_mean'].values[0])
            
    if rr_x:
        ax.plot(rr_x, rr_y, 's--', color='darkgreen', linewidth=1.5, alpha=0.8, label='Fix 2: Reranking Sweep')
        for i, a in enumerate(rr_alphas):
            if i < len(rr_x):
                ax.annotate(f"α={a}", (rr_x[i], rr_y[i]), textcoords="offset points", xytext=(0,-12), ha='center', fontsize=9, color='darkgreen')

    # 4. Plot Hybrid Mitigation points (New)
    # Start the line from sasrec_fair_lam0.1 (which is alpha=0.0)
    hb_alphas = [0.0, 0.1, 0.3]
    hb_x, hb_y = [], []
    for a in hb_alphas:
        if a == 0.0:
            row = agg[agg['model'] == "sasrec_fair_lam0.1"]
        else:
            row = agg[agg['model'] == f"sasrec_hybrid_lam0.1_alpha{a}"]
        if len(row):
            hb_x.append(row['Recall@10_mean'].values[0])
            hb_y.append(row['imd_EOD_mean'].values[0])
            
    if hb_x:
        ax.plot(hb_x, hb_y, 'X:', color='darkviolet', linewidth=1.5, alpha=0.9, label='Hybrid: Fair Loss (λ=0.1) + Reranking')
        
        # Plot error bars for the non-zero alpha hybrid points
        for i, a in enumerate(hb_alphas):
            if a > 0.0 and i < len(hb_x):
                # Retrieve row for getting standard deviations
                row = agg[agg['model'] == f"sasrec_hybrid_lam0.1_alpha{a}"]
                x_err = row['Recall@10_std'].values[0]
                y_err = row['imd_EOD_std'].values[0]
                ax.errorbar(
                    hb_x[i], hb_y[i], xerr=x_err, yerr=y_err,
                    fmt='X', color='darkviolet', markersize=9,
                    capsize=3, elinewidth=1, markeredgecolor='black', zorder=6
                )
                ax.annotate(f"α={a}", (hb_x[i], hb_y[i]), textcoords="offset points", xytext=(0,7), ha='center', fontsize=9, color='darkviolet', weight='bold')

    # 5. Plot LLM DeepSeek
    llm_row = agg[agg['model'] == 'llm']
    if len(llm_row):
        x = llm_row['Recall@10_mean'].values[0]
        y = llm_row['imd_EOD_mean'].values[0]
        x_err = llm_row['Recall@10_std'].values[0]
        y_err = llm_row['imd_EOD_std'].values[0]
        ax.errorbar(
            x, y, xerr=x_err, yerr=y_err,
            fmt='^', color='tab:purple', markersize=10, label='LLM (DeepSeek)',
            capsize=3, elinewidth=1, markeredgecolor='black', zorder=5
        )

    # 6. Plot Parallel LLMs
    ext_markers = {
        'llm_gemini': ('LLM (Gemini 2.0 Flash)', 'magenta', 'd'),
        'llm_mistral': ('LLM (Mistral 3.5)', 'cyan', 'p'),
        'llm_nvidia': ('LLM (Nemotron 3)', 'darkorange', 'h'),
    }
    
    for m_id, (label, color, marker) in ext_markers.items():
        if m_id in ext_llms:
            x = ext_llms[m_id]['Recall@10_mean']
            y = ext_llms[m_id]['imd_EOD_mean']
            x_err = ext_llms[m_id]['Recall@10_std']
            y_err = ext_llms[m_id]['imd_EOD_std']
            ax.errorbar(
                x, y, xerr=x_err, yerr=y_err,
                fmt=marker, color=color, markersize=9, label=label,
                capsize=3, elinewidth=1, markeredgecolor='black', zorder=4
            )

    ax.set_xlabel('Recommendation Accuracy (Recall@10)')
    ax.set_ylabel('Fairness Discrepancy (IMD Equal Opportunity Difference - EOD)')
    ax.set_title('Accuracy-Fairness Trade-off Curve including Hybrid Mitigation')
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='lower left')

    # Formatting percent labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))

    plt.tight_layout()
    out_path = RESULTS_DIR / 'accuracy_vs_fairness_tradeoff_hybrid.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved hybrid tradeoff plot to {out_path}")


def main():
    try:
        agg, ext_llms = load_model_stats()
        plot_tradeoff_hybrid(agg, ext_llms)
        print("Hybrid tradeoff plot generated successfully!")
    except Exception as e:
        print(f"Error plotting hybrid results: {e}")


if __name__ == "__main__":
    main()
