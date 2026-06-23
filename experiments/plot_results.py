"""Phase 6: Reporting and Visualization.

Generates:
1. results/accuracy_vs_fairness_tradeoff.png: Accuracy (Recall@10) vs Fairness (IMD EOD).
2. results/rq3_attribute_recall_gaps.png: Bar chart comparing recall gaps across attributes.
3. results/mitigation_recall_bar.png: Bar chart showing mitigation effects.
"""
from __future__ import annotations

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

RESULTS_DIR = pd.io.common.Path("d:/HCAI/results")
RAW_RUNS_CSV = RESULTS_DIR / "raw_experiment_runs.csv"


def load_model_stats() -> tuple[pd.DataFrame, dict[str, dict]]:
    """Load raw runs and return aggregated stats per model, plus external LLMs."""
    if not RAW_RUNS_CSV.exists():
        raise FileNotFoundError(f"Missing {RAW_RUNS_CSV}")

    df = pd.read_csv(RAW_RUNS_CSV)
    
    # Compute mean and std for all models in raw runs
    agg = df.groupby('model').agg({
        'Recall@10': ['mean', 'std'],
        'imd_EOD': ['mean', 'std'],
        'gender_recall_gap': ['mean', 'std'],
        'age_band_recall_gap': ['mean', 'std'],
        'disability_recall_gap': ['mean', 'std'],
        'imd_binary_recall_gap': ['mean', 'std'],
    })
    
    # Flatten multi-index columns
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()

    # Load external LLM runs if available
    external_llms = {}
    
    # Gemini
    gemini_csv = RESULTS_DIR / "gemini_llm_runs.csv"
    if gemini_csv.exists():
        try:
            gdf = pd.read_csv(gemini_csv)
            if len(gdf) > 0:
                external_llms['llm_gemini'] = {
                    'Recall@10_mean': gdf['Recall@10'].mean(),
                    'Recall@10_std': gdf['Recall@10'].std() if len(gdf) > 1 else 0.0,
                    'imd_EOD_mean': gdf['imd_EOD'].mean(),
                    'imd_EOD_std': gdf['imd_EOD'].std() if len(gdf) > 1 else 0.0,
                    'gender_recall_gap_mean': gdf['gender_recall_gap'].mean(),
                    'age_band_recall_gap_mean': gdf['age_band_recall_gap'].mean(),
                    'disability_recall_gap_mean': gdf['disability_recall_gap'].mean(),
                    'imd_binary_recall_gap_mean': gdf['imd_binary_recall_gap'].mean(),
                }
        except Exception as e:
            print(f"Warning loading Gemini runs: {e}")

    # Mistral
    mistral_csv = RESULTS_DIR / "mistral_llm_runs.csv"
    if mistral_csv.exists():
        try:
            mdf = pd.read_csv(mistral_csv)
            if len(mdf) > 0:
                external_llms['llm_mistral'] = {
                    'Recall@10_mean': mdf['Recall@10'].mean(),
                    'Recall@10_std': mdf['Recall@10'].std() if len(mdf) > 1 else 0.0,
                    'imd_EOD_mean': mdf['imd_EOD'].mean(),
                    'imd_EOD_std': mdf['imd_EOD'].std() if len(mdf) > 1 else 0.0,
                    'gender_recall_gap_mean': mdf['gender_recall_gap'].mean(),
                    'age_band_recall_gap_mean': mdf['age_band_recall_gap'].mean(),
                    'disability_recall_gap_mean': mdf['disability_recall_gap'].mean(),
                    'imd_binary_recall_gap_mean': mdf['imd_binary_recall_gap'].mean(),
                }
        except Exception as e:
            print(f"Warning loading Mistral runs: {e}")

    # Nvidia
    nvidia_csv = RESULTS_DIR / "nvidia_llm_runs.csv"
    if nvidia_csv.exists():
        try:
            ndf = pd.read_csv(nvidia_csv)
            if len(ndf) > 0:
                external_llms['llm_nvidia'] = {
                    'Recall@10_mean': ndf['Recall@10'].mean(),
                    'Recall@10_std': ndf['Recall@10'].std() if len(ndf) > 1 else 0.0,
                    'imd_EOD_mean': ndf['imd_EOD'].mean(),
                    'imd_EOD_std': ndf['imd_EOD'].std() if len(ndf) > 1 else 0.0,
                    'gender_recall_gap_mean': ndf['gender_recall_gap'].mean(),
                    'age_band_recall_gap_mean': ndf['age_band_recall_gap'].mean(),
                    'disability_recall_gap_mean': ndf['disability_recall_gap'].mean(),
                    'imd_binary_recall_gap_mean': ndf['imd_binary_recall_gap'].mean(),
                }
        except Exception as e:
            print(f"Warning loading Nvidia runs: {e}")

    return agg, external_llms


def plot_tradeoff(agg: pd.DataFrame, ext_llms: dict):
    """Plot 1: Accuracy vs Fairness Tradeoff (Recall@10 vs IMD EOD)."""
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
                ax.annotate(f"λ={l}", (fl_x[i], fl_y[i]), textcoords="offset points", xytext=(0,7), ha='center', fontsize=9, color='darkblue')

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
                # Put label slightly below or side to avoid overlap
                ax.annotate(f"α={a}", (rr_x[i], rr_y[i]), textcoords="offset points", xytext=(0,-12), ha='center', fontsize=9, color='darkgreen')

    # 4. Plot LLM DeepSeek
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

    # 5. Plot Parallel LLMs
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
    ax.set_title('Accuracy-Fairness Trade-off Curve on OULAD Dataset')
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='lower left')

    # Formatting percent labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.1f}%'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))

    plt.tight_layout()
    out_path = RESULTS_DIR / 'accuracy_vs_fairness_tradeoff.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved tradeoff plot to {out_path}")


def plot_gaps(agg: pd.DataFrame):
    """Plot 2: RQ3 recall gaps comparison across attributes."""
    models_to_plot = ['popularity', 'cf', 'sasrec', 'llm']
    labels = ['Popularity', 'CF (ALS)', 'SASRec Base', 'LLM (DeepSeek)']
    
    attributes = ['gender', 'age_band', 'disability', 'imd_binary']
    attr_labels = ['Gender', 'Age Band', 'Disability', 'Socioeconomic (IMD)']
    
    # Gather means
    data = {m: [] for m in models_to_plot}
    for m in models_to_plot:
        row = agg[agg['model'] == m]
        if len(row):
            for attr in attributes:
                val = row[f"{attr}_recall_gap_mean"].values[0]
                data[m].append(val)
        else:
            data[m] = [0.0] * len(attributes)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    x = np.arange(len(attributes))
    width = 0.2
    
    colors = ['#888888', '#ff7f0e', '#d62728', '#9467bd']
    
    for i, m in enumerate(models_to_plot):
        ax.bar(x + (i - 1.5) * width, data[m], width, label=labels[i], color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(attr_labels)
    ax.set_ylabel('Recall@10 Gap (Max - Min group recall)')
    ax.set_title('Per-Attribute Recommendation Disparity (Recall@10 Gap)')
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc')
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))

    plt.tight_layout()
    out_path = RESULTS_DIR / 'rq3_attribute_recall_gaps.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved RQ3 recall gaps bar chart to {out_path}")


def plot_mitigation(agg: pd.DataFrame):
    """Plot 3: Comparison of SASRec Base, Reranking (alpha=0.7), and FairSASRec (lam=1.0)."""
    models = ['sasrec', 'sasrec_rerank_a0.7', 'sasrec_fair_lam1.0']
    model_labels = ['SASRec Base', 'Rerank (α=0.7)', 'Fair Loss (λ=1.0)']
    
    rec_means = []
    eod_means = []
    
    for m in models:
        row = agg[agg['model'] == m]
        if len(row):
            rec_means.append(row['Recall@10_mean'].values[0])
            eod_means.append(row['imd_EOD_mean'].values[0])
        else:
            rec_means.append(0.0)
            eod_means.append(0.0)

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)

    color_rec = '#1f77b4'
    color_eod = '#2ca02c'

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax1.bar(x - width/2, rec_means, width, label='Recall@10', color=color_rec, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Accuracy (Recall@10)', color=color_rec)
    ax1.tick_params(axis='y', labelcolor=color_rec)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val*100:.1f}%'))

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, eod_means, width, label='IMD EOD', color=color_eod, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Fairness Bias (IMD EOD)', color=color_eod)
    ax2.tick_params(axis='y', labelcolor=color_eod)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val*100:.1f}%'))

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels)
    ax1.set_title('Mitigation Effects on Recommendation Accuracy and Fairness Bias')

    fig.tight_layout()
    out_path = RESULTS_DIR / 'mitigation_recall_bar.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved mitigation comparison bar chart to {out_path}")


def main():
    try:
        agg, ext_llms = load_model_stats()
        plot_tradeoff(agg, ext_llms)
        plot_gaps(agg)
        plot_mitigation(agg)
        print("All Phase 6 plots generated successfully!")
    except Exception as e:
        print(f"Error plotting results: {e}")


if __name__ == "__main__":
    main()
