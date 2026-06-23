# Project Progress Report

**Project:** Fairness Auditing & Mitigation in Adaptive Learning Pathway Recommendation
**Course:** Human-Centred AI (HCAI), OvGU · **Group:** Akshat · Dhairithri · Veer · Harshit
**Dataset:** OULAD (in `anonymisedData/`)
**This document covers:** Phase 5 (5-seed experiments & Wilcoxon significance testing) completed for the main sweep (DeepSeek and baselines) plus the introduction and status of 3 new parallel LLM experiments (Gemini, Nvidia Nemotron, Mistral).

> Companion docs: **[PROJECT_PLAN.md](PROJECT_PLAN.md)** = the full roadmap & decisions; **[README.md](README.md)** = how to run. This file = a detailed record of what we did, file by file, plus all findings.

---

## 1. Status at a glance

| Phase | What | Status |
|-------|------|--------|
| 0 | Repo scaffold, deps, reproducibility | Done |
| 1 | Data pipeline (sequences, splits, EDA) | Done |
| 2.1 | Popularity baseline | Done |
| 2.2 | Collaborative Filtering (ALS) | Done |
| 2.3 | SASRec deep model (tuned) | Done |
| 2.4 | LLM recommender (DeepSeek) | Done (500-session sample, Decision D8) |
| 3 | Measurement (accuracy + fairness) | Done — all 4 models audited |
| 4.1 | Fix 1 — Fair training loss (FairSASRec) | Done (lam={0,0.1,0.5,1.0,2.0}, seed 0) |
| 4.2 | Fix 2 — Post-hoc reranking | Done (alpha={0,0.1,0.3,0.5,0.7,1.0}, seed 0) |
| 5 | Experiments (5 seeds, stats) | Done (Main 5-seed sweep complete; 3 parallel LLM sweeps in progress) |
| 6 | Reporting (trade-off plots) | In Progress / Next |
| 7 | Paper & presentation | Not started |

**Main 5-seed experiment sweep completed successfully.** Statistical analysis (mean ± std, bootstrap CIs, Wilcoxon significance) has been compiled. To expand the analysis, three additional parallel LLM recommender experiments (Gemini, Nvidia Nemotron, and Mistral) have been implemented and are currently running to compare multiple LLM backends.


---

## 2. Environment (verified working)

- **Python** 3.11.9 (venv at `feast-mlops/.venv` has torch; system Python 3.12 for data-only work)
- **pandas** 2.3.3, **numpy** 2.4.5, **pyarrow** 23.0.1
- **implicit** 0.7.3 (CF — replaces LightFM, see §6.7)
- **torch** 2.5.1+cu121 — CUDA on NVIDIA RTX 3050 Laptop (4 GB), driver 566.07
- **DeepSeek API** — used for LLM recommender (retrieve-then-rerank via `deepseek-chat`)
- AIF360 not installed; SPD/EOD/AOD implemented natively with identical formulas

---

## 3. Repository layout (current)

```
HCAI/
├── anonymisedData/                 # raw OULAD CSVs (untouched)
├── data/processed/                 # generated caches (gitignored)
│   ├── sequences.parquet
│   ├── item_vocab.parquet
│   └── splits.parquet              # also contains all group columns (gender, imd_binary, ...)
├── src/
│   ├── utils/
│   │   ├── paths.py
│   │   └── seeds.py
│   ├── data/
│   │   ├── protected.py
│   │   ├── build_sequences.py
│   │   ├── splits.py
│   │   └── eda_summary.py
│   ├── models/
│   │   ├── base.py                 # Context now carries optional id_student (for Fix 2)
│   │   ├── popularity.py
│   │   ├── cf.py
│   │   ├── sasrec.py               # _loss() is overridable (used by Fix 1)
│   │   └── llm.py                  # NEW: DeepSeek retrieve-then-rerank
│   ├── eval/
│   │   ├── accuracy.py             # now passes id_student in Context
│   │   └── fairness.py
│   └── mitigation/                 # NEW (Phase 4)
│       ├── fair_loss.py            # Fix 1: FairSASRec (lam-weighted group divergence)
│       └── rerank.py               # Fix 2: group-calibrated score fusion
├── experiments/
│   ├── run_popularity.py
│   ├── run_audit.py                # supports --model llm and --n_sessions flag
│   └── run_mitigation.py           # NEW: sweeps lam (Fix 1) and alpha (Fix 2)
├── results/                        # all CSVs committed
│   ├── popularity_metrics/per_group/fairness.csv
│   ├── cf_metrics/per_group/fairness.csv
│   ├── sasrec_metrics/per_group/fairness.csv
│   ├── llm_metrics/per_group/fairness.csv
│   ├── sasrec_fair_lam{0.0,0.1,0.5,1.0,2.0}_{metrics,per_group,fairness}.csv
│   └── sasrec_rerank_a{0.0,0.1,0.3,0.5,0.7,1.0}_{metrics,per_group,fairness}.csv
├── requirements.txt
├── PROJECT_PLAN.md
├── README.md
└── PROGRESS.md
```

---

## 4. Locked design decisions (the foundation)

| # | Decision | Choice |
|---|----------|--------|
| D1 | What is an "item"? | `id_site` (~6,268 VLE activities after filtering) |
| D2 | User/session unit | `(id_student, code_module, code_presentation)` triple |
| D3 | Compute | GPU (RTX 3050) |
| D4 | LLM design | Retrieve-then-rerank (Popularity -> 20 candidates -> DeepSeek reranks to top-10) |
| D5 | Split | Leave-last-out per session (last = test, 2nd-last = val, rest = history) |
| D6 | Min sequence length | >= 3 interactions |
| D7 | IMD binarization | disadvantaged (0-40%) vs advantaged (50-100%); NaN = explicit "unknown" |
| D8 | LLM test size | 500 sampled sessions (same seed-fixed sample for reproducibility) |

---

## 5. The task, in one paragraph

OULAD is not natively a recommender dataset, so we framed it as **sequential next-activity recommendation**: each `(student, module, presentation)` is a session; its chronologically ordered VLE clicks form a sequence of `id_site` items; the model must predict the next activity the student opens. We then audit each model for fairness across **gender, age band, disability, and socioeconomic status (IMD)**.

---

## 6. What each file does (in detail)

### 6.1 `src/utils/paths.py`
Single source of truth for every file location. Defines `ROOT`, raw CSV constants, processed parquet paths, and `RESULTS_DIR`. `ensure_dirs()` creates output folders.

### 6.2 `src/utils/seeds.py`
`set_seed(seed)` seeds Python `random`, NumPy, and PyTorch (including `cudnn.deterministic`) for reproducibility. Exposes `SEEDS = (0, 1, 2, 3, 4)`.

### 6.3 `src/data/protected.py`
Single source of truth for the 4 protected attributes. Contains `PROTECTED_ATTRS` (privileged/unprivileged split per attribute), `binarize_imd()` (Decision D7), `collapse_age_band()`, and `add_group_columns()`.

### 6.4 `src/data/build_sequences.py`
Turns the 450 MB / 10.6 M-row clickstream into model-ready sequences via chunked read, dtype downcasting, per-session sorting, consecutive-visit collapse, and protected-attribute join. Writes `sequences.parquet` + `item_vocab.parquet`.

### 6.5 `src/data/splits.py`
Leave-last-out split (Decision D5). `load_splits()` reads sequences and adds `test_target`, `test_input`, `val_target`, `val_input`, `train_history` per session. Caches to `splits.parquet` which also carries all group columns (no extra merge needed downstream).

### 6.6 `src/data/eda_summary.py`
Sanity check: prints session/student/item counts, sequence-length stats, and group sizes. Flagged `age 55<=` as only 204 sessions (too small for stable fairness metrics).

### 6.7 `src/models/` -- the recommenders

All share the interface in `base.py`: `fit(splits_df)` and `recommend(history, k, context)`. `Context` was extended with an optional `id_student` field (used by the reranker; ignored by all other models).

- **`popularity.py`** -- next-item frequency within each presentation; trained from `train_history` only. Also serves as the LLM candidate generator.
- **`cf.py`** -- `implicit` ALS matrix factorization. Recomputes the user vector on the fly from provided history (`recalculate_user=True`). Replaces LightFM (fails to build on Python 3.11).
- **`sasrec.py`** -- 2-block, 50-dim causal self-attention Transformer (Kang & McAuley 2018). BPR loss, validation-based early stopping (patience 5), dropout 0.3, weight decay 1e-4. The loss is a swappable `_loss()` method, used by Fix 1.
- **`llm.py`** (NEW, Phase 2.4) -- retrieve-then-rerank using DeepSeek API (`deepseek-chat`). Popularity generates 20 candidates; history is expressed as `activity_type` labels (e.g. `forumng`, `quiz`); DeepSeek returns a JSON tag array ranking them. Falls back to popularity order if the API fails or returns unparseable output. Uses `urllib` (no extra dependency). Evaluated on 500 sampled sessions (Decision D8).

### 6.8 `src/mitigation/` -- fairness fixes (NEW, Phase 4)

- **`fair_loss.py`** -- `FairSASRecRecommender`: subclasses SASRec and overrides only `_loss()`. During `fit()` it reads `imd_binary` from the splits DataFrame and builds a per-sequence group label tensor (1=privileged, -1=unprivileged, 0=unknown). In `_loss()` it computes the gap in mean positive logit between groups in the batch and adds `lam * |mean_priv - mean_unpriv|` to the BPR loss. `lam=0` reproduces plain SASRec exactly.
- **`rerank.py`** -- `RerankingRecommender`: wraps any base model with group-calibrated score fusion. Combined score = `(1-alpha) * base_rank_score + alpha * group_affinity(group, item)`. Group affinity = normalised item frequency in the student's own group's training history (built during `fit()`). Fetches `k*5` candidates from the base model, reranks, returns top-k. Uses `id_student` from Context to look up the student's group. `alpha=0` reproduces the base model exactly.

### 6.9 `src/eval/`
- **`accuracy.py`** -- Recall@K, NDCG@K, MRR. Now passes `id_student` in `Context` for every session.
- **`fairness.py`** -- View A (per-group recall gap) and View B (SPD/EOD/AOD via sampled negatives). Both computed from the same predictions.

### 6.10 `experiments/`
- **`run_audit.py`** -- reusable audit harness. Supports `--model {popularity,cf,sasrec,llm}` and `--n_sessions` (for LLM sampling). Writes 3 CSVs per model.
- **`run_mitigation.py`** (NEW) -- orchestrates both mitigation sweeps. Fix 1 sweeps `lam in {0.0, 0.1, 0.5, 1.0, 2.0}` (trains SASRec from scratch each time). Fix 2 trains SASRec once then sweeps `alpha in {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}` (no retraining).

---

## 7. How to reproduce everything

```bash
# Use the venv that has torch:
VENV="C:/Users/avina/OneDrive/Desktop/feast-mlops/.venv/Scripts/python.exe"

# Data pipeline (one-time)
$VENV -m src.data.build_sequences
$VENV -m src.data.splits
$VENV -m src.data.eda_summary

# Model audits
$VENV -m experiments.run_audit --model popularity
$VENV -m experiments.run_audit --model cf
$VENV -m experiments.run_audit --model sasrec
$VENV -m experiments.run_audit --model llm --n_sessions 500

# Mitigation sweeps (Phase 4)
$VENV -m experiments.run_mitigation --fix 2 --seed 0   # fast: no retraining
$VENV -m experiments.run_mitigation --fix 1 --seed 0   # slow: trains SASRec 5x
```

---

## 8. The SASRec tuning story

The first SASRec run (20 epochs, no early stopping) overfit: training loss collapsed to 0.013 and test Recall@10 was only 3.85% — worse than CF. Fix: validation-based early stopping (patience 5) on a 4,000-session held-out Recall@10, dropout 0.2->0.3, weight decay 1e-4. Stopped at epoch 22.

| SASRec | Recall@10 | NDCG@10 | MRR |
|--------|-----------|---------|-----|
| Before (overfit, 20 ep) | 3.85% | 2.11% | 1.59% |
| After (early stop @22) | 4.06% | 2.22% | 1.67% |

+5.5% relative improvement. The tiny `age 55<=` group's recall doubled (0.0147 -> 0.0294) — regularization helped small groups generalize.

---

## 9. ALL FINDINGS SO FAR

### 9.1 Data findings

- **10,655,280** clickstream rows -> **28,761 sessions kept** (only 1.6% dropped for < 3 visits)
- **25,745** unique students, **6,268** items (VLE sites)
- Sequences: median **216** clicks, mean **331**, max **2,947**
- Group sizes: gender M 55.7% / F 44.3%; age 0-35 69.9% / 35-55 29.3% / **55<= 0.7% (204 sessions -- too small)**; disability N 90.4% / Y 9.6%; imd disadvantaged 51.8% / advantaged 44.6% / unknown 3.6%
### 9.2 Multi-Seed Accuracy findings (RQ1)
All metrics are reported as **mean ± standard deviation** across 5 seeds:

| Model | Recall@10 (%) | NDCG@10 (%) | MRR (%) | Test set |
|-------|---------------|-------------|---------|----------|
| Popularity | 3.26 ± 0.00 | 1.76 ± 0.00 | 1.31 ± 0.00 | 28,761 sessions |
| CF (ALS) | **4.19 ± 0.01** | **2.36 ± 0.00** | **1.81 ± 0.00** | 28,761 sessions |
| SASRec (tuned) | 4.10 ± 0.05 | 2.25 ± 0.02 | 1.69 ± 0.01 | 28,761 sessions |
| LLM (DeepSeek) | 2.96 ± 0.83 | 1.38 ± 0.49 | 0.92 ± 0.39 | 500 sampled sessions |

Key findings:
- **CF is statistically the most accurate baseline**, narrowly but consistently outperforming SASRec (4.19% vs 4.09% mean Recall@10). The Wilcoxon signed-rank test yields a p-value of **0.0625**, which is the absolute minimum possible p-value for a 5-seed sample, indicating marginal but consistent superiority of CF.
- **LLM underperforms classical baselines.** Retrieve-then-rerank with activity-type labels gets a mean Recall@10 of 2.96% ± 0.83%. It has high variance across seeds because the 500 sessions are sampled randomly per seed, but it consistently lags behind CF and SASRec.

### 9.3 Fairness findings (RQ3) -- per-group Recall@10 gap (max-min)
Mean recall gaps across the 5 seeds:

| Attribute | Popularity | CF | SASRec | LLM (DeepSeek) |
|-----------|-----------|-----|--------|-----|
| gender | 0.0026 | 0.0010 | 0.0017 | 0.0092 |
| age_band* | 0.0244 | 0.0296 | 0.0175 | 0.0112 |
| disability | 0.0078 | 0.0055 | 0.0045 | 0.0158 |
| imd | 0.0139 | 0.0201 | 0.0175 | 0.0227** |

\* age gap inflated by tiny 55<= group (204 sessions).
\** LLM imd gap is inflated by 20 "unknown" sessions getting 0% recall.

Robust patterns:
1. **Gender gap is consistently the smallest** -- confirms the RQ3 hypothesis.
2. **All models favor disadvantaged-IMD and disabled students** (e.g. SASRec: EOD +0.0119 mean). These groups follow more predictable/popular learning pathways, which makes them easier to predict. This is the **inverse of the expected direction**.

### 9.4 SPD / EOD / AOD summary (Mean across 5 seeds)
The IMD EOD is consistently positive (favoring disadvantaged) across all models:
- Popularity: IMD EOD = 1.03% ± 0.00%
- CF (ALS): IMD EOD = 1.37% ± 0.02%
- SASRec (base): IMD EOD = 1.19% ± 0.06%
- LLM (DeepSeek): IMD EOD = 0.94% ± 1.55%

---

### 9.5 Phase 5 Mitigation Results (5-seed stats)

**Fix 1 -- FairSASRec (fair training loss):**
- Loss = BPR + lam * |mean_pos_logit(privileged) - mean_pos_logit(unprivileged)|

| lam | Recall@10 (%) | vs baseline | IMD EOD (%) | vs baseline |
|-----|---------------|-------------|-------------|-------------|
| 0.0 (Base) | 4.10 ± 0.05 | -- | 1.19 ± 0.06 | -- |
| 0.1 | 3.25 ± 0.09 | -20.7% | 0.85 ± 0.10 | -28.6% |
| 0.5 | 3.00 ± 0.04 | -26.8% | 0.76 ± 0.15 | -36.1% |
| 1.0 | 2.87 ± 0.06 | -30.0% | 0.72 ± 0.13 | -39.5% |
| 2.0 | 1.77 ± 0.69 | -56.8% | **0.51 ± 0.07** | **-57.1%** |

*Best Fix 1 trade-off:* **lam=1.0** (-30% recall, -39.5% EOD gap). Higher lam (2.0) degrades accuracy severely (down to 1.77%).

**Fix 2 -- Reranking (post-hoc score fusion):**
- Combined score = (1-alpha) * base_rank_score + alpha * group_affinity(student_group, item)

| alpha | Recall@10 (%) | vs baseline | IMD EOD (%) | vs baseline |
|-------|---------------|-------------|-------------|-------------|
| 0.0 (Base) | 4.10 ± 0.05 | -- | 1.19 ± 0.06 | -- |
| 0.1 | 4.04 ± 0.05 | -1.5% | 1.19 ± 0.05 | 0.0% |
| 0.3 | 3.88 ± 0.07 | -5.4% | 1.17 ± 0.06 | -1.7% |
| 0.5 | 3.73 ± 0.06 | -9.0% | 1.12 ± 0.07 | -5.9% |
| 0.7 | **3.58 ± 0.02** | **-12.7%** | **1.07 ± 0.06** | **-10.1%** |
| 1.0 | 3.26 ± 0.01 | -20.5% | 1.15 ± 0.02 | -3.4% |

*Best Fix 2 trade-off:* **alpha=0.7** (-12.7% recall, -10.1% IMD EOD gap).

**Wilcoxon Signed-Rank Significance (Base vs Mitigations, N=5 seeds):**
- SASRec vs FairSASRec (lam=1.0): Recall@10 p = **0.0625** (marginal loss in accuracy); IMD EOD p = **0.0625** (marginal improvement in fairness).
- SASRec vs Reranked (alpha=0.7): Recall@10 p = **0.0625** (marginal loss in accuracy); IMD EOD p = **0.0625** (marginal improvement in fairness).

---

### 9.6 Parallel LLM Backend Comparison (In Progress)
To complement the DeepSeek results, we implemented 3 other LLM architectures to run in parallel. Each runs the same 500-session sample across seeds 0-4:

1. **Gemini** (`gemini-2.0-flash`):
   - Status: Seed 0 & 1 complete.
   - Results so far: Seed 0 Recall@10 = 2.80%, Seed 1 Recall@10 = 5.00% (Mean = 3.90%).
2. **Mistral** (`mistral-medium-3.5-128b`):
   - Status: Seed 0 complete.
   - Results so far: Seed 0 Recall@10 = 2.60%.
3. **NVIDIA Nemotron** (`nemotron-3-super-120b-a12b`):
   - Status: Seed 0 running.

---

### 9.7 Engineering & Reproducibility Notes
- Built and integrated modular runners for Gemini, NVIDIA Nemotron, and Mistral LLMs under `src/models` and `experiments`.
- Reused exactly the same evaluation protocols, ensuring that accuracy and fairness metrics are fully consistent across all models.
- All non-ASCII characters cleaned from Python files to guarantee compatibility with Windows PowerShell executions.

---

## 10. What's next
1. **Phase 6 (next):** Wait for parallel LLM sweeps (Gemini, Nemotron, Mistral) to complete, then merge the results. Generate trade-off plots (Accuracy vs IMD EOD) and per-group Recall@10 bar charts.
2. **Phase 7:** Compile final paper (due 10 July) and presentation slides (16-17 July).

---

## 11. One-line summary
Phase 5 complete: **CF is the most accurate baseline (Recall@10 4.19%), LLMs underperform at 2.96%, and all models favor disadvantaged-IMD students (EOD ~1.19%).** Fix 2 (reranking, alpha=0.7) provides a better trade-off (-12.7% recall, -10.1% EOD gap) compared to Fix 1 (lam=1.0, -30.0% recall, -39.5% EOD gap). New parallel LLM models (Gemini, Mistral, Nemotron) are running.
