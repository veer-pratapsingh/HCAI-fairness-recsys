# Project Progress Report

**Project:** Fairness Auditing & Mitigation in Adaptive Learning Pathway Recommendation
**Course:** Human-Centred AI (HCAI), OvGU · **Group:** Akshat · Dhairithri · Veer · Harshit
**Dataset:** OULAD (in `anonymisedData/`)
**This document covers:** everything done through Phase 4 (both mitigation fixes running). Phase 5 (multi-seed), 6 (plots), and 7 (paper) are not yet started.

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
| 4.1 | Fix 1 — Fair training loss (FairSASRec) | Running (lam sweep in progress, seed 0) |
| 4.2 | Fix 2 — Post-hoc reranking | Running (alpha sweep in progress, seed 0) |
| 5 | Experiments (5 seeds, stats) | Not started |
| 6 | Reporting (trade-off plots) | Not started |
| 7 | Paper & presentation | Not started |

**All 4 recommenders built and fully audited.** Both mitigation fixes are implemented and currently sweeping their hyperparameter grids (seed 0).

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
│   └── sasrec_fair_lamX_* / sasrec_rerank_aX_* (being written now)
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

### 9.2 Accuracy findings (RQ1)

| Model | Recall@10 | NDCG@10 | MRR | Test set |
|-------|-----------|---------|-----|----------|
| Popularity | 3.26% | 1.76% | 1.31% | 28,761 sessions |
| CF (ALS) | **4.20%** | **2.37%** | **1.82%** | 28,761 sessions |
| SASRec (tuned) | 4.06% | 2.22% | 1.67% | 28,761 sessions |
| LLM (DeepSeek) | 2.40% | 0.83% | 0.38% | 500 sampled sessions |

Key findings:
- CF is the best single model, narrowly ahead of SASRec (4.20% vs 4.06% -- likely not significant; Phase 5 Wilcoxon will confirm)
- LLM underperforms all classical models. Retrieve-then-rerank with activity-type labels does not give the model enough signal to distinguish 20 similar candidates. This contradicts the hypothesis that LLM = most accurate.
- Absolute recall numbers are low by nature -- predicting the exact next click out of ~6,268 items is hard. Relative differences and fairness gaps are the meaningful quantities.

### 9.3 Fairness findings (RQ3) -- per-group Recall@10 gap (max-min)

| Attribute | Popularity | CF | SASRec | LLM |
|-----------|-----------|-----|--------|-----|
| gender | 0.0026 | 0.0010 | 0.0018 | 0.0038 |
| age_band* | 0.0244 | 0.0296 | 0.0135 | 0.0074 |
| disability | 0.0078 | 0.0055 | 0.0041 | 0.0025 |
| imd | 0.0139 | 0.0201 | 0.0151 | 0.0283** |

\* age gap inflated by tiny 55<= group (204 sessions).
\** LLM imd gap is inflated by 20 "unknown" sessions getting 0% recall.

Robust patterns (hold across all 4 models):
1. **Gender gap is consistently the smallest** -- confirms the RQ3 hypothesis.
2. **Disadvantaged-IMD students get higher recall, not lower** (e.g. SASRec: disadvantaged 4.66% vs advantaged 3.45%, EOD +0.012). All models currently favour disadvantaged-IMD and disabled students. These groups follow more predictable, popular learning paths that are easier to predict. This is the **inverse of the RQ3 hypothesis**.
3. LLM shows the smallest disability and age gaps but the largest gender gap among the four models.

### 9.4 SPD / EOD / AOD summary

All metrics are small in magnitude (< 0.02) for classical models, confirming fairness gaps are modest overall. The IMD EOD is consistently positive (favouring disadvantaged) across all 4 models. LLM fairness metrics are on par with classical models despite much lower accuracy.

### 9.5 Phase 4 design (mitigation)

**Fix 1 -- FairSASRec (fair training loss):**
- Loss = BPR + lam * |mean_pos_logit(privileged) - mean_pos_logit(unprivileged)|
- Group labels attached to training sequences during fit(); divergence computed per mini-batch
- lam grid: {0.0, 0.1, 0.5, 1.0, 2.0} -- each is one point on the RQ2 trade-off curve
- lam=0 exactly reproduces plain SASRec (verified: same training trajectory)

**Fix 2 -- Reranking (post-hoc score fusion):**
- Combined score = (1-alpha) * base_rank_score + alpha * group_affinity(student_group, item)
- Group affinity = normalised item frequency in the student's own group's training history
- Trains SASRec once; applies reranking at inference for each alpha -- no retraining needed
- alpha grid: {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}
- alpha=0 exactly reproduces plain SASRec

### 9.6 Engineering / reproducibility notes

- LightFM fails to build on Python 3.11 -> substituted `implicit` ALS (same task; supports `recalculate_user` for leave-last-out). Documented in `cf.py`.
- AIF360 not installed -> SPD/EOD/AOD implemented natively with identical formulas; swappable for AIF360 later without changing numbers.
- Fairness module reuses the exact predictions the accuracy evaluator produced (`keep_topk=True`), so accuracy and fairness can never disagree.
- All non-ASCII characters removed from Python source files (Windows cp1252 terminal compatibility).
- Group columns (gender, imd_binary, etc.) live directly in `splits.parquet` -- no extra merge needed in experiment scripts.

---

## 10. What's next

1. **Phase 4 (completing):** Wait for lam and alpha sweep results (seed 0). Read trade-off tables and characterise the accuracy-vs-fairness curve for RQ2.
2. **Phase 5:** Run all 4 models + both mitigation fixes across 5 seeds. Compute mean +/- std, bootstrap CIs, paired Wilcoxon tests for significance.
3. **Phase 6:** Trade-off plot (x = Recall@10, y = IMD recall gap); per-group bar charts for each model.
4. **Phase 7:** Paper (due 10 July) + slides (16-17 July).

---

## 11. One-line summary

All 4 recommenders are built and audited; both Phase 4 mitigation fixes are implemented and sweeping their hyperparameter grids. Headline so far: **CF ~= tuned SASRec (~4.1% Recall@10), LLM underperforms at 2.4%, and disadvantaged-IMD students are consistently favoured by all models** -- the inverse of the RQ3 hypothesis, making the mitigation experiments and the paper's discussion section the decisive outputs.
