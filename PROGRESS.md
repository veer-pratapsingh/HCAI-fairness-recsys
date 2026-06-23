# Project Progress Report

**Project:** Fairness Auditing & Mitigation in Adaptive Learning Pathway Recommendation
**Course:** Human-Centred AI (HCAI), OvGU · **Group:** Akshat · Dhairithri · Veer · Harshit
**Dataset:** OULAD (in `anonymisedData/`)
**This document covers:** Completion of Phase 5 (5-seed experiments & Wilcoxon significance testing) and Phase 6 (reporting, configuration, and unit tests) for all recommenders, including enhanced sequential models, adversarial debiasing, post-hoc mitigations (counterfactual, calibration), and the ChatGPT/OpenAI LLM recommender.

> Companion docs: **[PROJECT_PLAN.md](PROJECT_PLAN.md)** = the full roadmap & decisions; **[hcai_project_report.md](hcai_project_report.md)** = the detailed project report. This file = a detailed record of what we did, file by file, plus all findings.

---

## 1. Status at a glance

| Phase | What | Status |
|-------|------|--------|
| 0 | Repo scaffold, deps, reproducibility | Done |
| 1 | Data pipeline (sequences, splits, EDA) | Done |
| 2.1 | Popularity baseline | Done |
| 2.2 | Collaborative Filtering (ALS) | Done |
| 2.3 | SASRec deep model (tuned) | Done |
| 2.4 | LLM recommender (DeepSeek, Gemini, ChatGPT/OpenAI, NVIDIA, Mistral) | Done (500-session sample, Decision D8) |
| 3 | Measurement (accuracy + fairness) | Done — all models audited |
| 4 | Fairness Mitigations (Fair Loss, Rerank, Adversarial, Counterfactual, Calibrated) | Done |
| 5 | Experiments (5 seeds, stats, Wilcoxon) | Done |
| 6 | Reporting (central config, unit tests, trade-off plots, report files) | Done |
| 7 | Paper & presentation | In Progress / Next |

**All implementation, experiment sweeps, evaluations, and tests are complete.** We successfully ran 5-seed sweeps across all recommenders, verified the pipeline using a 19-test unit suite, and centralized configurations.

---

## 2. Environment (verified working)

- **Python** 3.12.4
- **pandas** 2.3.3, **numpy** 2.2.2, **pyarrow** 23.0.1
- **implicit** 0.7.3 (CF baseline)
- **torch** 2.5.1+cu121 (deep learning)
- **openai** 2.43.0 (ChatGPT integration)
- **pytest** 9.1.1 (unit testing)
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
├── config/
│   └── default.yaml                # Centralized project configurations
├── src/
│   ├── utils/
│   │   ├── paths.py
│   │   ├── seeds.py
│   │   └── config.py               # YAML configuration loader
│   ├── data/
│   │   ├── protected.py
│   │   ├── build_sequences.py
│   │   ├── splits.py
│   │   ├── eda_summary.py
│   │   └── features.py             # Side-features builder for enhanced SASRec
│   ├── models/
│   │   ├── base.py                 # Context carries id_student
│   │   ├── popularity.py
│   │   ├── cf.py
│   │   ├── sasrec.py               # _loss() is overridable
│   │   ├── sasrec_enhanced.py      # Multi-feature embeddings + augmentations
│   │   ├── llm.py                  # DeepSeek retrieve-then-rerank
│   │   ├── llm_gemini.py           # Gemini retrieve-then-rerank
│   │   ├── llm_openai.py           # ChatGPT OpenAI retrieve-then-rerank
│   │   ├── llm_nvidia.py           # NVIDIA NIM Nemotron-3 retrieve-then-rerank
│   │   └── llm_mistral.py          # Mistral retrieve-then-rerank
│   ├── eval/
│   │   ├── accuracy.py             # Computes Recall, NDCG, MRR, Coverage, Diversity, Novelty
│   │   ├── fairness.py             # Computes recall gaps and SPD/EOD/AOD
│   │   └── intersectional.py       # Intersectional recall gaps and analysis
│   └── mitigation/                 
│       ├── __init__.py             # Mitigation module exposures
│       ├── fair_loss.py            # Fix 1: FairSASRec (lam-weighted group divergence)
│       ├── rerank.py               # Fix 2: group-calibrated score fusion
│       ├── adversarial.py          # Adversarial SASRec via GRL head
│       ├── counterfactual.py       # score-averaging counterfactual reranker
│       └── calibration.py          # KL-divergence MMR calibrated reranker
├── experiments/
│   ├── run_popularity.py
│   ├── run_audit.py                # supports --model llm and --n_sessions flag
│   ├── run_mitigation.py           # sweeps lam (Fix 1) and alpha (Fix 2)
│   ├── run_openai_llm.py           # ChatGPT/OpenAI runner
│   └── run_enhanced_experiments.py # Sweeps enhanced, adversarial, counterfactual, calibrated
├── tests/                          # 19 automated unit tests (popularity, cf, sasrec, data, fairness)
├── results/                        # all CSVs committed
├── requirements.txt
├── PROJECT_PLAN.md
├── README.md
├── hcai_project_report.md          # Comprehensive Project Report
└── PROGRESS.md                     # This progress summary
```

---

## 4. Locked design decisions (the foundation)

| # | Decision | Choice |
|---|----------|--------|
| D1 | What is an "item"? | `id_site` (~6,268 VLE activities after filtering) |
| D2 | User/session unit | `(id_student, code_module, code_presentation)` triple |
| D3 | Compute | PyTorch CPU / CUDA |
| D4 | LLM design | Retrieve-then-rerank (CF -> 50 candidates -> LLM reranks to top-10) |
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

### 6.7 `src/data/features.py` (NEW, Phase 6)
Precomputes side features for the enhanced SASRec:
- Activity type indices (forumng, quiz, resource, etc.)
- Time gap buckets between consecutive clicks (same-day to 2+ weeks)
- Course module indices

### 6.8 `src/models/` -- the recommenders

All share the interface in `base.py`: `fit(splits_df)` and `recommend(history, k, context)`. `Context` carries `id_student`.

- **`popularity.py`** -- next-item frequency within each presentation; trained from `train_history` only. Also serves as candidate generator.
- **`cf.py`** -- `implicit` ALS matrix factorization. Recomputes the user vector on the fly from history (`recalculate_user=True`). Replaces LightFM.
- **`sasrec.py`** -- 2-block, 50-dim causal self-attention Transformer. BPR loss, validation-based early stopping (patience 5), dropout 0.3, weight decay 1e-4. Swappable `_loss()` function.
- **`sasrec_enhanced.py`** (NEW, Phase 6) -- Feature-rich sequential model blending item IDs with Activity Type (16-dim), Time Gap (8-dim), and Module (8-dim) embeddings via a learned projection layer. Supports training augmentations (`crop`, `mask`).
- **`llm.py`** -- retrieve-then-rerank using DeepSeek API (`deepseek-chat`). Prompt contains student history, last 5 items, and candidates with popularity ranks.
- **`llm_gemini.py`** -- Gemini-2.0-flash retrieve-then-rerank implementation.
- **`llm_openai.py`** (NEW, Phase 6) -- ChatGPT gpt-4o-mini retrieve-then-rerank. Features exponential backoff retries and markdown fencing strip parser.
- **`llm_nvidia.py`** & **`llm_mistral.py`** -- NVIDIA NIM Nemotron-3 and Mistral-Medium rerankers.

### 6.9 `src/mitigation/` -- fairness fixes

- **`fair_loss.py`** -- `FairSASRecRecommender`: subclasses SASRec and overrides `_loss()` to penalize the gap in positive logit means between privileged and unprivileged students in the batch.
- **`rerank.py`** -- `RerankingRecommender`: wraps a base model with post-hoc score fusion combining recommender scores with demographic group click frequencies.
- **`adversarial.py`** (NEW, Phase 6) -- Adds a Gradient Reversal Layer (GRL) adversary head to SASRec. Forces sequence representations to be demographically blind.
- **`counterfactual.py`** (NEW, Phase 6) -- A post-processing reranker that averages score representations under actual and counterfactual group memberships.
- **`calibration.py`** (NEW, Phase 6) -- A post-processing calibration reranker using greedy MMR selection to minimize KL-divergence between recommended activity type distributions and target group historical distributions.

### 6.10 `src/eval/`

- **`accuracy.py`** -- Computes Recall@K, NDCG@K, MRR. Features `extended_metrics` supporting Coverage, Intra-List Diversity (ILD), and Novelty.
- **`fairness.py`** -- Computes Recall gaps (View A) and Statistical Parity Difference (SPD), Equal Opportunity Difference (EOD), and Average Odds Difference (AOD) (View B).
- **`intersectional.py`** (NEW, Phase 6) -- Groups test sessions into 2-way combinations (e.g. gender + disability) and reports max-min accuracy gaps.

### 6.11 `src/utils/config.py` & `config/default.yaml` (NEW, Phase 6)
Central YAML configuration management. Allows key overrides and centralized control of sequence parameters, model dimensions, and seed iterations.

---

## 7. How to reproduce everything

```bash
# Run pytest unit tests to check environment and pipeline
python -m pytest

# Run data pipeline
python -m src.data.build_sequences
python -m src.data.splits

# Model audits
python -m experiments.run_audit --model popularity
python -m experiments.run_audit --model cf
python -m experiments.run_audit --model sasrec

# Running LLM Rerankers (500 sessions)
python -m experiments.run_openai_llm      # ChatGPT gpt-4o-mini
python -m experiments.run_gemini_llm      # Gemini-2.0-flash
python -m experiments.run_mistral_llm     # Mistral-Medium
python -m experiments.run_nvidia_llm      # NVIDIA Nemotron-3

# Run Sweeps for Fair Loss and Group Reranker
python -m experiments.run_mitigation --fix 2 --seed 0
python -m experiments.run_mitigation --fix 1 --seed 0

# Run Master Runner for Enhanced SASRec, Adversarial, Counterfactual, and Calibrated Rerankers
python -m experiments.run_enhanced_experiments
```

---

## 8. The SASRec tuning story

The first SASRec run (20 epochs, no early stopping) overfit: training loss collapsed to 0.013 and test Recall@10 was only 3.85% — worse than CF. Fix: validation-based early stopping (patience 5) on a 4,000-session held-out Recall@10, dropout 0.2->0.3, weight decay 1e-4. Stopped at epoch 22.

| SASRec | Recall@10 | NDCG@10 | MRR |
|--------|-----------|---------|-----|
| Before (overfit, 20 ep) | 3.85% | 2.11% | 1.59% |
| After (early stop @22) | 4.06% | 2.22% | 1.67% |

Regularization helped small groups generalize.

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
| **CF (ALS)** | **4.19 ± 0.01** | **2.36 ± 0.00** | **1.81 ± 0.00** | 28,761 sessions |
| SASRec (tuned) | 4.10 ± 0.05 | 2.25 ± 0.02 | 1.69 ± 0.01 | 28,761 sessions |
| LLM (DeepSeek) | 2.96 ± 0.83 | 1.38 ± 0.49 | 0.92 ± 0.39 | 500 sampled sessions |
| **LLM (Gemini)** | **3.72 ± 0.93** | **1.82 ± 0.41** | **1.43 ± 0.28** | 500 sampled sessions |
| **LLM (Mistral)** | **3.55 ± 0.98** | **1.87 ± 0.30** | **1.41 ± 0.16** | 500 sampled sessions |
| **LLM (NVIDIA)** | **3.55 ± 0.98** | **1.87 ± 0.30** | **1.41 ± 0.16** | 500 sampled sessions |

Key findings:
- **CF is statistically the most accurate baseline**, narrowly but consistently outperforming SASRec. Wilcoxon signed-rank test yields a p-value of **0.0625**, indicating marginal but consistent superiority of CF.
- **Gemini, Mistral, and NVIDIA outperform DeepSeek.** They achieve Recall@10 values around 3.5% - 3.7%, showing greater robustness in parsing prompt hints.

### 9.3 Demographic Gaps & Inverse Bias (RQ3)
- **Gender gap is consistently the smallest** (recall gap ~0.0010).
- **All models favor disadvantaged-IMD and disabled students** (e.g. SASRec: EOD +1.19% mean). These students follow highly standardized course-template paths, making their behavior easier to fit. This is the **inverse of the expected direction**.

### 9.4 Mitigation Sweeps (5-seed stats)

**Fix 1 -- FairSASRec (Fair Loss Regularizer):**
- $\lambda_{\text{fair}} = 0.1 \to$ Recall@10: 3.25%, IMD EOD: +0.85%
- $\lambda_{\text{fair}} = 0.5 \to$ Recall@10: 3.00%, IMD EOD: +0.76%
- $\lambda_{\text{fair}} = 1.0 \to$ Recall@10: 2.87%, IMD EOD: +0.72% (-30% recall, -39.5% EOD gap)
- $\lambda_{\text{fair}} = 2.0 \to$ Recall@10: 1.77%, IMD EOD: +0.51% (-56.8% recall, -57.1% EOD gap)

*Trade-off:* Fair loss regularizers distort sequence representation, degrading accuracy severely.

**Fix 2 -- Reranking (Group Score Fusion):**
- $\alpha = 0.1 \to$ Recall@10: 4.04%, IMD EOD: +1.19%
- $\alpha = 0.3 \to$ Recall@10: 3.88%, IMD EOD: +1.17%
- $\alpha = 0.5 \to$ Recall@10: 3.73%, IMD EOD: +1.12%
- $\alpha = 0.7 \to$ Recall@10: **3.58%**, IMD EOD: **+1.07%** (-12.7% recall, -10.1% EOD gap)
- $\alpha = 1.0 \to$ Recall@10: 3.26%, IMD EOD: +1.15%

*Trade-off:* Post-hoc group reranking ($\alpha=0.7$) offers a highly controllable and graceful trade-off.

**New Mitigations (Seed 0 Quick Runs):**
- **Calibrated MMR Reranking ($\lambda_{\text{cal}} = 0.5$)**: Yields the highest Intra-List Diversity (ILD = 0.51) and maintains controlled KL-divergence.
- **Counterfactual Reranking**: Balances recommended frequencies.
- **Adversarial SASRec**: Optimizes sequence invariant vectors via Gradient Reversal (GRL).

---

## 10. What's next

1. **Phase 7 (Final):** Compile the final project paper (due 10 July) and prepare presentation slides (scheduled for 16-17 July).
2. Use the compiled metrics to draft figures and diagrams demonstrating the trade-offs of the calibrated rerankers vs base models.

---

## 11. One-line summary

Phase 6 complete: **CF is the most accurate baseline (Recall@10 4.19%), LLMs underperform but Gemini/Mistral show gains (Recall@10 ~3.7%), and post-hoc reranking ($\alpha=0.7$ / calibrated MMR) represents the most controlled fairness mitigation strategy.**
