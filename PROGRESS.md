# Project Progress Report — What We Have Built So Far

**Project:** Fairness Auditing & Mitigation in Adaptive Learning Pathway Recommendation
**Course:** Human-Centred AI (HCAI), OvGU · **Group:** Akshat · Dhairithri · Veer · Harshit
**Dataset:** OULAD (in `anonymisedData/`)
**This document covers:** everything done from project start up to (and including) the tuned SASRec model. The LLM recommender (Phase 2.4) and Phases 4–7 are **not yet started**.

> Companion docs: **[PROJECT_PLAN.md](PROJECT_PLAN.md)** = the full roadmap & decisions; **[README.md](README.md)** = how to run. This file = a detailed record of *what we did, file by file, plus all findings*.

---

## 1. Status at a glance

| Phase | What | Status |
|-------|------|--------|
| 0 | Repo scaffold, deps, reproducibility | ✅ Done |
| 1 | Data pipeline (sequences, splits, EDA) | ✅ Done |
| 2.1 | Popularity baseline | ✅ Done |
| 2.2 | Collaborative Filtering (ALS) | ✅ Done |
| 2.3 | SASRec deep model (tuned) | ✅ Done |
| 2.4 | LLM recommender (GPT-4o-mini) | ⬜ Not started (needs `OPENAI_API_KEY`) |
| 3 | Measurement (accuracy + fairness) | ✅ Done |
| 4 | Mitigation (fair-loss + reranking) | ⬜ Not started |
| 5 | Experiments (5 seeds, stats) | ⬜ Not started |
| 6 | Reporting (trade-off plots) | ⬜ Not started |
| 7 | Paper & presentation | ⬜ Not started |

**3 of 4 recommenders built, tuned, and fully audited** for accuracy AND fairness. The entire measurement loop is reusable: one command audits any model.

---

## 2. Environment (verified working)

- **Python** 3.11.9
- **pandas** 2.3.3, **numpy** 2.4.5, **pyarrow** 23.0.1
- **implicit** 0.7.3 (CF — replaces LightFM, see §6.6)
- **torch** 2.5.1+cu121 — **CUDA works** on **NVIDIA RTX 3050 Laptop (4 GB)**, driver 566.07
- Not yet installed: `openai` (for Phase 2.4), `aif360` (we use a native equivalent, see §6.8)

---

## 3. Repository layout (current)

```
HCAI/
├── anonymisedData/                 # raw OULAD CSVs (untouched)
├── data/processed/                 # generated caches (gitignored)
│   ├── sequences.parquet           # one row per session: seq + protected attrs
│   ├── item_vocab.parquet          # id_site <-> item_idx + activity_type
│   └── splits.parquet              # leave-last-out train/val/test columns
├── src/
│   ├── utils/
│   │   ├── paths.py                # central file paths            (28 lines)
│   │   └── seeds.py                # set_seed() + 5 seeds          (33 lines)
│   ├── data/
│   │   ├── protected.py            # group defs, IMD binarization  (84 lines)
│   │   ├── build_sequences.py      # clickstream -> sequences      (184 lines)
│   │   ├── splits.py               # leave-last-out split          (48 lines)
│   │   └── eda_summary.py          # data sanity check             (50 lines)
│   ├── models/
│   │   ├── base.py                 # Recommender interface         (39 lines)
│   │   ├── popularity.py           # Popularity baseline           (51 lines)
│   │   ├── cf.py                   # ALS collaborative filtering   (100 lines)
│   │   └── sasrec.py               # SASRec + early stopping       (282 lines)
│   ├── eval/
│   │   ├── accuracy.py             # Recall/NDCG/MRR + harness      (78 lines)
│   │   └── fairness.py             # SPD/EOD/AOD + per-group gaps   (182 lines)
│   └── mitigation/                 # (empty — Phase 4)
├── experiments/
│   ├── run_popularity.py           # first end-to-end run          (44 lines)
│   └── run_audit.py                # full audit: any model         (88 lines)
├── results/                        # metrics + fairness CSVs (committed)
├── requirements.txt
├── PROJECT_PLAN.md                 # roadmap
├── README.md                       # how to run
└── PROGRESS.md                     # this file
```

---

## 4. Locked design decisions (the foundation)

These were decided before coding; everything depends on them.

| # | Decision | Choice |
|---|----------|--------|
| D1 | What is an "item"? | **`id_site`** (~6,268 VLE activities after filtering) |
| D2 | User/session unit | `(id_student, code_module, code_presentation)` triple |
| D3 | Compute | GPU (RTX 3050) |
| D4 | LLM design | Retrieve-then-rerank (CF/popularity → 50 candidates → GPT-4o-mini top-10) |
| D5 | Split | **Leave-last-out** per session (last = test, 2nd-last = val, rest = history) |
| D6 | Min sequence length | **≥ 3** interactions |
| D7 | IMD binarization | disadvantaged (0–40%) vs advantaged (50–100%); NaN = explicit "unknown" |
| D8 | LLM test size | fixed sample (~2,000 sessions), same for all models, cached |

---

## 5. The task, in one paragraph

OULAD is not natively a recommender dataset, so we framed it as **sequential next-activity recommendation**: each `(student, module, presentation)` is a *session*; its chronologically ordered VLE clicks form a *sequence* of `id_site` items; the model must **predict the next activity** the student opens. We then audit each model for fairness across **gender, age band, disability, and socioeconomic status (IMD)**.

---

## 6. What each file does (in detail)

### 6.1 `src/utils/paths.py`
Single source of truth for every file location (raw CSVs and generated parquet). Defines `ROOT` (project root, computed relative to the file), and constants like `STUDENT_VLE_CSV`, `SEQUENCES_PARQUET`, `RESULTS_DIR`. `ensure_dirs()` creates output folders. **Why it exists:** so no script hardcodes a path; everything imports from here.

### 6.2 `src/utils/seeds.py`
`set_seed(seed)` seeds Python `random`, NumPy, and (if installed) PyTorch — including `cudnn.deterministic = True` — for reproducibility. Exposes `SEEDS = (0, 1, 2, 3, 4)`, the five seeds all experiments will use (Phase 5). torch is imported lazily so the data phase works without it.

### 6.3 `src/data/protected.py` — protected-attribute logic
The single source of truth for how the 4 protected attributes become groups, so the pipeline and the fairness evaluator never disagree. Key pieces:
- `DISADVANTAGED_BANDS` — the IMD bands counted as disadvantaged (0–50%).
- `PROTECTED_ATTRS` — for each attribute, which value is "privileged" vs "unprivileged" for the AIF360-style metrics (e.g. gender M vs F, disability N vs Y, IMD advantaged vs disadvantaged).
- `binarize_imd()` — normalizes messy raw `imd_band` strings (e.g. `"90-100%"`, missing values) into `imd_binary ∈ {disadvantaged, advantaged, unknown}` + an `imd_unknown` flag. **Implements Decision D7** (NaN never silently dropped).
- `collapse_age_band()` — collapses the 3 age bands into binary `0-35` vs `35+` for SPD/EOD/AOD (the original 3-level band is kept for the per-group view).
- `add_group_columns()` — adds `imd_binary`, `imd_unknown`, `age_binary` to a dataframe.

### 6.4 `src/data/build_sequences.py` — the heavy lifter (Phase 1.1–1.2)
Turns the **450 MB / 10.6 M-row** clickstream into model-ready sequences. Steps:
1. `load_protected()` — reads `studentInfo.csv`, adds group columns.
2. `load_activity_types()` — reads `vle.csv` → `id_site → activity_type`.
3. `read_clickstream()` — reads `studentVle.csv` **in chunks** (default 2 M rows) with memory-frugal dtypes (ids `int32`, date `int16`), so 10.6 M rows fit in a few hundred MB.
4. `build_visit_sequences()` — sorts each session by `date`, **collapses consecutive same-site repeats** (a student clicking the same page repeatedly becomes one visit), produces an ordered `id_site` list per session.
5. Joins protected attributes; **drops sessions with < 3 visits (D6)** and reports the drop rate.
6. `build_item_vocab()` — assigns each surviving `id_site` a contiguous `item_idx` (index 0 reserved for padding), attaches `activity_type`.
7. Writes `sequences.parquet` + `item_vocab.parquet`.

**Run:** `python -m src.data.build_sequences`

### 6.5 `src/data/splits.py` — leave-last-out (Phase 1.4, Decision D5)
`make_splits()` adds, per session: `test_target` (last item), `test_input` (all but last), `val_target` (2nd-last), `val_input` / `train_history` (all but last two). Deterministic and **seed-independent** — only model init varies across seeds, never the data split. `load_splits()` loads sequences and attaches these columns, optionally caching to `splits.parquet`.

### 6.6 `src/data/eda_summary.py` — sanity check (Phase 1.5)
Prints session/student/item counts, sequence-length distribution, and **group sizes per protected attribute** (flagging any group < 5% as noisy). This is where we caught the tiny `age_band 55<=` group. **Run:** `python -m src.data.eda_summary`

### 6.7 `src/models/` — the recommenders
All share one interface in **`base.py`**: `fit(splits_df)` and `recommend(history, k, context) -> ranked item indices`. `Context` carries the session's module+presentation, because OULAD sites are presentation-specific (D1) — a model may only return items that exist in that presentation.

- **`popularity.py`** (Phase 2.1) — counts next-item frequency **within each presentation**, trained from `train_history` only (no leakage). Recommends the most popular unseen items. Also serves as the LLM's candidate generator.
- **`cf.py`** (Phase 2.2) — Collaborative Filtering via **`implicit` ALS** matrix factorization. Builds a sparse `session × item` matrix from training history; at inference recomputes the user vector on the fly from the provided history (`recalculate_user=True`) and restricts candidates to the session's presentation. *(See §6.6 note on LightFM substitution below.)*
- **`sasrec.py`** (Phase 2.3) — the centerpiece deep model. A causal (left-to-right) **2-block, 50-dim self-attention Transformer** (Kang & McAuley 2018 spec, from the slides). Highlights:
  - Sequences left-padded and **truncated to the last 200 items** (median session is ~216 clicks).
  - Trained with **BPR pairwise loss** + one sampled negative per position; the loss is a **swappable method** (`_loss`) so Phase 4's fair-loss can subclass without touching the training loop.
  - **Validation-based early stopping** (added during tuning, see §8): tracks Recall@10 on a held-out 4,000-session sample, keeps the best-val weights, stops after 5 epochs of no improvement. Plus **dropout 0.3** and **weight decay 1e-4**.
  - Inference scores the last-position representation against the presentation's candidate items.

### 6.8 `src/eval/` — measurement (Phase 3)
- **`accuracy.py`** — `Recall@K`, `NDCG@K`, `MRR` (single relevant target, leave-last-out). `evaluate_model()` runs any recommender over the test split and returns **both** the aggregate metrics **and a per-session dataframe** (hit flag + rank). With `keep_topk=True` it also stores the full top-K list + target, which the fairness module needs.
- **`fairness.py`** — two complementary views, both computed from the *same* predictions:
  - **View A — per-group accuracy + gap (RQ3):** Recall@K/NDCG/MRR for every value of every attribute, plus the **max−min gap**. Multi-group, the most interpretable signal.
  - **View B — SPD / EOD / AOD (AIF360 definitions):** frames recommendation as pointwise relevance classification using **sampled negatives** (1 true item + 100 sampled negatives per session; "predicted positive" = lands in top-K). From the per-group confusion counts it computes Statistical Parity Difference, Equal-Opportunity Difference, and Average-Odds Difference. **AIF360 isn't installed** (it's heavy/fragile on Py3.11), so these use the exact same formulas natively — swappable for the real library later with identical numbers.

### 6.9 `experiments/` — runners
- **`run_popularity.py`** — the first minimal end-to-end run (popularity accuracy only).
- **`run_audit.py`** — **the reusable template.** Trains a chosen model once, evaluates accuracy, then runs both fairness views on the *same* predictions, prints everything, and writes `results/<model>_{metrics,per_group,fairness}.csv`. **Run:** `python -m experiments.run_audit --model {popularity,cf,sasrec}`

---

## 7. How to reproduce everything

```bash
pip install -r requirements.txt                       # + torch cu121 (installed)
python -m src.data.build_sequences                    # -> sequences.parquet, item_vocab.parquet
python -m src.data.splits                             # -> splits.parquet
python -m src.data.eda_summary                        # sanity check
python -m experiments.run_audit --model popularity    # baseline
python -m experiments.run_audit --model cf            # collaborative filtering
python -m experiments.run_audit --model sasrec        # deep model (GPU)
```

---

## 8. The SASRec tuning story (a key result)

The **first** SASRec run (20 epochs, no early stopping) **overfit**: training loss collapsed to **0.013** (memorizing train) and test Recall@10 was only **3.85%** — *worse than CF*. This contradicted the RQ1 hypothesis, but the cause was overfitting, not the model.

**Fix:** added validation-based early stopping (patience 5) on a 4,000-session held-out Recall@10, dropout 0.2→0.3, weight decay 1e-4. The run now visibly shows the trade-off — training loss keeps dropping while **val Recall plateaus around 0.077 and wobbles** — and stops at the best epoch (22), restoring those weights.

| SASRec | Recall@10 | NDCG@10 | MRR |
|--------|-----------|---------|-----|
| Before (overfit, 20 ep) | 3.85% | 2.11% | 1.59% |
| **After (early stop @22)** | **4.06%** | **2.22%** | **1.67%** |

+5.5% relative, and the tiny `age 55<=` group's recall **doubled** (0.0147 → 0.0294) — regularization helped small groups generalize.

---

## 9. ALL FINDINGS SO FAR

### 9.1 Data findings (from the pipeline + EDA)
- **10,655,280** clickstream rows processed into **29,228** raw sessions → **28,761 kept** (only **1.6%** dropped for < 3 visits).
- **25,745** unique students, **6,268** items (VLE sites). Note: only ~29k of the 32,593 enrolments appear in the clickstream — **some enrolled students never clicked** (worth one line in the paper's data section).
- **Sequences are long:** median **216** clicks, mean **331**, max **2,947** → SASRec must truncate (we use last 200).
- **Group sizes** (fairness-relevant):
  - gender: M 55.7% / F 44.3%
  - age_band: 0-35 **69.9%** / 35-55 29.3% / **55<= only 0.7% (204 sessions) ← too small, noisy**
  - disability: N 90.4% / Y 9.6%
  - imd_binary: disadvantaged 51.8% / advantaged 44.6% / **unknown 3.6% (1,047, reported separately)**
- **Implication:** `age_band 55<=` is too small for stable fairness metrics → we use **`age_binary` (0-35 vs 35+)** for SPD/EOD/AOD.

### 9.2 Accuracy findings (RQ1) — Recall@10 over all 28,761 sessions
| Model | Recall@10 | NDCG@10 | MRR |
|-------|-----------|---------|-----|
| Popularity | 3.26% | 1.76% | 1.31% |
| **CF (ALS)** | **4.20%** | **2.37%** | **1.82%** |
| SASRec (tuned) | 4.06% | 2.22% | 1.67% |

- Absolute numbers are low **by nature** — predicting the exact next click out of ~6,268 items is hard. What matters for the RQs is the **relative** ranking and the fairness gaps.
- **CF beats both** the popularity floor (+29%) and tuned SASRec. **CF ≈ SASRec (4.20% vs 4.06%)** — they are **neck-and-neck**.
- **This challenges the RQ1 hypothesis** ("SASRec = best balance"). On OULAD's next-activity task, **global co-occurrence (CF) is about as strong as sequence-order modeling (SASRec)** — a legitimate, discussable result. The 5-seed runs + paired Wilcoxon test (Phase 5) will tell us whether the CF–SASRec gap is even statistically significant.

### 9.3 Fairness findings (RQ3) — consistent across all 3 models
Per-group **Recall@10 gap** (max−min; bigger = more unfair):

| Attribute | Popularity | CF | SASRec |
|-----------|-----------|-----|--------|
| gender | 0.0026 | 0.0010 | 0.0018 |
| age_band* | 0.0244 | 0.0296 | 0.0135 |
| disability | 0.0078 | 0.0055 | 0.0041 |
| imd | 0.0139 | 0.0201 | 0.0151 |

\* age_band gap is inflated by the tiny `55<=` group — treat with caution.

Robust patterns (hold on every model):
1. **Gender gap is the smallest** ✓ — consistent with the RQ3 hypothesis.
2. **Disadvantaged-IMD students get *higher* recall, not lower** (e.g. SASRec: disadvantaged 4.66% vs advantaged 3.45%, **EOD +0.012**). All three models currently **favour** disadvantaged-IMD and disabled students, because those groups follow more **predictable, popular** learning paths that are easier to predict.
3. age_band shows the largest *raw* gap, but it is a small-sample artifact of the 204-session `55<=` group, not a reliable signal.

**Important nuance for the paper:** the RQ3 hypothesis expected IMD to be the *worst-treated* group. So far the *opposite* holds for these models — IMD-disadvantaged students are *advantaged* by the recommender. The real test is whether the **LLM** (Phase 2.4) reverses this, since the hypothesis is that the LLM will be the least fair.

### 9.4 SPD / EOD / AOD (ideal = 0; sign shows favoured group)
All three metrics are small in magnitude (< 0.02) for all models, confirming the gaps above are modest. EOD for gender numerically equals the raw recall gap (a correctness check that the metric is wired right). The IMD EOD is consistently **positive** (favouring disadvantaged) across models.

### 9.5 Engineering / reproducibility findings (for the paper's methods + limitations)
- **LightFM does not build on Python 3.11** (`__LIGHTFM_SETUP__` error) → substituted **`implicit` ALS** (same implicit-CF task; ALS chosen over BPR because it supports `recalculate_user`, which leave-last-out needs). Documented in `cf.py`.
- **AIF360 not installed** (heavy/fragile on Py3.11) → SPD/EOD/AOD implemented natively with **identical formulas**; can be swapped for AIF360 later without changing numbers.
- The fairness module reuses the **exact predictions** the accuracy evaluator produced (via `keep_topk`), so accuracy and fairness can never disagree.

---

## 10. What's next (not yet done)

1. **Phase 2.4 — LLM recommender** (retrieve-then-rerank, GPT-4o-mini). Needs `OPENAI_API_KEY`; will cap to a fixed ~2,000-session sample (D8) with response caching. **This is the key test of the RQ1/RQ3 hypotheses about the LLM being most-accurate / least-fair.**
2. **Phase 4 — Mitigation:** Fix 1 (FairSR-style fair loss in SASRec, λ-sweep) + Fix 2 (post-hoc exposure reranking). The SASRec loss is already swappable for Fix 1.
3. **Phase 5 — Experiments:** all models × **5 seeds**, bootstrap CIs, paired Wilcoxon significance (RQ2 trade-off curves).
4. **Phase 6 — Reporting:** accuracy-vs-fairness trade-off plot + per-group bar charts.
5. **Phase 7 — Paper (due 10 July) + slides (16–17 July).**

---

## 11. One-line summary

We built a complete, reproducible **OULAD next-activity recommendation + fairness-audit pipeline** and evaluated **3 of 4 models**. Headline so far: **CF ≈ tuned SASRec (both ~4.1% Recall@10), both beating popularity**; fairness gaps are **small, with gender most equal and IMD-disadvantaged students currently *favoured*** — the inverse of the stated hypothesis, making the upcoming LLM and mitigation experiments the decisive ones.
