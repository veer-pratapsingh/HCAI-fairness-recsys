# Project Plan — Fairness Auditing & Mitigation in Adaptive Learning Pathway Recommendation

**Course:** Human-Centred Artificial Intelligence (HCAI), OvGU
**Group:** Akshat · Dhairithri · Veer · Harshit
**Dataset:** OULAD (Open University Learning Analytics Dataset) — already present in `anonymisedData/`
**Paper due:** 10 July · **Presentation:** 16–17 July

---

## 0. One-paragraph summary

We frame OULAD as a **sequential recommendation** problem: predict the **next VLE activity** a student will open, given their past activity sequence. We build **4 recommenders** (Popularity, Collaborative Filtering, SASRec, LLM), **audit them for fairness** across 4 student groups (gender, age band, disability, socioeconomic/IMD), then apply **2 mitigation strategies** (fair training vs. output reranking) and report the **accuracy-vs-fairness trade-off**.

---

## 1. Locked design decisions

These were decided up front; the whole build depends on them.

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| D1 | What is an "item"? | **`id_site`** (~6,365 VLE activities) | True recsys task; matches SASRec/FairSR literature. (Robustness check: `activity_type`, ~20 types.) |
| D2 | User/session unit | `(id_student, code_module, code_presentation)` | A student can take multiple presentations; each is its own learning journey. |
| D3 | Compute | **GPU available** | Full ~10M-click dataset usable; SASRec trains comfortably. |
| D4 | LLM design | **Retrieve-then-rerank** | Can't generate over 6,365 items. CF/popularity gives 50 candidates → GPT-4o-mini reranks to top-10. |
| D5 | Split scheme | **Leave-last-out per session** | Last interaction = test, 2nd-to-last = validation, rest = history. Standard for SASRec, no leakage. |
| D6 | Min sequence length | **≥ 3 interactions** | Needed for history/val/test split. Report % of sessions dropped. |
| D7 | `imd_band` binarization | **0–40% (disadvantaged) vs 50–100% (advantaged)** | SPD/EOD/AOD require binary groups. Keep full 10 bands for per-group recall-gap (RQ3). NaN = explicit "unknown", reported, not silently dropped. |
| D8 | LLM test-set size | **Fixed random sample of N sessions, same N for all models** | Controls API cost while keeping model comparison fair. |

---

## 2. Research questions & hypotheses (what success looks like)

- **RQ1** — Of the 4 recommenders, which is most accurate **and** most fair?
  *Hypothesis:* SASRec = best balance; LLM = most accurate but least fair.
- **RQ2** — Of the 2 fixes (fair-training vs. reranking), which reduces unfairness more, and at what accuracy cost?
  *Hypothesis:* Reranking loses <5% accuracy — the practical fix.
- **RQ3** — Is unfairness worse for some groups than others?
  *Hypothesis:* Largest for IMD (socioeconomic), smallest for gender.

Every experiment must trace back to one of these three questions.

---

## 3. Dataset reference (actual columns in `anonymisedData/`)

| File | Rows | Key columns we use |
|------|------|--------------------|
| `studentVle.csv` (450 MB) | ~10.6M | `id_student`, `code_module`, `code_presentation`, `id_site`, `date`, `sum_click` — **the clickstream → sequences** |
| `vle.csv` | 6,364 | `id_site` → `activity_type` (resource, oucontent, forumng, quiz, …) |
| `studentInfo.csv` | 32,593 | `gender`, `age_band`, `disability`, `imd_band`, `region`, `highest_education`, `final_result` — **protected attributes** |
| `studentRegistration.csv` | 32,593 | `date_registration`, `date_unregistration` (optional filtering) |
| `studentAssessment.csv` | 173,912 | `id_assessment`, `score` (not core to recsys; optional signal) |
| `assessments.csv` | 206 | assessment metadata (optional) |
| `courses.csv` | 22 | `module_presentation_length` (optional) |

**Protected groups:** `gender` (M/F), `age_band` (`0-35` / `35-55` / `55<=`), `disability` (Y/N), `imd_band` (10 bands + NaN).

---

## 4. Repository structure (Phase 0 deliverable)

```
HCAI/
├── anonymisedData/          # raw CSVs — NEVER edit
├── data/processed/          # cached parquet (gitignored)
├── src/
│   ├── data/
│   │   ├── build_sequences.py   # clickstream -> sequences.parquet
│   │   ├── splits.py            # leave-last-out split
│   │   └── protected.py         # group labels, imd binarization, NaN handling
│   ├── models/
│   │   ├── base.py              # shared interface: fit(), recommend()
│   │   ├── popularity.py
│   │   ├── cf.py                # LightFM WARP
│   │   ├── sasrec.py            # PyTorch, 2-layer, 50-dim
│   │   └── llm.py               # retrieve-then-rerank, GPT-4o-mini
│   ├── eval/
│   │   ├── accuracy.py          # Recall@10, NDCG@10, MRR
│   │   └── fairness.py          # SPD, EOD, AOD (AIF360) + per-group recall gap
│   ├── mitigation/
│   │   ├── fair_loss.py         # Fix 1: FairSR-style training
│   │   └── rerank.py            # Fix 2: post-hoc exposure rerank
│   └── utils/
│       ├── seeds.py             # set_seed() for random/numpy/torch
│       └── io.py
├── experiments/             # YAML configs + run scripts
├── results/                 # metrics.csv, plots (committed)
├── paper/
├── requirements.txt
├── PROJECT_PLAN.md          # this file
└── README.md
```

**`requirements.txt`:** `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `lightfm`, `torch`, `aif360`, `scipy`, `matplotlib`, `openai`, `pyyaml`, `tqdm` (Python ≥ 3.10).

---

## 5. Step-by-step plan

### Phase 0 — Scaffold (½ day)
- [ ] Create the directory structure above.
- [ ] Write `requirements.txt`; `pip install -r requirements.txt`.
- [ ] `utils/seeds.py`: single `set_seed(s)` seeding `random`, `numpy`, `torch` (+ `cudnn.deterministic = True`). **Every run takes a seed; we need 5 seeds.**
- [ ] `git init`; commit raw-data checksum + `requirements.txt`. `.gitignore`: `data/processed/`, secrets, `__pycache__`.
- [ ] Store OpenAI key in an **environment variable** — never in code or git.

### Phase 1 — Data pipeline (Week 1) — *foundation*
- [ ] **1.1 Build sequences** (`build_sequences.py`):
  - Chunk-read `studentVle.csv` (`chunksize≈2M`); cast ids to `int32`, `sum_click` to `int16`.
  - Group by `(id_student, code_module, code_presentation)`, sort by `date`.
  - Collapse consecutive/same-day repeats of the same `id_site` (optionally keep `sum_click` as weight).
  - Drop sessions with < 3 interactions (D6); **record the drop rate**.
  - Save `data/processed/sequences.parquet`: one row per session = ordered `id_site` list + 4 protected attributes joined from `studentInfo.csv`.
- [ ] **1.2 Vocab maps:** `id_site → contiguous index` (index 0 = padding). Save it; all models share it. Join `vle.csv` to attach `activity_type` per site.
- [ ] **1.3 Protected attributes** (`protected.py`): binarize `imd_band` per D7; handle NaN explicitly; produce per-session group labels for all 4 attributes.
- [ ] **1.4 Split** (`splits.py`): leave-last-out per session (D5); deterministic and **seed-independent** (only model init/shuffle varies by seed).
- [ ] **1.5 EDA notebook:** group sizes per attribute, sequence-length distribution, % dropped. **Flag any tiny group now** — fairness metrics on small groups are noisy.

**Week 1 deliverable (matches slides):** data pipeline + fairness module + popularity baseline running.

### Phase 2 — The four recommenders (Weeks 1–4)
All implement the same `base.py` interface: `fit(train)` and `recommend(history) -> ranked item indices`.
- [ ] **2.1 Popularity** (Week 1): next-item frequency **conditioned on presentation** (sites are presentation-specific); exclude already-seen. Also serves as LLM candidate generator.
- [ ] **2.2 CF** (Week 2): **LightFM + WARP loss**; `student × item` matrix (binary or `sum_click`-weighted); rank all items, exclude seen. Tune **for accuracy only** (ceteris paribus).
- [ ] **2.3 SASRec** (Week 3): PyTorch, **2 self-attention blocks, 50-dim**, causal mask, next-item prediction with negative sampling. Train on GPU, checkpoint best validation. **Keep the loss a swappable function** (Fix 1 will replace it).
- [ ] **2.4 LLM** (Week 4): retrieve-then-rerank (D4) — CF/popularity top-50 → GPT-4o-mini reranks to top-10. Prompt with readable history (`activity_type` + site info). **Cache every response by prompt hash**; evaluate on the fixed N-sample (D8); validate JSON output; budget API spend before the 5-seed run.

**Week 4 deliverable:** all 4 models callable through one interface, smoke-tested.

### Phase 3 — Measurement (build in Week 1, right after popularity)
- [ ] **3.1 Accuracy** (`accuracy.py`): `Recall@10`, `NDCG@10`, `MRR` vs. held-out next item.
- [ ] **3.2 Fairness** (`fairness.py`, **AIF360**): favorable outcome = "true next item in top-K" (hit). Compute **SPD, EOD, AOD** per attribute via AIF360 `ClassificationMetric` (privileged vs unprivileged). Also compute **per-group Recall@10 gap** (max−min) — the interpretable number for RQ3.
  - *Gotcha:* AIF360 needs binarized groups + a 0/1 outcome per user → write an adapter into `BinaryLabelDataset`.
- [ ] **3.3** Single `evaluate(model)` returns `{accuracy, per-attribute SPD/EOD/AOD, per-group recall}`. Powers RQ1 and RQ3.

### Phase 4 — Mitigation (Week 5)
- [ ] **4.1 Fix 1 — Fair training** (`fair_loss.py`): FairSR-style `loss = rec_loss + λ · group_divergence` (penalize predicted-score distribution gap between groups within a batch). Plug into SASRec loop. **Sweep λ** ∈ {0, 0.1, 0.5, 1, …} — each λ = one trade-off point.
- [ ] **4.2 Fix 2 — Reranking** (`rerank.py`): post-hoc top-K rerank to balance group **exposure** (MMR-style fairness penalty / reject-option idea). Apply on the **same SASRec base** as Fix 1 → clean RQ2 comparison. Strength knob sweeps the curve.

**Week 5 deliverable:** both fixes runnable, each with a fairness↔accuracy knob.

### Phase 5 — Experiments (Week 6)
- [ ] **5.1 Run matrix:** RQ1 = {4 models} × {5 seeds}; RQ2 = {SASRec} × {Fix1 λ-sweep, Fix2 strength-sweep} × {5 seeds}; RQ3 = per-attribute breakdown. Drive from YAML configs; log every run.
- [ ] **5.2 Aggregate:** mean ± std across seeds; **bootstrap CIs** + **paired Wilcoxon** (`scipy.stats`) to test significance of fairness gains / accuracy losses.
- [ ] **5.3** Cache & checkpoint everything; never recompute.

**Week 6 deliverable:** `results/metrics.csv` with every number the paper needs.

### Phase 6 — Reporting (Weeks 6–7)
- [ ] **Trade-off plot:** x = accuracy (Recall@10/NDCG@10), y = fairness (e.g. 1−|SPD| or 1/recall-gap). Each model = a point; each fix = a curve. **Answers RQ1 + RQ2 visually.**
- [ ] **RQ3 figure:** per-group recall gap per attribute (bar chart) — confirm/refute "IMD worst, gender smallest."
- [ ] **Tables:** accuracy × fairness per model with CIs and significance stars.

### Phase 7 — Paper & presentation (Weeks 7–9)
- [ ] **Paper (10 July):** Intro → Related work → **Data & task definition (document D1–D8!)** → Methods → Results (RQ1/2/3) → Discussion → Limitations (task framing, dropped sessions, LLM sampling, small groups) → Conclusion.
- [ ] **Slides (16–17 July):** lead with the trade-off plot; one slide per RQ verdict.

---

## 6. Tech stack (per slides)

- **Language:** Python 3.10+
- **Data:** pandas, numpy, pyarrow
- **ML/DL:** scikit-learn, LightFM (WARP), PyTorch
- **LLM:** OpenAI API (GPT-4o-mini)
- **Fairness:** IBM AIF360
- **Stats:** scipy.stats (paired Wilcoxon, bootstrap CIs)
- **Repro:** 5 seeds, git, requirements.txt
- **Controls:** ceteris paribus; all models tuned for **accuracy only** (then audited for fairness).

---

## 7. Metrics cheat-sheet

| Category | Metrics |
|----------|---------|
| Accuracy | Recall@10, NDCG@10, MRR |
| Fairness (binary group, AIF360) | SPD (statistical parity diff), EOD (equal-opportunity diff), AOD (average-odds diff) |
| Fairness (interpretable) | per-group Recall@10 gap (max−min) |

---

## 8. Risk register

| Risk | Impact | Mitigation |
|------|--------|------------|
| `studentVle.csv` 450 MB blows up memory | Pipeline fails | Chunked read, downcast dtypes, cache parquet once |
| `imd_band` NaN silently corrupts fairness | Wrong RQ3 result | Explicit "unknown" group, reported (D7) |
| Tiny protected group | Noisy/unstable fairness metrics | Check group sizes in EDA (1.5); report CIs |
| LLM API cost / nondeterminism | Budget overrun, irreproducible | Fixed N-sample (D8), response caching, JSON validation |
| Data leakage in split | Inflated accuracy | Leave-last-out, seed-independent split (D5) |
| Sites are presentation-specific | Popularity/CF recommend invalid items | Condition recommendations on the session's presentation |
| SASRec loss not swappable | Fix 1 hard to implement | Write loss as a pluggable function from the start |

---

## 9. Key references

- **Primary:** Li, Hsu & Zhang (2022). *FairSR: Fairness-aware Sequential Recommendation.* ACM TIST 13(1).
- Kuzilek, Hlosta & Zdrahal (2017). *OULAD dataset paper.*
- Kang & McAuley (2018). *SASRec.*
- Baker & Hawn (2021). *Algorithmic Bias in Education.*

---

## 10. Definition of done

- [ ] All 4 recommenders evaluated on identical splits, 5 seeds, with CIs.
- [ ] Both mitigation fixes produce trade-off curves on the same SASRec base.
- [ ] RQ1/RQ2/RQ3 each have a figure + a stated verdict (hypothesis confirmed or refuted).
- [ ] `results/metrics.csv` + plots committed; runs reproducible from configs.
- [ ] Paper submitted (10 July); slides ready (16–17 July).
