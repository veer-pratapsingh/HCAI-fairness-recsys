# Fairness Auditing & Mitigation in Adaptive Learning Pathway Recommendation

HCAI project (OvGU). Auditing and mitigating fairness in next-activity
recommendation on the OULAD dataset. See **[PROJECT_PLAN.md](PROJECT_PLAN.md)**
for the full step-by-step roadmap and design decisions.

## Setup

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows bash
pip install -r requirements.txt
```

### Data

The raw OULAD CSVs must be in `anonymisedData/`. They are **not committed** to
this repo (the clickstream `studentVle.csv` is ~450 MB, over GitHub's limit).
OULAD is free and public (CC-BY 4.0): download it from
<https://analyse.kmi.open.ac.uk/open_dataset> and unzip the 7 CSVs
(`studentVle.csv`, `studentInfo.csv`, `vle.csv`, `courses.csv`, `assessments.csv`,
`studentAssessment.csv`, `studentRegistration.csv`) into `anonymisedData/`.

## Phase 1 — build the data (Week 1, DONE)

Run from the project root, in order:

```bash
python -m src.data.build_sequences   # clickstream -> data/processed/sequences.parquet (+ item_vocab)
python -m src.data.splits            # leave-last-out -> data/processed/splits.parquet
python -m src.data.eda_summary       # sanity-check group sizes & sequence lengths
```

`build_sequences.py` streams the 450 MB `studentVle.csv` in chunks; lower
`--chunksize` if memory is tight.

### What you get

| Artifact | Contents |
|----------|----------|
| `data/processed/sequences.parquet` | one row per session: ordered `seq` (item indices), `seq_len`, and the 4 protected attributes + derived group columns |
| `data/processed/item_vocab.parquet` | `id_site` ↔ contiguous `item_idx` (0 = padding) + `activity_type` |
| `data/processed/splits.parquet` | leave-last-out `train_history` / `val_*` / `test_*` columns |

### Current data summary (full dataset)

- **28,761 sessions**, 25,745 students, **6,268 items** (VLE sites)
- Median sequence length 216 (mean 331) — long; SASRec should truncate to the last N.
- Group sizes flagged for fairness noise: `age_band 55<=` (0.7% → use `age_binary`),
  `imd unknown` (3.6%, reported as its own group per Decision D7).

## Layout

```
src/data/    build_sequences.py, splits.py, protected.py, eda_summary.py
src/models/  popularity, cf, sasrec, llm        (Phase 2 — next)
src/eval/    accuracy, fairness                  (Phase 3)
src/mitigation/ fair_loss, rerank               (Phase 4)
src/utils/   seeds.py, paths.py
```

## Reproducibility

Every run takes a seed; experiments use the 5 seeds in `src.utils.seeds.SEEDS`.
Call `set_seed(s)` at the start of each run.
