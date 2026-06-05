"""Phase 3.2: fairness audit of recommender predictions.

Two complementary views, both computed from the SAME per-session predictions the
accuracy evaluator produced (so accuracy and fairness never disagree):

A) Per-group accuracy + gap (the interpretable signal, drives RQ3)
   Recall@K / NDCG@K / MRR for every value of every protected attribute, plus
   the max-min GAP across groups. Multi-group, no binarization needed.

B) AIF360-style group-fairness metrics (SPD / EOD / AOD) per binary attribute
   We frame recommendation as pointwise relevance classification using sampled
   negatives (the standard "sampled metrics" setup):
     - each test session contributes 1 positive (its true next item) and
       `n_neg` sampled negatives drawn from the session's presentation;
     - a pair is "predicted positive" iff that item lands in the model's top-K.
   From the per-group confusion counts:
     SPD = PPR_unpriv - PPR_priv         (statistical parity difference)
     EOD = TPR_unpriv - TPR_priv         (equal-opportunity difference)
     AOD = 0.5*((FPR_u - FPR_p) + (TPR_u - TPR_p))   (average-odds difference)
   These are the exact AIF360 ClassificationMetric definitions; AIF360 can be
   dropped in later without changing any numbers (it is not yet installed).
   Ideal value for all three is 0; sign shows which group is favoured.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.protected import PROTECTED_ATTRS
from src.utils import paths

# Attribute -> the per-session column holding its (binarized) group label.
ATTR_COLUMN = {
    "gender": "gender",
    "age_band": "age_binary",   # 3-level 55<= is too small; use binary for SPD/EOD/AOD
    "disability": "disability",
    "imd": "imd_binary",
}
# Multi-group columns used for the per-group recall-gap view (RQ3).
GROUP_COLUMNS = ["gender", "age_band", "disability", "imd_binary"]


# ----------------------------------------------------------------------------
# View A: per-group accuracy + gap
# ----------------------------------------------------------------------------
def per_group_accuracy(per_session: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Recall@K / NDCG@K / MRR per value of each protected attribute, with gaps.

    `per_session` must already carry the group columns (merge them on
    id_student/code_module/code_presentation before calling).
    """
    rows = []
    for attr in GROUP_COLUMNS:
        grp = per_session.groupby(attr, observed=True)
        agg = grp.agg(
            n=("hit", "size"),
            recall=("hit", "mean"),
            ndcg=("ndcg", "mean"),
            mrr=("rr", "mean"),
        ).reset_index().rename(columns={attr: "group_value"})
        agg.insert(0, "attribute", attr)
        rows.append(agg)
        # Gap row: max - min recall across this attribute's groups (ignore tiny ones noted in EDA).
        gap = agg["recall"].max() - agg["recall"].min()
        rows.append(pd.DataFrame([{
            "attribute": attr, "group_value": "__GAP(max-min)__",
            "n": int(agg["n"].sum()), "recall": gap,
            "ndcg": agg["ndcg"].max() - agg["ndcg"].min(),
            "mrr": agg["mrr"].max() - agg["mrr"].min(),
        }]))
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------
# View B: SPD / EOD / AOD via sampled-negative confusion counts
# ----------------------------------------------------------------------------
def _presentation_item_universe(sequences: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """Valid candidate items per (module, presentation) = union of items seen there."""
    universe: dict[tuple[str, str], set] = {}
    for module, pres, seq in zip(
        sequences["code_module"], sequences["code_presentation"], sequences["seq"]
    ):
        universe.setdefault((module, pres), set()).update(seq)
    return {key: np.fromiter(items, dtype=np.int64) for key, items in universe.items()}


def _confusion_counts(
    per_session: pd.DataFrame,
    test_input: pd.Series,
    universe: dict[tuple[str, str], np.ndarray],
    k: int,
    n_neg: int,
    seed: int,
) -> pd.DataFrame:
    """Per-session TP and FP counts using sampled negatives.

    Returns the input frame with added columns: tp (0/1), fp (0..n_neg), n_neg.
    """
    rng = np.random.RandomState(seed)
    tp_list, fp_list, nneg_list = [], [], []
    for row, seen in zip(per_session.itertuples(index=False), test_input):
        topk = set(row.topk)
        tp_list.append(1 if row.target in topk else 0)

        items = universe.get((row.code_module, row.code_presentation))
        exclude = set(seen)
        exclude.add(row.target)
        pool = items[~np.isin(items, list(exclude))] if items is not None else np.array([], dtype=np.int64)
        if len(pool) == 0:
            fp_list.append(0)
            nneg_list.append(0)
            continue
        m = min(n_neg, len(pool))
        negs = rng.choice(pool, size=m, replace=False)
        fp_list.append(int(sum(1 for n in negs if n in topk)))
        nneg_list.append(m)

    out = per_session.copy()
    out["tp"] = tp_list
    out["fp"] = fp_list
    out["n_neg"] = nneg_list
    return out


def _rates(df: pd.DataFrame) -> dict[str, float]:
    """TPR / FPR / PPR from confusion counts over a group of sessions."""
    n_pos = len(df)
    n_neg = int(df["n_neg"].sum())
    tp = int(df["tp"].sum())
    fp = int(df["fp"].sum())
    tpr = tp / n_pos if n_pos else float("nan")
    fpr = fp / n_neg if n_neg else float("nan")
    ppr = (tp + fp) / (n_pos + n_neg) if (n_pos + n_neg) else float("nan")
    return {"TPR": tpr, "FPR": fpr, "PPR": ppr, "n": n_pos}


def group_fairness_metrics(
    per_session: pd.DataFrame,
    test_input: pd.Series,
    sequences: pd.DataFrame,
    k: int = 10,
    n_neg: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """SPD / EOD / AOD per binary protected attribute (AIF360 definitions)."""
    universe = _presentation_item_universe(sequences)
    counts = _confusion_counts(per_session, test_input, universe, k, n_neg, seed)

    rows = []
    for attr, cfg in PROTECTED_ATTRS.items():
        col = ATTR_COLUMN[attr]
        priv_val, unpriv_val = cfg["privileged"], cfg["unprivileged"]
        priv = counts[counts[col] == priv_val]
        unpriv = counts[counts[col] == unpriv_val]
        if len(priv) == 0 or len(unpriv) == 0:
            continue
        p, u = _rates(priv), _rates(unpriv)
        spd = u["PPR"] - p["PPR"]
        eod = u["TPR"] - p["TPR"]
        aod = 0.5 * ((u["FPR"] - p["FPR"]) + (u["TPR"] - p["TPR"]))
        rows.append({
            "attribute": attr,
            "privileged": priv_val, "unprivileged": unpriv_val,
            "n_priv": p["n"], "n_unpriv": u["n"],
            "recall_priv": p["TPR"], "recall_unpriv": u["TPR"],
            "SPD": spd, "EOD": eod, "AOD": aod,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Convenience: attach group labels to a per-session frame
# ----------------------------------------------------------------------------
def attach_groups(per_session: pd.DataFrame, sequences: pd.DataFrame) -> pd.DataFrame:
    """Merge protected-group columns onto per-session predictions by session key."""
    keys = ["id_student", "code_module", "code_presentation"]
    group_cols = keys + ["gender", "age_band", "age_binary", "disability", "imd_binary"]
    return per_session.merge(sequences[group_cols], on=keys, how="left")


def load_sequences() -> pd.DataFrame:
    return pd.read_parquet(paths.SEQUENCES_PARQUET)
