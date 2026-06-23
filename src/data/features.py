"""Precompute side-feature sequences for Enhanced SASRec.

Builds parallel sequences of:
  - activity_type indices (from item_vocab.parquet)
  - time gap buckets    (from studentVle.csv ``date`` column)
  - module indices      (from splits_df ``code_module`` column)

These features are consumed by :class:`~src.models.sasrec_enhanced.EnhancedSASRecRecommender`
to enrich the item embeddings with contextual side information.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils import paths

# ---------------------------------------------------------------------------
# Time-gap buckets
# ---------------------------------------------------------------------------
# 0 = padding (unused real value), 1 = same-day, 2 = next-day, 3 = 2-3 days,
# 4 = 4-7 days, 5 = 1-2 weeks, 6 = 2+ weeks.
N_TIME_BUCKETS = 7  # including the padding index


def _bucket_gap(days: float) -> int:
    """Map a day-difference to a discrete time-gap bucket index.

    Parameters
    ----------
    days : float
        Non-negative difference in days between two consecutive activities.

    Returns
    -------
    int
        Bucket index in [1, 6].
    """
    if days <= 0:
        return 1   # same day
    if days <= 1:
        return 2   # next day
    if days <= 3:
        return 3   # 2-3 days
    if days <= 7:
        return 4   # 4-7 days
    if days <= 14:
        return 5   # 1-2 weeks
    return 6       # 2+ weeks


# ---------------------------------------------------------------------------
# Activity-type map
# ---------------------------------------------------------------------------

def build_activity_type_map() -> Tuple[Dict[int, int], int]:
    """Build ``item_idx -> activity_type_idx`` mapping from the item vocabulary.

    Index 0 is reserved for padding; real types start at 1.

    Returns
    -------
    mapping : dict[int, int]
        ``{item_idx: activity_type_idx}``
    n_types : int
        Total number of type indices (including the padding index 0).
    """
    vocab = pd.read_parquet(paths.ITEM_VOCAB_PARQUET)
    types = sorted(vocab["activity_type"].unique().tolist())
    type_to_idx: Dict[str, int] = {t: i + 1 for i, t in enumerate(types)}  # 0=pad
    mapping: Dict[int, int] = {
        int(row.item_idx): type_to_idx[row.activity_type]
        for row in vocab.itertuples()
    }
    n_types = len(type_to_idx) + 1  # +1 for the padding index
    return mapping, n_types


# ---------------------------------------------------------------------------
# Module map
# ---------------------------------------------------------------------------

def build_module_map(splits_df: pd.DataFrame) -> Tuple[Dict[str, int], int]:
    """Build ``code_module -> module_idx`` mapping.

    Index 0 is reserved for padding/unknown; real modules start at 1.

    Returns
    -------
    mapping : dict[str, int]
        ``{code_module: module_idx}``
    n_modules : int
        Total number of module indices (including the padding index 0).
    """
    modules = sorted(splits_df["code_module"].unique().tolist())
    mapping: Dict[str, int] = {m: i + 1 for i, m in enumerate(modules)}
    n_modules = len(modules) + 1
    return mapping, n_modules


# ---------------------------------------------------------------------------
# Time-gap sequences
# ---------------------------------------------------------------------------

# Session key: (id_student, code_module, code_presentation)
_SessionKey = Tuple[int, str, str]


def build_time_gap_sequences(splits_df: pd.DataFrame) -> Dict[_SessionKey, List[int]]:
    """Compute per-session time-gap bucket sequences from raw clickstream dates.

    For each session the raw ``studentVle.csv`` is used to recover the ``date``
    column (relative day within the module presentation).  Consecutive activities
    within the collapsed sequence are differenced and mapped to buckets via
    :func:`_bucket_gap`.

    The first item in every sequence gets bucket **1** (same-day, i.e. no prior
    gap).  Padding positions should be filled with bucket **0** by the caller.

    Parameters
    ----------
    splits_df : pd.DataFrame
        The splits dataframe with ``id_student``, ``code_module``,
        ``code_presentation``, and ``seq`` (list of item_idx) columns.

    Returns
    -------
    dict
        ``{(id_student, code_module, code_presentation): [gap_bucket, ...]}``.
        The list length equals ``len(seq)`` for that session.
    """
    # ---- load raw clickstream with dates --------------------------------
    try:
        vle = pd.read_csv(
            paths.STUDENT_VLE_CSV,
            usecols=["code_module", "code_presentation", "id_student",
                     "id_site", "date"],
            dtype={
                "code_module": str,
                "code_presentation": str,
                "id_student": "int32",
                "id_site": "int32",
                "date": "float32",  # float to tolerate NaN
            },
        )
    except Exception:
        # If the raw CSV is unavailable, fall back to all same-day buckets.
        return _fallback_gap_sequences(splits_df)

    # ---- load item vocab for id_site -> item_idx mapping ----------------
    vocab = pd.read_parquet(paths.ITEM_VOCAB_PARQUET)
    site_to_idx: Dict[int, int] = dict(
        zip(vocab["id_site"].tolist(), vocab["item_idx"].tolist())
    )

    # ---- sort and deduplicate exactly like build_sequences.py -----------
    session_keys = ["id_student", "code_module", "code_presentation"]
    vle = vle.sort_values(session_keys + ["date"], kind="stable")

    # Collapse consecutive same-site visits (mirrors build_visit_sequences).
    grp = vle.groupby(session_keys, observed=True, sort=False)
    prev_site = grp["id_site"].shift()
    same_session = ~grp.cumcount().eq(0)
    is_repeat = same_session & prev_site.eq(vle["id_site"])
    collapsed = vle.loc[~is_repeat].copy()

    # Map id_site -> item_idx (drop rows for sites not in vocab).
    collapsed["item_idx"] = collapsed["id_site"].map(site_to_idx)
    collapsed = collapsed.dropna(subset=["item_idx"])
    collapsed["item_idx"] = collapsed["item_idx"].astype("int32")

    # ---- build gap-bucket sequences per session -------------------------
    result: Dict[_SessionKey, List[int]] = {}

    for (student, module, pres), group in collapsed.groupby(
        session_keys, observed=True, sort=False
    ):
        dates = group["date"].values.astype(float)
        items = group["item_idx"].values

        # Compute day-gaps; first position has no predecessor -> same-day (1).
        gaps = [1]  # first item
        for j in range(1, len(dates)):
            d = dates[j] - dates[j - 1]
            if np.isnan(d):
                gaps.append(1)  # missing date -> default same-day
            else:
                gaps.append(_bucket_gap(d))

        key: _SessionKey = (int(student), str(module), str(pres))
        result[key] = gaps

    return result


def _fallback_gap_sequences(splits_df: pd.DataFrame) -> Dict[_SessionKey, List[int]]:
    """Return all-same-day gaps when studentVle.csv is unavailable."""
    result: Dict[_SessionKey, List[int]] = {}
    for row in splits_df.itertuples():
        key = (int(row.id_student), str(row.code_module), str(row.code_presentation))
        seq_len = len(row.seq)
        result[key] = [1] * seq_len  # all same-day
    return result


# ---------------------------------------------------------------------------
# Combined feature bundle
# ---------------------------------------------------------------------------

@dataclass
class FeatureBundle:
    """Container for all precomputed side features.

    Attributes
    ----------
    type_map : dict[int, int]
        ``item_idx -> activity_type_idx``
    n_types : int
        Size of the activity-type embedding table (includes padding idx 0).
    module_map : dict[str, int]
        ``code_module -> module_idx``
    n_modules : int
        Size of the module embedding table (includes padding idx 0).
    gap_sequences : dict[tuple, list[int]]
        ``(id_student, code_module, code_presentation) -> [gap_bucket, ...]``
    """
    type_map: Dict[int, int] = field(default_factory=dict)
    n_types: int = 1
    module_map: Dict[str, int] = field(default_factory=dict)
    n_modules: int = 1
    gap_sequences: Dict[_SessionKey, List[int]] = field(default_factory=dict)


def build_features(splits_df: pd.DataFrame) -> FeatureBundle:
    """Precompute all side features required by the Enhanced SASRec.

    Parameters
    ----------
    splits_df : pd.DataFrame
        The splits dataframe (output of :func:`src.data.splits.make_splits`).

    Returns
    -------
    FeatureBundle
        Dataclass with all feature lookups ready to use.
    """
    print("  [features] building activity-type map ...", flush=True)
    type_map, n_types = build_activity_type_map()

    print("  [features] building module map ...", flush=True)
    module_map, n_modules = build_module_map(splits_df)

    print("  [features] building time-gap sequences ...", flush=True)
    gap_sequences = build_time_gap_sequences(splits_df)
    n_covered = sum(
        1 for row in splits_df.itertuples()
        if (int(row.id_student), str(row.code_module), str(row.code_presentation))
        in gap_sequences
    )
    print(f"  [features] gap sequences cover {n_covered}/{len(splits_df)} sessions",
          flush=True)

    return FeatureBundle(
        type_map=type_map,
        n_types=n_types,
        module_map=module_map,
        n_modules=n_modules,
        gap_sequences=gap_sequences,
    )
