"""Protected-attribute handling: group definitions, IMD binarization, NaN policy.

This is the single source of truth for how the 4 protected attributes
(gender, age_band, disability, imd_band) are turned into groups. Both the data
pipeline and the fairness evaluator import from here so the definitions never
drift apart.

Decision D7 (see PROJECT_PLAN.md):
  - imd_band is binarized into disadvantaged (0-40%) vs advantaged (50-100%).
  - The full 10-band value is kept for the per-group recall-gap analysis (RQ3).
  - Missing imd_band is an explicit "unknown" group, reported, never silently dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Raw IMD bands that count as socioeconomically disadvantaged.
# OULAD encodes these as strings like "0-10%", "10-20", ... (the trailing "%"
# and exact spacing vary between rows), so we normalise before comparing.
DISADVANTAGED_BANDS = {"0-10%", "10-20%", "20-30%", "30-40%", "40-50%"}

# For AIF360-style metrics (SPD/EOD/AOD) each attribute needs a binary
# privileged vs unprivileged split. `privileged` is the value treated as 1.
# These are conventional choices for an *audit* (does the model disadvantage the
# historically marginalised group?), not value judgements.
PROTECTED_ATTRS = {
    "gender": {"column": "gender", "privileged": "M", "unprivileged": "F"},
    "age_band": {"column": "age_band", "privileged": "0-35", "unprivileged": "35+"},
    "disability": {"column": "disability", "privileged": "N", "unprivileged": "Y"},
    "imd": {"column": "imd_binary", "privileged": "advantaged", "unprivileged": "disadvantaged"},
}


def _normalise_imd(value: object) -> str:
    """Normalise a raw imd_band cell to the canonical "lo-hi%" form."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return ""
    if not s.endswith("%"):
        s = s + "%"
    return s


def binarize_imd(imd_band: pd.Series) -> pd.DataFrame:
    """Map raw imd_band to (imd_binary, imd_unknown).

    Returns a DataFrame with:
      - imd_binary: "disadvantaged" / "advantaged" / "unknown"
      - imd_unknown: bool flag for missing values
    """
    norm = imd_band.map(_normalise_imd)
    unknown = norm == ""
    binary = np.where(
        unknown,
        "unknown",
        np.where(norm.isin(DISADVANTAGED_BANDS), "disadvantaged", "advantaged"),
    )
    return pd.DataFrame({"imd_binary": binary, "imd_unknown": unknown})


def collapse_age_band(age_band: pd.Series) -> pd.Series:
    """Collapse OULAD's 3 age bands into a binary 0-35 vs 35+ for SPD/EOD/AOD.

    The original 3-level age_band is retained separately for the per-group
    recall-gap analysis; this binary form is only for the AIF360 metrics.
    """
    return np.where(age_band.astype(str).str.strip() == "0-35", "0-35", "35+")


def add_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived group columns (imd_binary, imd_unknown, age_binary) in place.

    Expects df to already contain the raw studentInfo columns:
    gender, age_band, disability, imd_band.
    """
    out = df.copy()
    imd = binarize_imd(out["imd_band"])
    out["imd_binary"] = imd["imd_binary"].values
    out["imd_unknown"] = imd["imd_unknown"].values
    out["age_binary"] = collapse_age_band(out["age_band"])
    return out
