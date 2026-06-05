"""Phase 1.1-1.4: turn the raw OULAD clickstream into model-ready sequences.

Pipeline:
  1. Read studentInfo.csv  -> protected attributes per enrolment.
  2. Read vle.csv          -> id_site -> activity_type map.
  3. Stream studentVle.csv -> per-session ordered visit sequences (collapsing
     consecutive same-site repeats), in chunks to bound memory.
  4. Join protected attributes, drop sessions shorter than MIN_SEQ_LEN.
  5. Build a contiguous item vocab (index 0 reserved for padding) and map the
     sequences onto it.
  6. Write sequences.parquet + item_vocab.parquet.

A "session" (the recommender's user unit, Decision D2) is one
(id_student, code_module, code_presentation) triple.

Run from the project root:
    python -m src.data.build_sequences
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.data.protected import add_group_columns
from src.utils import paths

# Decision D6: minimum interactions to keep a session (need history + val + test).
MIN_SEQ_LEN = 3

# Session key (Decision D2).
SESSION_KEYS = ["id_student", "code_module", "code_presentation"]

# Columns we actually need from the 450 MB clickstream, with memory-frugal dtypes.
VLE_USECOLS = ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]
VLE_DTYPES = {
    "code_module": "category",
    "code_presentation": "category",
    "id_student": "int32",
    "id_site": "int32",
    "date": "int16",
    "sum_click": "int32",
}


def load_protected() -> pd.DataFrame:
    """Load per-enrolment protected attributes from studentInfo.csv."""
    cols = [
        "code_module", "code_presentation", "id_student",
        "gender", "age_band", "disability", "imd_band",
    ]
    info = pd.read_csv(paths.STUDENT_INFO_CSV, usecols=cols)
    info = add_group_columns(info)  # adds imd_binary, imd_unknown, age_binary
    return info


def load_activity_types() -> pd.DataFrame:
    """Load id_site -> activity_type from vle.csv (used for the LLM prompt + RQ analysis)."""
    vle = pd.read_csv(paths.VLE_CSV, usecols=["id_site", "activity_type"])
    return vle.drop_duplicates("id_site").reset_index(drop=True)


def read_clickstream(chunksize: int) -> pd.DataFrame:
    """Read studentVle.csv in chunks and concatenate the downcast frames.

    With downcasting the ~10.6M rows fit comfortably in a few hundred MB.
    """
    frames = []
    reader = pd.read_csv(
        paths.STUDENT_VLE_CSV,
        usecols=VLE_USECOLS,
        dtype=VLE_DTYPES,
        chunksize=chunksize,
    )
    for i, chunk in enumerate(reader):
        frames.append(chunk)
        print(f"  read chunk {i + 1} ({len(chunk):,} rows)", flush=True)
    df = pd.concat(frames, ignore_index=True)
    print(f"  total clickstream rows: {len(df):,}", flush=True)
    return df


def build_visit_sequences(clicks: pd.DataFrame) -> pd.DataFrame:
    """Order each session by date and collapse consecutive same-site visits.

    Returns one row per session with an ordered list of raw id_site values.
    """
    # Stable sort so same-day ties keep their original (file) order.
    clicks = clicks.sort_values(SESSION_KEYS + ["date"], kind="stable")

    # Collapse consecutive repeats of the same site within a session
    # (covers same-day repeat clicks and adjacent re-visits, Decision D1/D6).
    grp = clicks.groupby(SESSION_KEYS, observed=True, sort=False)
    prev_site = grp["id_site"].shift()
    same_session = ~grp.cumcount().eq(0)  # False at the first row of each session
    is_repeat = same_session & prev_site.eq(clicks["id_site"])
    collapsed = clicks.loc[~is_repeat]

    sequences = (
        collapsed.groupby(SESSION_KEYS, observed=True, sort=False)["id_site"]
        .agg(list)
        .reset_index()
        .rename(columns={"id_site": "seq_sites"})
    )
    sequences["seq_len"] = sequences["seq_sites"].str.len()
    return sequences


def build_item_vocab(sequences: pd.DataFrame, activity: pd.DataFrame) -> pd.DataFrame:
    """Assign each surviving id_site a contiguous index (0 reserved for padding)."""
    all_sites = np.unique(np.concatenate(sequences["seq_sites"].to_numpy()))
    all_sites.sort()  # deterministic ordering
    vocab = pd.DataFrame({"id_site": all_sites})
    vocab["item_idx"] = np.arange(1, len(vocab) + 1, dtype="int32")  # 0 = padding
    vocab = vocab.merge(activity, on="id_site", how="left")
    vocab["activity_type"] = vocab["activity_type"].fillna("unknown")
    return vocab


def map_sequences_to_idx(sequences: pd.DataFrame, vocab: pd.DataFrame) -> pd.DataFrame:
    """Replace raw id_site lists with contiguous item-index lists."""
    site_to_idx = dict(zip(vocab["id_site"].tolist(), vocab["item_idx"].tolist()))
    sequences = sequences.copy()
    sequences["seq"] = sequences["seq_sites"].map(
        lambda sites: [site_to_idx[s] for s in sites]
    )
    return sequences.drop(columns=["seq_sites"])


def main(chunksize: int) -> None:
    paths.ensure_dirs()

    print("[1/6] Loading protected attributes (studentInfo.csv) ...", flush=True)
    protected = load_protected()

    print("[2/6] Loading activity types (vle.csv) ...", flush=True)
    activity = load_activity_types()

    print("[3/6] Reading clickstream (studentVle.csv) ...", flush=True)
    clicks = read_clickstream(chunksize)

    print("[4/6] Building per-session visit sequences ...", flush=True)
    sequences = build_visit_sequences(clicks)
    del clicks
    n_before = len(sequences)

    # Join protected attributes and drop short sessions.
    sequences = sequences.merge(protected, on=SESSION_KEYS, how="left")
    sequences = sequences[sequences["seq_len"] >= MIN_SEQ_LEN].reset_index(drop=True)
    n_after = len(sequences)
    dropped = n_before - n_after
    print(
        f"      sessions: {n_before:,} total -> {n_after:,} kept "
        f"({dropped:,} dropped, {100 * dropped / max(n_before, 1):.1f}% with < {MIN_SEQ_LEN} visits)",
        flush=True,
    )

    print("[5/6] Building item vocabulary ...", flush=True)
    vocab = build_item_vocab(sequences, activity)
    sequences = map_sequences_to_idx(sequences, vocab)
    print(f"      vocab size: {len(vocab):,} items (indices 1..{len(vocab)}, 0 = padding)", flush=True)

    print("[6/6] Writing parquet artifacts ...", flush=True)
    keep_cols = SESSION_KEYS + [
        "seq", "seq_len",
        "gender", "age_band", "age_binary",
        "disability", "imd_band", "imd_binary", "imd_unknown",
    ]
    sequences[keep_cols].to_parquet(paths.SEQUENCES_PARQUET, index=False)
    vocab.to_parquet(paths.ITEM_VOCAB_PARQUET, index=False)
    print(f"      wrote {paths.SEQUENCES_PARQUET}", flush=True)
    print(f"      wrote {paths.ITEM_VOCAB_PARQUET}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build OULAD recommendation sequences.")
    parser.add_argument(
        "--chunksize", type=int, default=2_000_000,
        help="Rows per read chunk for the 450 MB clickstream (lower = less memory).",
    )
    args = parser.parse_args()
    main(args.chunksize)
