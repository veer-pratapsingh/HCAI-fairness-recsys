"""Central path definitions so every script agrees on where files live."""
from __future__ import annotations

from pathlib import Path

# Project root = two levels up from this file (src/utils/paths.py -> HCAI/)
ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "anonymisedData"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

# Raw CSVs
STUDENT_VLE_CSV = RAW_DIR / "studentVle.csv"
VLE_CSV = RAW_DIR / "vle.csv"
STUDENT_INFO_CSV = RAW_DIR / "studentInfo.csv"
STUDENT_REGISTRATION_CSV = RAW_DIR / "studentRegistration.csv"

# Processed artifacts (produced by build_sequences.py)
SEQUENCES_PARQUET = PROCESSED_DIR / "sequences.parquet"
ITEM_VOCAB_PARQUET = PROCESSED_DIR / "item_vocab.parquet"
SPLITS_PARQUET = PROCESSED_DIR / "splits.parquet"


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
