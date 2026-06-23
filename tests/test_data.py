"""Tests for data splits and feature preprocessing."""
from __future__ import annotations

from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.data.splits import make_splits
from src.data.features import _bucket_gap, build_module_map, build_activity_type_map, build_features


def test_make_splits(small_splits: pd.DataFrame) -> None:
    """Test that make_splits generates inputs and targets of correct length."""
    # make_splits is already applied in small_splits fixture, let's run it on raw sequences
    raw_df = small_splits[["id_student", "code_module", "code_presentation", "seq"]]
    splits = make_splits(raw_df)
    
    assert "test_target" in splits.columns
    assert "test_input" in splits.columns
    assert "val_target" in splits.columns
    assert "val_input" in splits.columns
    assert "train_history" in splits.columns
    
    # Verify split logic
    for row in splits.itertuples():
        seq = row.seq
        assert row.test_target == seq[-1]
        assert row.test_input == seq[:-1]
        assert row.val_target == seq[-2]
        assert row.val_input == seq[:-2]
        assert row.train_history == seq[:-2]


def test_bucket_gap() -> None:
    """Test that time differences map to correct discrete buckets."""
    assert _bucket_gap(0.0) == 1
    assert _bucket_gap(0.5) == 2
    assert _bucket_gap(1.0) == 2
    assert _bucket_gap(2.5) == 3
    assert _bucket_gap(6.0) == 4
    assert _bucket_gap(12.0) == 5
    assert _bucket_gap(20.0) == 6


def test_build_module_map(small_splits: pd.DataFrame) -> None:
    """Test that code modules are correctly mapped to integer indices."""
    mapping, n_modules = build_module_map(small_splits)
    # small_splits has modules 'AAA' and 'BBB'
    assert "AAA" in mapping
    assert "BBB" in mapping
    assert mapping["AAA"] in [1, 2]
    assert mapping["BBB"] in [1, 2]
    assert mapping["AAA"] != mapping["BBB"]
    assert n_modules == 3  # padding (0) + 2 modules


@patch("pandas.read_parquet")
def test_build_activity_type_map(mock_read_parquet) -> None:
    """Test mapping item indices to activity type indices using mocked vocabulary."""
    mock_vocab = pd.DataFrame({
        "item_idx": [1, 2, 3],
        "activity_type": ["forumng", "quiz", "forumng"]
    })
    mock_read_parquet.return_value = mock_vocab
    
    mapping, n_types = build_activity_type_map()
    assert mapping[1] == mapping[3]  # same type
    assert mapping[1] != mapping[2]  # different type
    assert n_types == 3  # padding (0) + 2 unique types (forumng, quiz)
