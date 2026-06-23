"""Tests for fairness and intersectional evaluation modules."""
from __future__ import annotations

import pandas as pd
import pytest

from src.eval.fairness import per_group_accuracy, group_fairness_metrics, attach_groups
from src.eval.intersectional import intersectional_analysis


@pytest.fixture
def dummy_per_session(small_splits: pd.DataFrame) -> pd.DataFrame:
    """Create a dummy per_session dataframe matching evaluate_model output."""
    # Let's say top-3 was recommended and we hit the target sometimes.
    df = pd.DataFrame({
        "id_student": small_splits["id_student"],
        "code_module": small_splits["code_module"],
        "code_presentation": small_splits["code_presentation"],
        "hit": [1, 0, 1, 0, 1] * 4,  # alternating hits
        "ndcg": [1.0, 0.0, 0.5, 0.0, 0.6] * 4,
        "rr": [1.0, 0.0, 0.5, 0.0, 0.5] * 4,
        "target": small_splits["test_target"],
        "topk": [[1, 2, 3]] * 20,
    })
    return df


def test_attach_groups(dummy_per_session: pd.DataFrame, small_splits: pd.DataFrame) -> None:
    """Test that attach_groups correctly adds demographic attributes."""
    merged = attach_groups(dummy_per_session, small_splits)
    for col in ["gender", "age_band", "age_binary", "disability", "imd_binary"]:
        assert col in merged.columns
    assert len(merged) == len(dummy_per_session)


def test_per_group_accuracy(dummy_per_session: pd.DataFrame, small_splits: pd.DataFrame) -> None:
    """Test that per-group accuracy and recall gaps are calculated."""
    merged = attach_groups(dummy_per_session, small_splits)
    grp_acc = per_group_accuracy(merged, k=10)
    
    assert isinstance(grp_acc, pd.DataFrame)
    assert "attribute" in grp_acc.columns
    assert "group_value" in grp_acc.columns
    assert "recall" in grp_acc.columns
    
    # Check that GAP rows exist
    gaps = grp_acc[grp_acc["group_value"] == "__GAP(max-min)__"]
    assert len(gaps) == 4


def test_group_fairness_metrics(dummy_per_session: pd.DataFrame, small_splits: pd.DataFrame) -> None:
    """Test that AIF360-style group fairness metrics are computed."""
    merged = attach_groups(dummy_per_session, small_splits)
    test_input = small_splits["test_input"]
    
    # Run group fairness metrics
    metrics = group_fairness_metrics(
        merged, test_input, small_splits, k=3, n_neg=5, seed=42
    )
    
    assert isinstance(metrics, pd.DataFrame)
    assert "SPD" in metrics.columns
    assert "EOD" in metrics.columns
    assert "AOD" in metrics.columns
    assert len(metrics) > 0


def test_intersectional_analysis(dummy_per_session: pd.DataFrame, small_splits: pd.DataFrame) -> None:
    """Test that intersectional fairness analyses and gaps are computed."""
    merged = attach_groups(dummy_per_session, small_splits)
    inter_df = intersectional_analysis(merged, k=10)
    
    assert isinstance(inter_df, pd.DataFrame)
    assert "attr1" in inter_df.columns
    assert "attr2" in inter_df.columns
    assert "recall" in inter_df.columns
    
    gaps = inter_df[inter_df["val1"] == "__GAP(max-min)__"]
    # 6 combinations of 4 attributes: gender/imd, gender/disability, etc.
    assert len(gaps) == 6
