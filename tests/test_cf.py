"""Tests for the CFRecommender model."""
from __future__ import annotations

import pytest
import pandas as pd

from src.models.cf import CFRecommender
from src.models.base import Context


def test_cf_fit_recommend(small_splits: pd.DataFrame, context_aaa: Context) -> None:
    """Test that CFRecommender fits and recommends correct number of items."""
    recommender = CFRecommender(factors=4, iterations=2)
    recommender.fit(small_splits)
    
    # Recommend top-5 items
    history = [1, 2]
    recs = recommender.recommend(history, k=5, context=context_aaa)
    
    assert isinstance(recs, list)
    assert len(recs) <= 5
    # Ensure history items are excluded
    for item in history:
        assert item not in recs


def test_cf_empty_history(small_splits: pd.DataFrame, context_aaa: Context) -> None:
    """Test that CFRecommender returns empty list when history is empty."""
    recommender = CFRecommender(factors=4, iterations=2)
    recommender.fit(small_splits)
    
    recs = recommender.recommend([], k=3, context=context_aaa)
    assert recs == []


def test_cf_unknown_presentation(small_splits: pd.DataFrame) -> None:
    """Test that CFRecommender handles unknown presentations gracefully."""
    recommender = CFRecommender(factors=4, iterations=2)
    recommender.fit(small_splits)
    
    context = Context(code_module="XYZ", code_presentation="9999X", id_student=999)
    recs = recommender.recommend([1], k=5, context=context)
    assert recs == []
