"""Tests for base SASRec and EnhancedSASRec recommenders."""
from __future__ import annotations

import pytest
import pandas as pd
import torch

from src.models.sasrec import SASRecRecommender
from src.models.sasrec_enhanced import EnhancedSASRecRecommender
from src.models.base import Context


def test_sasrec_fit_recommend(small_splits: pd.DataFrame, context_aaa: Context) -> None:
    """Test that SASRecRecommender fits and recommends on CPU."""
    recommender = SASRecRecommender(
        embed_dim=8,
        num_blocks=1,
        num_heads=1,
        max_len=10,
        epochs=1,
        batch_size=2,
        val_sample=2,
        device="cpu",
        verbose=False,
    )
    recommender.fit(small_splits)
    
    # Recommend top-3 items
    history = [1, 2, 3]
    recs = recommender.recommend(history, k=3, context=context_aaa)
    
    assert isinstance(recs, list)
    assert len(recs) <= 3
    # Ensure history items are excluded
    for item in history:
        assert item not in recs


def test_enhanced_sasrec_fit_recommend(small_splits: pd.DataFrame, context_aaa: Context) -> None:
    """Test that EnhancedSASRecRecommender fits and recommends on CPU."""
    recommender = EnhancedSASRecRecommender(
        type_dim=4,
        gap_dim=4,
        module_dim=4,
        augment="none",
        embed_dim=8,
        num_blocks=1,
        num_heads=1,
        max_len=10,
        epochs=1,
        batch_size=2,
        val_sample=2,
        device="cpu",
        verbose=False,
    )
    # Fit model (which builds features dynamically)
    recommender.fit(small_splits)
    
    # Recommend top-3 items
    history = [1, 2, 3]
    recs = recommender.recommend(history, k=3, context=context_aaa)
    
    assert isinstance(recs, list)
    assert len(recs) <= 3
    for item in history:
        assert item not in recs


@pytest.mark.parametrize("augment", ["crop", "mask", "none"])
def test_enhanced_sasrec_augmentations(small_splits: pd.DataFrame, context_aaa: Context, augment: str) -> None:
    """Test that EnhancedSASRecRecommender trains under different augmentation options."""
    recommender = EnhancedSASRecRecommender(
        type_dim=4,
        gap_dim=4,
        module_dim=4,
        augment=augment,
        embed_dim=8,
        num_blocks=1,
        num_heads=1,
        max_len=10,
        epochs=1,
        batch_size=2,
        val_sample=2,
        device="cpu",
        verbose=False,
    )
    recommender.fit(small_splits)
    history = [1, 2, 3]
    recs = recommender.recommend(history, k=3, context=context_aaa)
    assert len(recs) <= 3
