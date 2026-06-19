"""Phase 4.1: Fix 1 - FairSR-style fairness-aware training for SASRec.

The standard SASRec loss (BPR pairwise) optimises accuracy only; it is blind to
whether the model is learning to serve different student groups equally well.

FairSR (Li, Hsu & Zhang 2022) adds a group-divergence penalty to the training
loss:

    L_total = L_BPR  +  lam * L_fair

where L_fair penalises the gap in mean predicted relevance scores between
privileged and unprivileged students within the same mini-batch.

Implementation:
  - Subclasses SASRecRecommender and overrides only _loss().
  - During fit() we attach per-session group labels to the training sequences, so
    that _loss() can identify which rows in the batch belong to each group.
  - The fairness attribute to use is configurable (fair_attr); defaults to IMD
    (the attribute with the largest observed gap from Phase 3).
  - lam is the fairness trade-off knob - sweep {0.0, 0.1, 0.5, 1.0, 2.0} for RQ2.

    lam = 0  ->  identical to plain SASRec (baseline Fix 1 point on the curve)
    lam > 0  ->  fairness penalty active; higher lam = more fairness, less accuracy
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.sasrec import SASRecRecommender


class FairSASRecRecommender(SASRecRecommender):
    """SASRec with FairSR group-divergence penalty in the loss."""

    def __init__(
        self,
        lam: float = 0.5,
        fair_attr: str = "imd_binary",          # column from sequences.parquet
        privileged_val: str = "advantaged",
        unprivileged_val: str = "disadvantaged",
        **sasrec_kwargs,
    ) -> None:
        super().__init__(**sasrec_kwargs)
        self.lam = lam
        self.fair_attr = fair_attr
        self.privileged_val = privileged_val
        self.unprivileged_val = unprivileged_val
        # Built in fit(): int8 tensor, 1=privileged, -1=unprivileged, 0=unknown
        self._group_labels: "torch.Tensor | None" = None  # noqa: F821

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"sasrec_fair_lam{self.lam}"

    # ------------------------------------------------------------------
    # Override fit() to build group labels alongside training sequences
    # ------------------------------------------------------------------
    def fit(self, splits_df: pd.DataFrame) -> "FairSASRecRecommender":
        import torch

        # --- build group label array parallel to train_seqs ---
        group_col = splits_df.get(self.fair_attr)
        labels = []
        for module, pres, hist, g in zip(
            splits_df["code_module"],
            splits_df["code_presentation"],
            splits_df["train_history"],
            group_col if group_col is not None else [None] * len(splits_df),
        ):
            if len(hist) >= 2:
                if g == self.privileged_val:
                    labels.append(1)
                elif g == self.unprivileged_val:
                    labels.append(-1)
                else:
                    labels.append(0)

        self._group_labels = torch.tensor(labels, dtype=torch.int8)

        # delegate to parent (which calls self._loss, now overridden)
        return super().fit(splits_df)

    # ------------------------------------------------------------------
    # Override _loss() to add the group-divergence penalty
    # ------------------------------------------------------------------
    def _loss(self, pos_logits, neg_logits, pad_mask, batch_rows=None):
        """BPR + lam * |mean(pos_score_priv) - mean(pos_score_unpriv)|."""
        import torch
        import torch.nn.functional as F

        valid = ~pad_mask
        bpr = -F.logsigmoid(pos_logits - neg_logits)
        bpr_loss = (bpr * valid).sum() / valid.sum().clamp(min=1)

        if self.lam == 0.0 or batch_rows is None or self._group_labels is None:
            return bpr_loss

        # Identify privileged / unprivileged rows in this batch.
        batch_labels = self._group_labels[batch_rows.cpu()]  # (B,)
        priv_mask = (batch_labels == 1).to(pos_logits.device)   # (B,)
        unpriv_mask = (batch_labels == -1).to(pos_logits.device)

        if priv_mask.sum() == 0 or unpriv_mask.sum() == 0:
            return bpr_loss

        # Mean positive logit per group (averaged over valid positions).
        # pos_logits: (B, L-1); valid: (B, L-1)
        def group_mean(mask):
            # mask: (B,) -> expand to (B, L-1) for position masking
            m = mask.unsqueeze(1) & valid       # (B, L-1)
            denom = m.sum().clamp(min=1)
            return (pos_logits * m).sum() / denom

        mean_priv = group_mean(priv_mask)
        mean_unpriv = group_mean(unpriv_mask)
        fair_loss = torch.abs(mean_priv - mean_unpriv)

        return bpr_loss + self.lam * fair_loss
