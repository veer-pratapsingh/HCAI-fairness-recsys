"""Adversarial debiasing for SASRec.

Adds a gradient reversal adversary that tries to predict the protected
attribute (e.g. imd_binary) from item representations.  The recommender
learns representations that are both accurate AND group-invariant.

    L_total = L_BPR  −  lambda_adv · L_adversary

The gradient reversal layer (GRL) flips the gradient sign during backprop,
so the recommender learns to *fool* the adversary while the adversary (with
its own optimizer) learns to classify the protected attribute.

Architecture overview::

    ┌──────────────────┐      ┌──────────────┐
    │ SASRec backbone  │─────►│  BPR scorer  │  → L_BPR
    │ (item & pos emb, │      └──────────────┘
    │  transformer)    │
    │                  │─GRL──►┌──────────────┐
    └──────────────────┘       │  Adversary   │  → L_adv (CE)
                               │ (MLP head)   │
                               └──────────────┘

Sweep values for lambda_adv: {0.01, 0.05, 0.1, 0.5}.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.data.protected import PROTECTED_ATTRS
from src.models.base import Context, Recommender
from src.models.sasrec import SASRecRecommender


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------
class _GradientReversalFunction:
    """Autograd function that reverses gradients in the backward pass.

    This is defined as a nested class to avoid issues with torch not being
    imported at module level (mirrors the lazy-import convention of SASRec).
    The actual ``torch.autograd.Function`` subclass is created lazily inside
    ``_get_grl_fn()``.
    """


def _get_grl_fn():
    """Return the GRL ``torch.autograd.Function`` subclass (lazy import)."""
    import torch

    class GradientReversalFunction(torch.autograd.Function):
        """Reverses gradients by multiplying with −λ in the backward pass."""

        @staticmethod
        def forward(ctx, x, lambda_: float):  # type: ignore[override]
            ctx.lambda_ = lambda_
            return x.clone()

        @staticmethod
        def backward(ctx, grad_output):  # type: ignore[override]
            return -ctx.lambda_ * grad_output, None

    return GradientReversalFunction


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class AdversarialSASRecRecommender(SASRecRecommender):
    """SASRec with an adversarial head for group-invariant representations.

    Parameters
    ----------
    lambda_adv : float
        Weight of the adversarial loss in the combined objective.  Higher
        values push the backbone toward group-invariant embeddings at the
        cost of some recommendation accuracy.
    fair_attr : str
        Column in ``splits_df`` that holds the binary protected-attribute
        label (e.g. ``"imd_binary"``).
    privileged_val, unprivileged_val : str
        The two group-label strings.  Sessions not matching either value are
        excluded from the adversarial loss (but still trained for BPR).
    adv_lr : float
        Learning rate for the adversary optimizer (separate from the main
        optimizer).
    **sasrec_kwargs
        All remaining keyword arguments are forwarded to
        :class:`SASRecRecommender`.
    """

    def __init__(
        self,
        lambda_adv: float = 0.1,
        fair_attr: str = "imd_binary",
        privileged_val: str = "advantaged",
        unprivileged_val: str = "disadvantaged",
        adv_lr: float = 1e-3,
        **sasrec_kwargs,
    ) -> None:
        super().__init__(**sasrec_kwargs)
        self.lambda_adv = lambda_adv
        self.fair_attr = fair_attr
        self.privileged_val = privileged_val
        self.unprivileged_val = unprivileged_val
        self.adv_lr = adv_lr

        # Built in fit():
        # int8 tensor — 1 = privileged, 0 = unprivileged, -1 = unknown/skip
        self._group_labels: "torch.Tensor | None" = None  # noqa: F821
        self._adversary = None
        self._grl_fn = None

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"sasrec_adv_lam{self.lambda_adv}"

    # ------------------------------------------------------------------
    # Network construction — adds adversary head
    # ------------------------------------------------------------------
    def _build(self):
        """Build the base SASRec network *and* the adversary MLP head."""
        import torch.nn as nn

        net = super()._build()

        # Adversary: classifies the last-position representation → 2 groups
        self._adversary = nn.Sequential(
            nn.Linear(self.embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )
        return net

    # ------------------------------------------------------------------
    # Training — custom loop with GRL + dual optimizers
    # ------------------------------------------------------------------
    def fit(self, splits_df: pd.DataFrame) -> "AdversarialSASRecRecommender":  # noqa: C901
        """Train SASRec with adversarial debiasing.

        The training loop mirrors :meth:`SASRecRecommender.fit` but adds:
        1. A gradient reversal layer between the backbone and the adversary.
        2. A separate optimizer for the adversary head.
        3. The combined loss ``L_BPR − lambda_adv · L_adv``.
        """
        import torch
        import torch.nn.functional as F

        torch.manual_seed(self.seed)
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        device = self._device

        # ---------- data prep (same as base) ----------
        self._n_items = int(
            splits_df["train_history"]
            .map(lambda h: max(h) if len(h) else 0)
            .max()
        )
        if "seq" in splits_df:
            self._n_items = max(self._n_items, int(splits_df["seq"].map(max).max()))

        valid: dict[tuple[str, str], set] = defaultdict(set)
        train_seqs: list[list[int]] = []
        group_labels: list[int] = []

        group_col = splits_df.get(self.fair_attr)
        for module, pres, hist, g in zip(
            splits_df["code_module"],
            splits_df["code_presentation"],
            splits_df["train_history"],
            group_col if group_col is not None else [None] * len(splits_df),
        ):
            valid[(module, pres)].update(hist)
            if len(hist) >= 2:
                train_seqs.append(self._left_pad(hist))
                if g == self.privileged_val:
                    group_labels.append(1)
                elif g == self.unprivileged_val:
                    group_labels.append(0)
                else:
                    group_labels.append(-1)  # skip in adversarial loss

        self._valid_items = {
            k: np.fromiter(v, dtype=np.int64) for k, v in valid.items()
        }
        self._group_labels = torch.tensor(group_labels, dtype=torch.int8)

        # Validation set for early stopping
        rng = np.random.RandomState(self.seed)
        n_all = len(splits_df)
        sample_idx = (
            rng.choice(n_all, size=min(self.val_sample, n_all), replace=False)
            if self.val_sample
            else np.arange(n_all)
        )
        vi = splits_df["val_input"].to_numpy()
        vt = splits_df["val_target"].to_numpy()
        val_seq = torch.tensor(
            np.array([self._left_pad(vi[j]) for j in sample_idx]), dtype=torch.long
        )
        val_target = torch.tensor(
            [int(vt[j]) for j in sample_idx], dtype=torch.long
        )

        data = torch.tensor(np.array(train_seqs), dtype=torch.long)

        # ---------- build & optimizers ----------
        self._net = self._build().to(device)
        self._adversary = self._adversary.to(device)
        self._grl_fn = _get_grl_fn()

        # Main optimizer covers backbone only
        opt_main = torch.optim.Adam(
            self._net.parameters(),
            lr=self.lr,
            betas=(0.9, 0.98),
            weight_decay=self.weight_decay,
        )
        # Adversary optimizer — separate so we can control its learning
        opt_adv = torch.optim.Adam(
            self._adversary.parameters(),
            lr=self.adv_lr,
        )

        n = data.shape[0]
        best_val = -1.0
        best_state: dict | None = None
        best_adv_state: dict | None = None
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self._net.train()
            self._adversary.train()
            perm = torch.randperm(n)
            total_loss = 0.0

            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                seq = data[idx].to(device)  # (B, L)
                inp = seq[:, :-1]
                pos = seq[:, 1:]  # next-item targets
                pad_mask = pos == 0
                neg = torch.randint(1, self._n_items + 1, pos.shape, device=device)

                repr_ = self._net(inp)  # (B, L-1, dim)
                pos_emb = self._net.item_emb(pos)
                neg_emb = self._net.item_emb(neg)
                pos_logits = (repr_ * pos_emb).sum(-1)
                neg_logits = (repr_ * neg_emb).sum(-1)

                # --- BPR loss ---
                bpr_valid = ~pad_mask
                bpr_loss = -F.logsigmoid(pos_logits - neg_logits)
                bpr_loss = (bpr_loss * bpr_valid).sum() / bpr_valid.sum().clamp(min=1)

                # --- Adversarial loss ---
                adv_loss = torch.tensor(0.0, device=device)
                batch_labels = self._group_labels[idx.cpu()]  # (B,)
                known_mask = batch_labels >= 0  # exclude unknown (-1)

                if known_mask.sum() > 0:
                    # Last-position representation for each sequence
                    last_repr = repr_[:, -1, :]  # (B, dim)
                    known_repr = last_repr[known_mask.to(device)]  # (K, dim)
                    known_targets = batch_labels[known_mask].long().to(device)  # (K,)

                    # Gradient reversal: backbone gradients are reversed
                    reversed_repr = self._grl_fn.apply(known_repr, self.lambda_adv)
                    adv_logits = self._adversary(reversed_repr)  # (K, 2)
                    adv_loss = F.cross_entropy(adv_logits, known_targets)

                # Combined: backbone minimises BPR while GRL fools adversary
                loss = bpr_loss + adv_loss

                opt_main.zero_grad()
                opt_adv.zero_grad()
                loss.backward()
                opt_main.step()
                opt_adv.step()

                total_loss += float(bpr_loss) * len(idx)

            val_recall = self._val_recall(val_seq, val_target)
            if self.verbose:
                print(
                    f"  [sasrec_adv] epoch {epoch + 1:>3}/{self.epochs}  "
                    f"loss={total_loss / n:.4f}  val_recall@10={val_recall:.4f}"
                    f"{'  *' if val_recall > best_val else ''}",
                    flush=True,
                )

            if val_recall > best_val:
                best_val = val_recall
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self._net.state_dict().items()
                }
                best_adv_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self._adversary.state_dict().items()
                }
                self.best_epoch_ = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    if self.verbose:
                        print(
                            f"  [sasrec_adv] early stop at epoch {epoch + 1} "
                            f"(best={best_val:.4f} @ epoch {self.best_epoch_})",
                            flush=True,
                        )
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        if best_adv_state is not None:
            self._adversary.load_state_dict(best_adv_state)
        self.best_val_recall_ = best_val
        return self

    # recommend() is inherited unchanged from SASRecRecommender
