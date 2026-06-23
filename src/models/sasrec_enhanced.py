"""Feature-rich SASRec with multi-feature embeddings and data augmentation.

Extends :class:`~src.models.sasrec.SASRecRecommender` by injecting three
additional embedding channels into the sequence representation:

  1. **Activity-type embedding** — what *kind* of VLE resource the item is
     (forum, quiz, OU-collaborate, etc.).
  2. **Time-gap embedding** — how long since the previous activity (bucketed
     into same-day / next-day / 2-3 d / 4-7 d / 1-2 wk / 2+ wk).
  3. **Module embedding** — which OULAD module the session belongs to.

These are concatenated with the item embedding and linearly projected back to
``embed_dim`` before the positional encoding and Transformer blocks, so the
rest of the architecture (multi-head attention, feed-forward, layer-norm, and
the causal mask) stays identical.

Optionally the training loop can apply two lightweight data-augmentation
strategies:

  * **crop** — randomly crop a contiguous sub-sequence of the full training
    sequence (minimum length 3).
  * **mask** — randomly replace ~15 % of item indices with a ``[MASK]`` token
    (using ``n_items + 1``).

The ``_loss()`` signature is *unchanged*, so both the plain BPR loss from
:class:`SASRecRecommender` and the group-divergence loss from
:class:`~src.mitigation.fair_loss.FairSASRecRecommender` work without
modification.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.features import (
    FeatureBundle,
    N_TIME_BUCKETS,
    build_features,
)
from src.models.base import Context, Recommender
from src.models.sasrec import SASRecRecommender


class EnhancedSASRecRecommender(SASRecRecommender):
    """SASRec with activity-type, time-gap and module side-feature embeddings.

    Parameters
    ----------
    type_dim : int
        Dimensionality of the activity-type embedding (default 16).
    gap_dim : int
        Dimensionality of the time-gap embedding (default 8).
    module_dim : int
        Dimensionality of the module embedding (default 8).
    augment : str
        Training-time data augmentation: ``'none'``, ``'crop'`` or ``'mask'``
        (default ``'none'``).
    mask_prob : float
        Probability of masking each item when ``augment='mask'``
        (default 0.15).
    **sasrec_kwargs
        All remaining keyword arguments are forwarded to
        :class:`SASRecRecommender`.
    """

    name = "sasrec_enhanced"

    def __init__(
        self,
        type_dim: int = 16,
        gap_dim: int = 8,
        module_dim: int = 8,
        augment: str = "none",
        mask_prob: float = 0.15,
        **sasrec_kwargs,
    ) -> None:
        super().__init__(**sasrec_kwargs)
        self.type_dim = type_dim
        self.gap_dim = gap_dim
        self.module_dim = module_dim
        self.augment = augment
        self.mask_prob = mask_prob

        # Populated in fit():
        self._features: Optional[FeatureBundle] = None

    # ------------------------------------------------------------------
    # Network definition (overrides SASRecRecommender._build)
    # ------------------------------------------------------------------
    def _build(self):
        """Build the enhanced SASRec network with side-feature embeddings."""
        import torch
        import torch.nn as nn

        assert self._features is not None, "Call fit() first to populate _features"

        n_items = self._n_items
        max_len = self.max_len
        dim = self.embed_dim
        blocks = self.num_blocks
        heads = self.num_heads
        dropout = self.dropout

        n_types = self._features.n_types
        n_modules = self._features.n_modules
        type_dim = self.type_dim
        gap_dim = self.gap_dim
        module_dim = self.module_dim

        class EnhancedSASRec(nn.Module):
            """SASRec with concatenated side-feature embeddings.

            The item, activity-type, time-gap, and module embeddings are
            concatenated and projected to ``dim`` before adding positional
            embeddings and feeding into the Transformer stack.
            """

            def __init__(self):
                super().__init__()
                self.max_len = max_len

                # Core embeddings (same as base SASRec).
                # Mask token uses index n_items + 1 when augment='mask'.
                self.item_emb = nn.Embedding(n_items + 2, dim, padding_idx=0)
                self.pos_emb = nn.Embedding(max_len, dim)

                # Side-feature embeddings.
                self.type_emb = nn.Embedding(n_types, type_dim, padding_idx=0)
                self.gap_emb = nn.Embedding(N_TIME_BUCKETS, gap_dim, padding_idx=0)
                self.module_emb = nn.Embedding(n_modules, module_dim, padding_idx=0)

                # Projection from concatenated features back to dim.
                self.proj = nn.Linear(dim + type_dim + gap_dim + module_dim, dim)

                self.dropout_layer = nn.Dropout(dropout)

                # Transformer blocks (identical structure to base SASRec).
                self.layers = nn.ModuleList([
                    nn.ModuleDict({
                        "ln1": nn.LayerNorm(dim),
                        "attn": nn.MultiheadAttention(
                            dim, heads, dropout=dropout, batch_first=True,
                        ),
                        "ln2": nn.LayerNorm(dim),
                        "ff1": nn.Linear(dim, dim),
                        "ff2": nn.Linear(dim, dim),
                    })
                    for _ in range(blocks)
                ])
                self.last_ln = nn.LayerNorm(dim)

            def seq_repr(
                self,
                seq: "torch.Tensor",
                type_ids: "torch.Tensor | None" = None,
                gap_ids: "torch.Tensor | None" = None,
                module_ids: "torch.Tensor | None" = None,
            ) -> "torch.Tensor":
                """Compute contextual sequence representations.

                Parameters
                ----------
                seq : Tensor, shape (B, L)
                    Left-padded item-index sequences.
                type_ids : Tensor, shape (B, L), optional
                    Activity-type indices (0 = padding).
                gap_ids : Tensor, shape (B, L), optional
                    Time-gap bucket indices (0 = padding).
                module_ids : Tensor, shape (B,), optional
                    Module index per session (broadcast across L).

                Returns
                -------
                Tensor, shape (B, L, dim)
                """
                import torch as _torch

                B, L = seq.shape
                device = seq.device

                # Item embedding.
                x_item = self.item_emb(seq)  # (B, L, dim)

                # Side-feature embeddings (fall back to zeros when absent).
                if type_ids is not None:
                    x_type = self.type_emb(type_ids)  # (B, L, type_dim)
                else:
                    x_type = _torch.zeros(B, L, type_dim, device=device)

                if gap_ids is not None:
                    x_gap = self.gap_emb(gap_ids)  # (B, L, gap_dim)
                else:
                    x_gap = _torch.zeros(B, L, gap_dim, device=device)

                if module_ids is not None:
                    # module_ids: (B,) -> (B, 1, module_dim) -> broadcast to (B, L, module_dim)
                    x_mod = self.module_emb(module_ids).unsqueeze(1).expand(B, L, -1)
                else:
                    x_mod = _torch.zeros(B, L, module_dim, device=device)

                # Concatenate and project to dim.
                concat = _torch.cat([x_item, x_type, x_gap, x_mod], dim=-1)
                x = self.proj(concat)  # (B, L, dim)

                # Add positional embeddings.
                positions = _torch.arange(L, device=device).unsqueeze(0).expand(B, L)
                x = x + self.pos_emb(positions)
                x = self.dropout_layer(x)

                # Causal Transformer blocks (identical to base SASRec).
                pad_mask = seq == 0  # (B, L)
                causal = _torch.triu(
                    _torch.ones(L, L, device=device, dtype=_torch.bool), diagonal=1,
                )
                for layer in self.layers:
                    h = layer["ln1"](x)
                    attn_out, _ = layer["attn"](
                        h, h, h, attn_mask=causal,
                        key_padding_mask=pad_mask, need_weights=False,
                    )
                    x = x + attn_out
                    h = layer["ln2"](x)
                    h = layer["ff2"](_torch.relu(layer["ff1"](h)))
                    x = x + h

                return self.last_ln(x)  # (B, L, dim)

            def forward(
                self,
                seq: "torch.Tensor",
                type_ids: "torch.Tensor | None" = None,
                gap_ids: "torch.Tensor | None" = None,
                module_ids: "torch.Tensor | None" = None,
            ) -> "torch.Tensor":
                """Forward pass delegating to :meth:`seq_repr`."""
                return self.seq_repr(seq, type_ids, gap_ids, module_ids)

        return EnhancedSASRec()

    # ------------------------------------------------------------------
    # Feature-sequence helpers
    # ------------------------------------------------------------------
    def _build_type_seq(self, item_seq: list[int]) -> list[int]:
        """Map a list of item indices to activity-type indices."""
        assert self._features is not None
        tm = self._features.type_map
        return [tm.get(i, 0) for i in item_seq]

    def _build_gap_seq(
        self,
        session_key: Tuple[int, str, str],
        seq_len: int,
    ) -> list[int]:
        """Retrieve (and truncate/pad) the time-gap bucket sequence."""
        assert self._features is not None
        gaps = self._features.gap_sequences.get(session_key, [])
        if not gaps:
            return [1] * seq_len  # default: all same-day
        # Align to the last seq_len entries (matching left-pad truncation).
        gaps = gaps[-seq_len:]
        if len(gaps) < seq_len:
            gaps = [0] * (seq_len - len(gaps)) + gaps  # left-pad with 0
        return gaps

    def _left_pad_feature(self, seq: list[int]) -> list[int]:
        """Left-pad a feature sequence to ``max_len`` (mirror of ``_left_pad``)."""
        seq = list(seq)[-self.max_len:]
        pad = [0] * (self.max_len - len(seq))
        return pad + seq

    # ------------------------------------------------------------------
    # Data augmentation
    # ------------------------------------------------------------------
    def _augment_crop(
        self,
        seq: "torch.Tensor",
        type_ids: "torch.Tensor",
        gap_ids: "torch.Tensor",
        rng: np.random.RandomState,
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Randomly crop each sequence in the batch (in-place on clones).

        The cropped region is a contiguous sub-sequence with minimum length 3.
        Cropping is applied per-row to a *clone* of the batch tensors.
        """
        import torch

        B, L = seq.shape
        seq = seq.clone()
        type_ids = type_ids.clone()
        gap_ids = gap_ids.clone()

        for b in range(B):
            # Find real (non-padding) start.
            non_pad = (seq[b] != 0).nonzero(as_tuple=True)[0]
            if len(non_pad) < 3:
                continue
            real_start = int(non_pad[0])
            real_len = L - real_start
            if real_len <= 3:
                continue
            crop_len = rng.randint(3, real_len + 1)  # [3, real_len]
            # Random start within the real portion.
            max_start = real_start + real_len - crop_len
            crop_start = rng.randint(real_start, max_start + 1)

            # Zero out everything outside the crop window, then shift right.
            cropped_seq = seq[b, crop_start:crop_start + crop_len].clone()
            cropped_type = type_ids[b, crop_start:crop_start + crop_len].clone()
            cropped_gap = gap_ids[b, crop_start:crop_start + crop_len].clone()

            seq[b] = 0
            type_ids[b] = 0
            gap_ids[b] = 0
            # Right-align (left-pad with 0).
            seq[b, L - crop_len:] = cropped_seq
            type_ids[b, L - crop_len:] = cropped_type
            gap_ids[b, L - crop_len:] = cropped_gap

        return seq, type_ids, gap_ids

    def _augment_mask(
        self,
        seq: "torch.Tensor",
        rng: np.random.RandomState,
    ) -> "torch.Tensor":
        """Randomly replace items with [MASK] token (n_items + 1).

        Only non-padding positions are eligible.  The last real position is
        never masked (it is the prediction target boundary).
        """
        import torch

        seq = seq.clone()
        B, L = seq.shape
        mask_token = self._n_items + 1

        for b in range(B):
            non_pad = (seq[b] != 0).nonzero(as_tuple=True)[0]
            if len(non_pad) < 2:
                continue
            # Exclude the very last real position from masking.
            eligible = non_pad[:-1]
            n_mask = max(1, int(len(eligible) * self.mask_prob))
            chosen = rng.choice(eligible.cpu().numpy(), size=n_mask, replace=False)
            seq[b, chosen] = mask_token

        return seq

    # ------------------------------------------------------------------
    # Training (overrides SASRecRecommender.fit)
    # ------------------------------------------------------------------
    def fit(self, splits_df: pd.DataFrame) -> "EnhancedSASRecRecommender":
        """Train the enhanced SASRec with side features and optional augmentation.

        This method mirrors the parent's ``fit()`` closely but adds three
        parallel tensors (``type_data``, ``gap_data``, ``module_data``) that
        are sliced alongside the item sequences during training and passed to
        the enhanced network.
        """
        import torch

        torch.manual_seed(self.seed)
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        device = self._device

        # ---- compute n_items (same logic as parent) ---------------------
        self._n_items = int(
            splits_df["train_history"]
            .map(lambda h: max(h) if len(h) else 0)
            .max()
        )
        if "seq" in splits_df.columns:
            self._n_items = max(
                self._n_items, int(splits_df["seq"].map(max).max())
            )

        # ---- precompute side features -----------------------------------
        self._features = build_features(splits_df)

        # ---- build training data (items + features) ---------------------
        valid: dict[tuple[str, str], set] = defaultdict(set)
        train_seqs: list[list[int]] = []
        train_types: list[list[int]] = []
        train_gaps: list[list[int]] = []
        train_modules: list[int] = []

        for row in splits_df.itertuples():
            hist = row.train_history
            module = row.code_module
            pres = row.code_presentation
            student = row.id_student

            valid[(module, pres)].update(hist)

            if len(hist) < 2:
                continue

            # Item sequence (left-padded).
            train_seqs.append(self._left_pad(hist))

            # Activity-type sequence (parallel to items).
            type_seq = self._build_type_seq(hist)
            train_types.append(self._left_pad_feature(type_seq))

            # Time-gap sequence (parallel to items).
            session_key = (int(student), str(module), str(pres))
            gap_seq = self._build_gap_seq(session_key, len(hist))
            train_gaps.append(self._left_pad_feature(gap_seq))

            # Module index (scalar per session).
            train_modules.append(self._features.module_map.get(module, 0))

        self._valid_items = {
            k: np.fromiter(v, dtype=np.int64) for k, v in valid.items()
        }

        # ---- validation set (same as parent, uses base forward) ---------
        rng_np = np.random.RandomState(self.seed)
        n_all = len(splits_df)
        sample_idx = (
            rng_np.choice(n_all, size=min(self.val_sample, n_all), replace=False)
            if self.val_sample
            else np.arange(n_all)
        )
        vi = splits_df["val_input"].to_numpy()
        vt = splits_df["val_target"].to_numpy()
        val_seq = torch.tensor(
            np.array([self._left_pad(vi[j]) for j in sample_idx]),
            dtype=torch.long,
        )
        val_target = torch.tensor(
            [int(vt[j]) for j in sample_idx], dtype=torch.long,
        )

        # Build validation feature tensors for early-stopping scorer.
        val_types_list = []
        val_gaps_list = []
        val_modules_list = []
        for j in sample_idx:
            row = splits_df.iloc[j]
            hist_val = row["val_input"]
            val_types_list.append(
                self._left_pad_feature(self._build_type_seq(hist_val))
            )
            sk = (int(row["id_student"]), str(row["code_module"]),
                  str(row["code_presentation"]))
            val_gaps_list.append(
                self._left_pad_feature(self._build_gap_seq(sk, len(hist_val)))
            )
            val_modules_list.append(
                self._features.module_map.get(row["code_module"], 0)
            )
        val_type_data = torch.tensor(np.array(val_types_list), dtype=torch.long)
        val_gap_data = torch.tensor(np.array(val_gaps_list), dtype=torch.long)
        val_module_data = torch.tensor(val_modules_list, dtype=torch.long)

        # ---- tensors ----------------------------------------------------
        data = torch.tensor(np.array(train_seqs), dtype=torch.long)
        type_data = torch.tensor(np.array(train_types), dtype=torch.long)
        gap_data = torch.tensor(np.array(train_gaps), dtype=torch.long)
        module_data = torch.tensor(train_modules, dtype=torch.long)

        # ---- build & optimiser ------------------------------------------
        self._net = self._build().to(device)
        opt = torch.optim.Adam(
            self._net.parameters(),
            lr=self.lr,
            betas=(0.9, 0.98),
            weight_decay=self.weight_decay,
        )

        # ---- training loop ----------------------------------------------
        n = data.shape[0]
        best_val = -1.0
        best_state = None
        epochs_no_improve = 0
        aug_rng = np.random.RandomState(self.seed + 42)

        for epoch in range(self.epochs):
            self._net.train()
            perm = torch.randperm(n)
            total = 0.0

            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                seq = data[idx].to(device)          # (B, L)
                t_ids = type_data[idx].to(device)   # (B, L)
                g_ids = gap_data[idx].to(device)    # (B, L)
                m_ids = module_data[idx].to(device) # (B,)

                # ---- data augmentation ----------------------------------
                if self.augment == "crop":
                    seq, t_ids, g_ids = self._augment_crop(
                        seq, t_ids, g_ids, aug_rng,
                    )
                elif self.augment == "mask":
                    seq = self._augment_mask(seq, aug_rng)

                # ---- teacher-forcing: inp -> pos (next items) -----------
                inp = seq[:, :-1]
                pos = seq[:, 1:]
                t_inp = t_ids[:, :-1]
                g_inp = g_ids[:, :-1]

                pad_mask = pos == 0

                # Sampled negatives (avoid 0 = padding).
                neg = torch.randint(
                    1, self._n_items + 1, pos.shape, device=device,
                )

                repr_ = self._net(inp, t_inp, g_inp, m_ids)  # (B, L-1, dim)
                pos_emb = self._net.item_emb(pos)
                neg_emb = self._net.item_emb(neg)
                pos_logits = (repr_ * pos_emb).sum(-1)
                neg_logits = (repr_ * neg_emb).sum(-1)

                loss = self._loss(pos_logits, neg_logits, pad_mask,
                                  batch_rows=idx)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss) * len(idx)

            # ---- validation (early stopping) ----------------------------
            val_recall = self._val_recall_enhanced(
                val_seq, val_target,
                val_type_data, val_gap_data, val_module_data,
            )
            if self.verbose:
                print(
                    f"  [sasrec_enhanced] epoch {epoch + 1:>3}/{self.epochs}  "
                    f"loss={total / n:.4f}  val_recall@10={val_recall:.4f}"
                    f"{'  *' if val_recall > best_val else ''}",
                    flush=True,
                )

            if val_recall > best_val:
                best_val = val_recall
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self._net.state_dict().items()
                }
                self.best_epoch_ = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    if self.verbose:
                        print(
                            f"  [sasrec_enhanced] early stop at epoch {epoch + 1} "
                            f"(best={best_val:.4f} @ epoch {self.best_epoch_})",
                            flush=True,
                        )
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self.best_val_recall_ = best_val
        return self

    # ------------------------------------------------------------------
    # Validation scorer with side features
    # ------------------------------------------------------------------
    def _val_recall_enhanced(
        self,
        val_seq: "torch.Tensor",
        val_target: "torch.Tensor",
        val_type_data: "torch.Tensor",
        val_gap_data: "torch.Tensor",
        val_module_data: "torch.Tensor",
        k: int = 10,
        batch: int = 512,
    ) -> float:
        """Recall@K on validation set, passing side features to the network."""
        import torch

        self._net.eval()
        hits = 0
        with torch.no_grad():
            emb = self._net.item_emb.weight  # (n_items+2, dim)
            for i in range(0, val_seq.shape[0], batch):
                seq = val_seq[i : i + batch].to(self._device)
                tgt = val_target[i : i + batch].to(self._device)
                t_ids = val_type_data[i : i + batch].to(self._device)
                g_ids = val_gap_data[i : i + batch].to(self._device)
                m_ids = val_module_data[i : i + batch].to(self._device)

                repr_ = self._net(seq, t_ids, g_ids, m_ids)[:, -1, :]
                logits = repr_ @ emb.t()
                logits[:, 0] = float("-inf")     # padding
                logits.scatter_(1, seq, float("-inf"))  # exclude seen
                topk = logits.topk(k, dim=1).indices
                hits += (topk == tgt.unsqueeze(1)).any(1).sum().item()

        return hits / val_seq.shape[0]

    # ------------------------------------------------------------------
    # Inference (overrides SASRecRecommender.recommend)
    # ------------------------------------------------------------------
    def recommend(
        self, history: list[int], k: int, context: Context,
    ) -> list[int]:
        """Rank candidate items using the enhanced network with side features.

        Builds activity-type and time-gap feature sequences for the given
        history on the fly, then scores all valid candidates for the session's
        presentation.
        """
        import torch

        candidates = self._valid_items.get(context.key)
        if candidates is None or len(candidates) == 0 or not history:
            return []

        self._net.eval()
        with torch.no_grad():
            # Item sequence.
            seq = torch.tensor(
                [self._left_pad(history)], dtype=torch.long, device=self._device,
            )

            # Activity-type ids.
            type_seq = self._build_type_seq(history)
            type_ids = torch.tensor(
                [self._left_pad_feature(type_seq)],
                dtype=torch.long, device=self._device,
            )

            # Time-gap ids.
            session_key = (
                int(context.id_student) if context.id_student is not None else 0,
                context.code_module,
                context.code_presentation,
            )
            gap_seq = self._build_gap_seq(session_key, len(history))
            gap_ids = torch.tensor(
                [self._left_pad_feature(gap_seq)],
                dtype=torch.long, device=self._device,
            )

            # Module id.
            assert self._features is not None
            mod_idx = self._features.module_map.get(context.code_module, 0)
            module_ids = torch.tensor(
                [mod_idx], dtype=torch.long, device=self._device,
            )

            # Compute representation and score candidates.
            repr_ = self._net(seq, type_ids, gap_ids, module_ids)[:, -1, :]
            cand = torch.tensor(
                candidates, dtype=torch.long, device=self._device,
            )
            cand_emb = self._net.item_emb(cand)
            scores = (repr_ @ cand_emb.t()).squeeze(0)

            seen = set(history)
            n = min(k + len(seen), len(candidates))
            top = torch.topk(scores, n).indices.cpu().numpy()

        out: list[int] = []
        for j in top:
            item = int(candidates[j])
            if item not in seen:
                out.append(item)
                if len(out) == k:
                    break
        return out
