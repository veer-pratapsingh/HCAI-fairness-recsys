"""Phase 2.3: SASRec — the self-attentive sequential recommender (centerpiece).

Kang & McAuley (2018). A causal (left-to-right) Transformer over the student's
activity sequence; the representation after the last position scores all items
for next-item prediction.

Spec from the slides: 2 self-attention blocks, 50-dim embeddings.

Design choices for this project:
  - Sequences are long (median ~216); we truncate to the last `max_len` items.
  - Training target = next item at each position (standard SASRec teacher forcing)
    with one sampled negative per position (BPR-style pairwise loss).
  - The loss is deliberately a swappable method (`_loss`) so Phase 4's fair-loss
    (Fix 1) can subclass and add the group-divergence penalty without touching
    the training loop.
  - Candidates at inference are restricted to the session's presentation (D1).

This module imports torch lazily so the data/eval phases work without it.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.models.base import Context, Recommender


class _SASRecNet:
    """Built lazily inside fit() once torch is importable (see _build)."""


class SASRecRecommender(Recommender):
    name = "sasrec"

    def __init__(
        self,
        max_len: int = 200,
        embed_dim: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        patience: int = 5,
        val_sample: int = 4000,
        batch_size: int = 128,
        seed: int = 0,
        device: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs          # max epochs; early stopping usually halts sooner
        self.patience = patience      # epochs of no val improvement before stopping
        self.val_sample = val_sample  # sessions used for the early-stopping signal
        self.batch_size = batch_size
        self.seed = seed
        self._device = device
        self.verbose = verbose
        self._net = None
        self._n_items = 0
        self._valid_items: dict[tuple[str, str], np.ndarray] = {}
        self.best_epoch_ = None
        self.best_val_recall_ = None

    # ---- network definition (built lazily so torch stays an optional dep) ----
    def _build(self):
        import torch
        import torch.nn as nn

        class SASRec(nn.Module):
            def __init__(self, n_items, max_len, dim, blocks, heads, dropout):
                super().__init__()
                self.max_len = max_len
                self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
                self.pos_emb = nn.Embedding(max_len, dim)
                self.dropout = nn.Dropout(dropout)
                self.layers = nn.ModuleList([
                    nn.ModuleDict({
                        "ln1": nn.LayerNorm(dim),
                        "attn": nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                        "ln2": nn.LayerNorm(dim),
                        "ff1": nn.Linear(dim, dim),
                        "ff2": nn.Linear(dim, dim),
                    }) for _ in range(blocks)
                ])
                self.last_ln = nn.LayerNorm(dim)

            def seq_repr(self, seq):
                # seq: (B, L) padded with 0 on the LEFT (recent items at the end)
                import torch
                B, L = seq.shape
                positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
                x = self.item_emb(seq) + self.pos_emb(positions)
                x = self.dropout(x)
                pad_mask = seq == 0  # (B, L) True where padding
                causal = torch.triu(
                    torch.ones(L, L, device=seq.device, dtype=torch.bool), diagonal=1
                )
                for layer in self.layers:
                    h = layer["ln1"](x)
                    attn_out, _ = layer["attn"](
                        h, h, h, attn_mask=causal, key_padding_mask=pad_mask, need_weights=False
                    )
                    x = x + attn_out
                    h = layer["ln2"](x)
                    h = layer["ff2"](torch.relu(layer["ff1"](h)))
                    x = x + h
                return self.last_ln(x)  # (B, L, dim)

            def forward(self, seq):
                return self.seq_repr(seq)

        net = SASRec(self._n_items, self.max_len, self.embed_dim,
                     self.num_blocks, self.num_heads, self.dropout)
        return net

    # ---------- data prep ----------
    def _left_pad(self, seq):
        seq = list(seq)[-self.max_len:]
        pad = [0] * (self.max_len - len(seq))
        return pad + seq

    # ---------- loss (swappable; Fix 1 overrides this) ----------
    def _loss(self, pos_logits, neg_logits, pad_mask, batch_rows=None):
        """BPR loss over valid (non-pad) positions. Returns a scalar tensor."""
        import torch
        import torch.nn.functional as F
        valid = ~pad_mask
        loss = -F.logsigmoid(pos_logits - neg_logits)
        return (loss * valid).sum() / valid.sum().clamp(min=1)

    # ---------- validation scorer (early-stopping signal) ----------
    def _val_recall(self, val_seq, val_target, k: int = 10, batch: int = 512) -> float:
        """Batched Recall@K on held-out val targets (scored over all items).

        Used only for model selection / early stopping, so we score globally
        (fast) rather than per-presentation; this is a consistent proxy.
        """
        import torch

        self._net.eval()
        hits = 0
        with torch.no_grad():
            emb = self._net.item_emb.weight                 # (n_items+1, dim)
            for i in range(0, val_seq.shape[0], batch):
                seq = val_seq[i:i + batch].to(self._device)
                tgt = val_target[i:i + batch].to(self._device)
                repr_ = self._net(seq)[:, -1, :]            # (B, dim)
                logits = repr_ @ emb.t()                    # (B, n_items+1)
                logits[:, 0] = float("-inf")                # padding
                logits.scatter_(1, seq, float("-inf"))      # exclude already-seen
                topk = logits.topk(k, dim=1).indices        # (B, k)
                hits += (topk == tgt.unsqueeze(1)).any(1).sum().item()
        return hits / val_seq.shape[0]

    # ---------- training ----------
    def fit(self, splits_df: pd.DataFrame) -> "SASRecRecommender":
        import torch

        torch.manual_seed(self.seed)
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        device = self._device

        self._n_items = int(splits_df["train_history"].map(
            lambda h: max(h) if len(h) else 0).max())
        # also account for val/test targets being valid item ids
        self._n_items = max(self._n_items, int(splits_df["seq"].map(max).max())
                            if "seq" in splits_df else self._n_items)

        valid: dict[tuple[str, str], set] = defaultdict(set)
        train_seqs = []
        for module, pres, hist in zip(
            splits_df["code_module"], splits_df["code_presentation"], splits_df["train_history"],
        ):
            valid[(module, pres)].update(hist)
            if len(hist) >= 2:
                train_seqs.append(self._left_pad(hist))
        self._valid_items = {k: np.fromiter(v, dtype=np.int64) for k, v in valid.items()}

        # Validation set for early stopping: predict val_target from val_input.
        rng = np.random.RandomState(self.seed)
        n_all = len(splits_df)
        sample_idx = (rng.choice(n_all, size=min(self.val_sample, n_all), replace=False)
                      if self.val_sample else np.arange(n_all))
        vi = splits_df["val_input"].to_numpy()
        vt = splits_df["val_target"].to_numpy()
        val_seq = torch.tensor(
            np.array([self._left_pad(vi[j]) for j in sample_idx]), dtype=torch.long)
        val_target = torch.tensor([int(vt[j]) for j in sample_idx], dtype=torch.long)

        data = torch.tensor(np.array(train_seqs), dtype=torch.long)
        self._net = self._build().to(device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr,
                               betas=(0.9, 0.98), weight_decay=self.weight_decay)

        n = data.shape[0]
        best_val = -1.0
        best_state = None
        epochs_no_improve = 0
        for epoch in range(self.epochs):
            self._net.train()
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                seq = data[idx].to(device)               # (B, L)
                inp = seq[:, :-1]
                pos = seq[:, 1:]                           # next-item targets
                pad_mask = pos == 0
                # sampled negatives (avoid 0/padding)
                neg = torch.randint(1, self._n_items + 1, pos.shape, device=device)

                repr_ = self._net(inp)                     # (B, L-1, dim)
                pos_emb = self._net.item_emb(pos)
                neg_emb = self._net.item_emb(neg)
                pos_logits = (repr_ * pos_emb).sum(-1)
                neg_logits = (repr_ * neg_emb).sum(-1)
                loss = self._loss(pos_logits, neg_logits, pad_mask, batch_rows=idx)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss) * len(idx)

            val_recall = self._val_recall(val_seq, val_target)
            if self.verbose:
                print(f"  [sasrec] epoch {epoch + 1:>3}/{self.epochs}  "
                      f"loss={total / n:.4f}  val_recall@10={val_recall:.4f}"
                      f"{'  *' if val_recall > best_val else ''}", flush=True)

            if val_recall > best_val:
                best_val = val_recall
                best_state = {k: v.detach().cpu().clone() for k, v in self._net.state_dict().items()}
                self.best_epoch_ = epoch + 1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    if self.verbose:
                        print(f"  [sasrec] early stop at epoch {epoch + 1} "
                              f"(best={best_val:.4f} @ epoch {self.best_epoch_})", flush=True)
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)          # restore best-val weights
        self.best_val_recall_ = best_val
        return self

    # ---------- inference ----------
    def recommend(self, history: list[int], k: int, context: Context) -> list[int]:
        import torch

        candidates = self._valid_items.get(context.key)
        if candidates is None or len(candidates) == 0 or not history:
            return []
        self._net.eval()
        with torch.no_grad():
            seq = torch.tensor([self._left_pad(history)], dtype=torch.long, device=self._device)
            repr_ = self._net(seq)[:, -1, :]              # (1, dim) last position
            cand = torch.tensor(candidates, dtype=torch.long, device=self._device)
            cand_emb = self._net.item_emb(cand)           # (C, dim)
            scores = (repr_ @ cand_emb.t()).squeeze(0)    # (C,)
            seen = set(history)
            n = min(k + len(seen), len(candidates))
            top = torch.topk(scores, n).indices.cpu().numpy()
        out = []
        for j in top:
            item = int(candidates[j])
            if item not in seen:
                out.append(item)
                if len(out) == k:
                    break
        return out
