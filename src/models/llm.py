"""Phase 2.4: LLM-based recommender using DeepSeek API.

Strategy: Retrieve-then-Rerank (Decision D4).
  1. A fast candidate generator (Popularity) proposes `n_candidates` items
     from the session's presentation.
  2. DeepSeek is asked to rerank those candidates given the student's recent
     click history, expressed as activity-type labels (not raw item IDs).
  3. Items the LLM returns — in its preferred order — are returned as the
     final recommendation list.

Why this design:
  - We cannot ask the LLM to generate one of ~6 000 raw item IDs reliably.
  - Providing activity-type labels (e.g. "forumng", "quiz", "resource") gives
    the model human-readable context without leaking IDs.
  - Retrieval is done by Popularity (already fitted) so this model re-uses the
    infrastructure from Phase 2.1.
  - API calls are batched per session; a simple retry loop handles transient
    rate-limit errors.

API cost control:
  - `max_history` caps how many recent items we describe to the model.
  - `n_candidates` caps the reranking list length.
  - If the LLM response cannot be parsed, we fall back to the Popularity order.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter

import pandas as pd

from src.models.base import Context, Recommender
from src.models.popularity import PopularityRecommender

_DEEPSEEK_BASE = "https://api.deepseek.com"
_DEFAULT_MODEL = "deepseek-chat"


class LLMRecommender(Recommender):
    """Retrieve-then-Rerank using DeepSeek chat completions."""

    name = "llm"

    def __init__(
        self,
        api_key: str | None = None,
        n_candidates: int = 50,
        max_history: int = 10,
        model: str = _DEFAULT_MODEL,
        seed: int = 0,
        use_rank_hint: bool = True,
        retriever: Recommender | None = None,
        retriever_type: str = "popularity",
    ) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.n_candidates = n_candidates
        self.max_history = max_history
        self.model = model
        self.seed = seed
        self.use_rank_hint = use_rank_hint
        self.retriever_type = retriever_type

        # Resolve retriever: explicit param takes priority over type string.
        if retriever is not None:
            self.retriever = retriever
        elif retriever_type == "cf":
            from src.models.cf import CFRecommender
            self.retriever = CFRecommender()
        else:
            self.retriever = PopularityRecommender()

        # item_idx -> activity_type label (e.g. "forumng")
        self._activity_label: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, splits_df: pd.DataFrame) -> "LLMRecommender":
        # Fit the underlying retriever.
        self.retriever.fit(splits_df)

        # Build item_idx -> activity_type mapping from the vocab parquet.
        from src.utils import paths
        vocab = pd.read_parquet(paths.ITEM_VOCAB_PARQUET)
        self._activity_label = dict(
            zip(vocab["item_idx"].tolist(), vocab["activity_type"].tolist())
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend(self, history: list[int], k: int, context: Context) -> list[int]:
        # Step 1: retrieve candidates.
        candidates = self.retriever.recommend(history, self.n_candidates, context)
        if not candidates:
            return []

        # Step 2: rerank with LLM.
        reranked = self._llm_rerank(history, candidates, k)
        return reranked[:k]

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _llm_rerank(
        self, history: list[int], candidates: list[int], k: int
    ) -> list[int]:
        """Ask DeepSeek to rerank `candidates` and return item indices."""
        # Represent history as last `max_history` activity-type labels.
        recent = history[-self.max_history:]
        history_labels = [
            self._activity_label.get(i, "unknown") for i in recent
        ]

        # "Last 5" recency summary for the enhanced prompt.
        last_5_labels = ", ".join(history_labels[-5:]) if history_labels else "none"

        # Map candidate item_idx to a short tag the LLM will echo back.
        cand_tags = {f"C{i:02d}": item for i, item in enumerate(candidates)}
        if self.use_rank_hint:
            cand_descriptions = [
                f"{tag}: {self._activity_label.get(item, 'unknown')} "
                f"(popularity rank {idx + 1} of {len(candidates)})"
                for idx, (tag, item) in enumerate(cand_tags.items())
            ]
        else:
            cand_descriptions = [
                f"{tag}: {self._activity_label.get(item, 'unknown')}"
                for tag, item in cand_tags.items()
            ]

        prompt = (
            "You are a learning-activity recommender for an online university course.\n\n"
            "A student's recent activity sequence (oldest → newest):\n"
            f"  {', '.join(history_labels)}\n\n"
            f"Recent pattern (last 5): {last_5_labels}\n\n"
            "Candidate next activities to rerank:\n"
            + "\n".join(f"  {d}" for d in cand_descriptions)
            + "\n\nConsider:\n"
            "1. What activity type naturally follows the student's recent pattern\n"
            "2. The popularity rank indicates how commonly students engage with each activity\n"
            "3. Activities similar to recent ones may be more relevant\n\n"
            f"Rerank all {len(candidates)} candidates from most to least relevant "
            "for this student's NEXT click.\n"
            "Reply with ONLY a JSON array of the tags in your preferred order, "
            'e.g. ["C03","C00","C11",...]. No other text.'
        )

        raw = self._call_api(prompt)
        if raw is None:
            # Fallback: return candidates in retriever order.
            return candidates

        try:
            # Strip markdown fencing if present.
            text = raw.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()

            # Extract JSON array brackets for robust parsing.
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                text = text[start:end + 1]

            tag_order: list[str] = json.loads(text)
            reranked = [cand_tags[t] for t in tag_order if t in cand_tags]
            # Append any candidates the LLM missed (preserve coverage).
            seen = set(reranked)
            for item in candidates:
                if item not in seen:
                    reranked.append(item)
            return reranked
        except Exception:
            return candidates

    def _call_api(self, prompt: str, retries: int = 3) -> str | None:
        """POST to DeepSeek chat completions; return raw content string or None."""
        try:
            import urllib.request
        except ImportError:
            return None

        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 256,
        }).encode()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    f"{_DEEPSEEK_BASE}/chat/completions",
                    data=payload,
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode())
                    return body["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"[LLM] API error after {retries} retries: {exc}")
                    return None
        return None
