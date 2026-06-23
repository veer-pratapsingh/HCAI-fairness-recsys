"""LLM recommender variant using OpenAI (ChatGPT) API.

Same retrieve-then-rerank strategy as llm.py (DeepSeek), but calls the
OpenAI API via the official ``openai`` Python SDK.  This lets us compare
multiple LLM backends on the same task.

Features over the base LLM variant:
  - Supports both CF and Popularity retrievers via ``retriever_type``.
  - Enhanced prompt with "last 5" recency pattern and rank hints.
  - Robust JSON parsing that strips markdown fencing and extracts brackets.

Usage:
    python -m experiments.run_openai_llm --seeds 0,1,2,3,4
"""
from __future__ import annotations

import json
import os
import re
import time

import pandas as pd

from src.models.base import Context, Recommender
from src.models.popularity import PopularityRecommender

_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAILLMRecommender(Recommender):
    """Retrieve-then-Rerank using OpenAI chat completions."""

    name = "llm_openai"

    def __init__(
        self,
        api_key: str | None = None,
        n_candidates: int = 50,
        max_history: int = 10,
        model: str = _DEFAULT_MODEL,
        seed: int = 0,
        use_rank_hint: bool = True,
        retriever: Recommender | None = None,
        retriever_type: str = "cf",
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
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
        self._client = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, splits_df: pd.DataFrame) -> "OpenAILLMRecommender":
        """Fit the underlying retriever and initialise the OpenAI client."""
        self.retriever.fit(splits_df)

        # Build item_idx -> activity_type mapping from the vocab parquet.
        from src.utils import paths
        vocab = pd.read_parquet(paths.ITEM_VOCAB_PARQUET)
        self._activity_label = dict(
            zip(vocab["item_idx"].tolist(), vocab["activity_type"].tolist())
        )

        # Initialise the OpenAI client (standard endpoint).
        from openai import OpenAI
        self._client = OpenAI(api_key=self.api_key)
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
        """Ask OpenAI to rerank `candidates` and return item indices."""
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
        """Call OpenAI chat completions via SDK; return content string or None."""
        if self._client is None:
            return None

        for attempt in range(retries):
            try:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=256,
                )
                return completion.choices[0].message.content.strip()
            except Exception as exc:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"[OpenAI LLM] API error after {retries} retries: {exc}")
                    return None
        return None
