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
from src.models.llm_base import LLMRerankMixin
from src.models.popularity import PopularityRecommender

_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAILLMRecommender(LLMRerankMixin, Recommender):
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
        """Delegate to the shared CoT-aware, fallback-tracking mixin."""
        return self._rerank_with_llm(history, candidates)
    
    def _call_api(self, prompt: str, retries: int = 4) -> str | None:
        """Call OpenAI chat completions via SDK; return content string or None."""
        if self._client is None:
            return None

        for attempt in range(retries):
            try:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=512,
                )
                return completion.choices[0].message.content.strip()
            except Exception as exc:
                msg = str(exc)
                # Rate limit (429): back off longer before retrying.
                if "429" in msg or "rate" in msg.lower():
                    if attempt < retries - 1:
                        time.sleep(5 * (attempt + 1))  # 5s, 10s, 15s
                        continue
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"[OpenAI LLM] API error after {retries} retries: {exc}")
                    return None
        return None