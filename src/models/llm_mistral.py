"""LLM recommender variant using Mistral Medium 3.5 via NVIDIA NIM API.

Same retrieve-then-rerank strategy as llm.py (DeepSeek), but calls the
Mistral model hosted on NVIDIA NIM. This gives a 4th LLM backend comparison.

Usage:
    python -m experiments.run_mistral_llm --seeds 0,1,2,3,4
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd

from src.models.base import Context, Recommender
from src.models.llm_base import LLMRerankMixin
from src.models.popularity import PopularityRecommender

_NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
_DEFAULT_MODEL = "mistralai/mistral-medium-3.5-128b"


class MistralLLMRecommender(LLMRerankMixin, Recommender):
    """Retrieve-then-Rerank using Mistral Medium 3.5 via NVIDIA NIM."""

    name = "llm_mistral"

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
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
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

    def fit(self, splits_df: pd.DataFrame) -> "MistralLLMRecommender":
        self.retriever.fit(splits_df)

        from src.utils import paths
        vocab = pd.read_parquet(paths.ITEM_VOCAB_PARQUET)
        self._activity_label = dict(
            zip(vocab["item_idx"].tolist(), vocab["activity_type"].tolist())
        )

        # Initialize OpenAI-compatible client pointing at NVIDIA NIM
        from openai import OpenAI
        self._client = OpenAI(
            base_url=_NVIDIA_BASE,
            api_key=self.api_key,
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend(self, history: list[int], k: int, context: Context) -> list[int]:
        candidates = self.retriever.recommend(history, self.n_candidates, context)
        if not candidates:
            return []

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

    def _call_api(self, prompt: str, retries: int = 3) -> str | None:
        """Call Mistral via NVIDIA NIM OpenAI client; return content or None."""
        if self._client is None:
            return None

        for attempt in range(retries):
            try:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=256,
                    stream=False,
                )
                return completion.choices[0].message.content.strip()
            except Exception as exc:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"[Mistral LLM] API error after {retries} retries: {exc}")
                    return None
        return None
