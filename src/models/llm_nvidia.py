"""LLM recommender variant using NVIDIA Nemotron API (OpenAI-compatible).

Same retrieve-then-rerank strategy as llm.py (DeepSeek), but calls the
NVIDIA NIM API with the Nemotron reasoning model. This lets us compare
three different LLM backends on the same task.

Usage:
    python -m experiments.run_nvidia_llm --seeds 0,1,2,3,4
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
_DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"


class NvidiaLLMRecommender(LLMRerankMixin, Recommender):
    """Retrieve-then-Rerank using NVIDIA Nemotron via OpenAI-compatible API."""

    name = "llm_nvidia"

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
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
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

    def fit(self, splits_df: pd.DataFrame) -> "NvidiaLLMRecommender":
        self.retriever.fit(splits_df)

        from src.utils import paths
        vocab = pd.read_parquet(paths.ITEM_VOCAB_PARQUET)
        self._activity_label = dict(
            zip(vocab["item_idx"].tolist(), vocab["activity_type"].tolist())
        )

        # Initialize OpenAI client pointing at NVIDIA NIM
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
        """Call NVIDIA NIM via OpenAI client; return content string or None."""
        if self._client is None:
            return None

        for attempt in range(retries):
            try:
                # Use non-streaming for simpler parsing
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
                    print(f"[NVIDIA LLM] API error after {retries} retries: {exc}")
                    return None
        return None
