"""LLM recommender variant using Google Gemini API.

Same retrieve-then-rerank strategy as llm.py (DeepSeek), but calls the
Gemini REST API instead. This lets us compare two different LLM backends
on the same task.

Usage:
    python -m experiments.run_gemini_llm --seeds 0,1,2,3,4
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd

from src.models.base import Context, Recommender
from src.models.llm_base import LLMRerankMixin
from src.models.popularity import PopularityRecommender

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiLLMRecommender(LLMRerankMixin, Recommender):
    """Retrieve-then-Rerank using Google Gemini API."""

    name = "llm_gemini"

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
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
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

    def fit(self, splits_df: pd.DataFrame) -> "GeminiLLMRecommender":
        self.retriever.fit(splits_df)

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
        """POST to Gemini generateContent; return raw text or None."""
        try:
            import urllib.request
        except ImportError:
            return None

        url = (
            f"{_GEMINI_BASE}/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )

        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 256,
            },
        }).encode()

        headers = {"Content-Type": "application/json"}

        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    url, data=payload, headers=headers, method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode())
                    return (
                        body["candidates"][0]["content"]["parts"][0]["text"]
                        .strip()
                    )
            except Exception as exc:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"[Gemini LLM] API error after {retries} retries: {exc}")
                    return None
        return None
