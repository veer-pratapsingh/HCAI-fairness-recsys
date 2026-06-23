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
from src.models.popularity import PopularityRecommender

_NVIDIA_BASE = "https://integrate.api.nvidia.com/v1"
_DEFAULT_MODEL = "mistralai/mistral-medium-3.5-128b"


class MistralLLMRecommender(Recommender):
    """Retrieve-then-Rerank using Mistral Medium 3.5 via NVIDIA NIM."""

    name = "llm_mistral"

    def __init__(
        self,
        api_key: str | None = None,
        n_candidates: int = 20,
        max_history: int = 10,
        model: str = _DEFAULT_MODEL,
        seed: int = 0,
    ) -> None:
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        self.n_candidates = n_candidates
        self.max_history = max_history
        self.model = model
        self.seed = seed

        self._pop: PopularityRecommender = PopularityRecommender()
        self._activity_label: dict[int, str] = {}
        self._client = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, splits_df: pd.DataFrame) -> "MistralLLMRecommender":
        self._pop.fit(splits_df)

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
        candidates = self._pop.recommend(history, self.n_candidates, context)
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
        """Ask Mistral to rerank candidates and return item indices."""
        recent = history[-self.max_history:]
        history_labels = [
            self._activity_label.get(i, "unknown") for i in recent
        ]

        cand_tags = {f"C{i:02d}": item for i, item in enumerate(candidates)}
        cand_descriptions = [
            f"{tag}: {self._activity_label.get(item, 'unknown')}"
            for tag, item in cand_tags.items()
        ]

        prompt = (
            "You are a learning-activity recommender for an online university course.\n"
            "A student's recent activity sequence (oldest → newest):\n"
            f"  {', '.join(history_labels)}\n\n"
            "Candidate next activities to rerank (tag: type):\n"
            + "\n".join(f"  {d}" for d in cand_descriptions)
            + f"\n\nRerank all {len(candidates)} candidates from most to least relevant "
            "for this student's NEXT click.\n"
            "Reply with ONLY a JSON array of the tags in your preferred order, "
            'e.g. ["C03","C00","C11",...]. No other text.'
        )

        raw = self._call_api(prompt)
        if raw is None:
            return candidates

        try:
            text = raw.strip()
            # Remove markdown fencing if present
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()

            # Find the JSON array in the response
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                text = text[start:end + 1]

            tag_order: list[str] = json.loads(text)
            reranked = [cand_tags[t] for t in tag_order if t in cand_tags]
            seen = set(reranked)
            for item in candidates:
                if item not in seen:
                    reranked.append(item)
            return reranked
        except Exception:
            return candidates

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
