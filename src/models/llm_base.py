"""Shared retrieve-then-rerank logic for all LLM recommender backends.

Every LLM backend (DeepSeek, Gemini, OpenAI, NVIDIA, Mistral) uses an identical
reranking contract: a fast retriever proposes candidates, the LLM reranks them.
Historically that logic (prompt construction + response parsing + fallback) was
copy-pasted into five files, which meant a change had to be made five times and
they had already drifted. This mixin makes it a single source of truth.

Two things this module fixes relative to the earlier per-file implementations:

1. Chain-of-Thought (CoT) prompting. The model is now asked to *reason step by
   step* before producing its ranking, and to return a JSON object
   ``{"reasoning": "...", "ranking": ["C03", ...]}``. The course brief (Task 2)
   requires a CoT guidance component; the previous prompt actively suppressed it
   ("Reply with ONLY a JSON array ... No other text"). The reasoning strings are
   captured so a few can be shown in the paper.

2. Fallback accounting. When the API call fails or the response cannot be parsed,
   we fall back to the retriever's candidate order (unchanged behaviour) BUT we
   now COUNT it. Two "different" LLMs producing byte-identical metrics is the
   signature of both silently falling back to the same retriever. Exposing
   ``fallback_rate`` lets us detect and honestly report that instead of
   publishing fallback artifacts as if they were LLM outputs.

A backend class becomes CoT-and-fallback-aware simply by inheriting this mixin
and implementing ``_call_api(self, prompt) -> str | None``.
"""
from __future__ import annotations

import json
from typing import Sequence


class LLMRerankMixin:
    """Prompt construction, CoT-aware parsing, and fallback tracking.

    Expects the host class to provide these attributes (all five backends do):
        self.max_history : int
        self.use_rank_hint : bool
        self._activity_label : dict[int, str]
    and the method:
        self._call_api(prompt: str) -> str | None
    """

    # -- fallback / reasoning accounting -------------------------------------
    # These are initialised lazily so existing __init__ methods need no change.
    _llm_total_calls: int = 0
    _llm_fallback_calls: int = 0
    _reasoning_log: list[str] | None = None

    # ------------------------------------------------------------------------
    # Public diagnostics
    # ------------------------------------------------------------------------
    @property
    def fallback_rate(self) -> float:
        """Fraction of rerank calls that fell back to retriever order."""
        if not self._llm_total_calls:
            return 0.0
        return self._llm_fallback_calls / self._llm_total_calls

    @property
    def n_calls(self) -> int:
        return self._llm_total_calls

    @property
    def n_fallbacks(self) -> int:
        return self._llm_fallback_calls

    def reset_diagnostics(self) -> None:
        self._llm_total_calls = 0
        self._llm_fallback_calls = 0
        self._reasoning_log = []

    def sample_reasoning(self, n: int = 3) -> list[str]:
        """Return up to n captured reasoning chains (for the paper appendix)."""
        if not self._reasoning_log:
            return []
        return self._reasoning_log[:n]

    # ------------------------------------------------------------------------
    # Prompt construction (CoT)
    # ------------------------------------------------------------------------
    def _build_prompt(self, history: list[int], candidates: Sequence[int]) -> tuple[str, dict[str, int]]:
        recent = history[-self.max_history:]
        history_labels = [self._activity_label.get(i, "unknown") for i in recent]
        last_5_labels = ", ".join(history_labels[-5:]) if history_labels else "none"

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
            "A student's recent activity sequence (oldest \u2192 newest):\n"
            f"  {', '.join(history_labels)}\n\n"
            f"Recent pattern (last 5): {last_5_labels}\n\n"
            "Candidate next activities to rerank:\n"
            + "\n".join(f"  {d}" for d in cand_descriptions)
            + "\n\nThink step by step about which activity the student is most "
            "likely to engage with next. Consider:\n"
            "1. What activity type naturally follows the student's recent pattern.\n"
            "2. The popularity rank (how commonly students engage with each activity).\n"
            "3. Activities similar to the student's recent ones may be more relevant.\n\n"
            f"Then rerank all {len(candidates)} candidates from most to least "
            "relevant for this student's NEXT click.\n\n"
            "Respond with a single JSON object and nothing else, in exactly this form:\n"
            '{"reasoning": "<your concise step-by-step reasoning>", '
            '"ranking": ["C03", "C00", "C11", ...]}\n'
            "The \"ranking\" array must contain every candidate tag exactly once."
        )
        return prompt, cand_tags

    # ------------------------------------------------------------------------
    # Response parsing (CoT-aware, with fallback accounting)
    # ------------------------------------------------------------------------
    def _parse_response(
        self, raw: str | None, candidates: Sequence[int], cand_tags: dict[str, int]
    ) -> tuple[list[int], bool]:
        """Return (reranked_items, fell_back).

        Accepts either the new CoT object form {"reasoning":..,"ranking":[..]}
        or a bare JSON array (backwards compatible). On any failure returns the
        retriever's candidate order and fell_back=True.
        """
        if raw is None:
            return list(candidates), True

        text = raw.strip()
        # Strip markdown fences if present.
        if text.startswith("```"):
            lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        reasoning = ""
        tag_order: list[str] | None = None

        # Preferred path: a JSON object with reasoning + ranking.
        obj_start, obj_end = text.find("{"), text.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            try:
                obj = json.loads(text[obj_start:obj_end + 1])
                if isinstance(obj, dict) and "ranking" in obj:
                    tag_order = list(obj["ranking"])
                    reasoning = str(obj.get("reasoning", ""))
            except Exception:
                tag_order = None

        # Fallback path: a bare JSON array of tags.
        if tag_order is None:
            arr_start, arr_end = text.find("["), text.rfind("]")
            if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
                try:
                    tag_order = list(json.loads(text[arr_start:arr_end + 1]))
                except Exception:
                    tag_order = None

        if not tag_order:
            return list(candidates), True

        reranked = [cand_tags[t] for t in tag_order if t in cand_tags]
        if not reranked:
            return list(candidates), True

        # Append any candidates the LLM dropped, preserving full coverage.
        seen = set(reranked)
        for item in candidates:
            if item not in seen:
                reranked.append(item)

        if reasoning:
            if self._reasoning_log is None:
                self._reasoning_log = []
            if len(self._reasoning_log) < 50:  # cap memory
                self._reasoning_log.append(reasoning)

        return reranked, False

    # ------------------------------------------------------------------------
    # Orchestration used by every backend's _llm_rerank()
    # ------------------------------------------------------------------------
    def _rerank_with_llm(self, history: list[int], candidates: Sequence[int]) -> list[int]:
        """Full rerank: build prompt, call API, parse, track fallback."""
        candidates = list(candidates)
        if not candidates:
            return candidates

        # lazily init counters (works regardless of subclass __init__)
        if self._reasoning_log is None:
            self._reasoning_log = []

        prompt, cand_tags = self._build_prompt(history, candidates)
        raw = self._call_api(prompt)
        reranked, fell_back = self._parse_response(raw, candidates, cand_tags)

        self._llm_total_calls += 1
        if fell_back:
            self._llm_fallback_calls += 1
        return reranked