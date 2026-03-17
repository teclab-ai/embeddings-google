"""
LLM abstraction supporting Gemini (via google-genai), OpenAI, and Anthropic Claude.

Usage
-----
    llm = build_llm()           # reads LLM_PROVIDER from config
    answer = llm.answer(query, context)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


# ── Base class ────────────────────────────────────────────────────────────────

class BaseLLM(ABC):
    @abstractmethod
    def answer(self, query: str, context: str) -> str:
        """Generate an answer for `query` given the retrieved `context`."""

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            "You are a helpful assistant with access to retrieved multimodal context.\n\n"
            "RETRIEVED CONTEXT\n"
            "─────────────────\n"
            f"{context}\n\n"
            "─────────────────\n"
            "Using the context above, answer the following question.\n"
            "If the context does not contain enough information, say so clearly.\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )


# ── Gemini ────────────────────────────────────────────────────────────────────

class GeminiLLM(BaseLLM):
    def __init__(self, model: Optional[str] = None) -> None:
        from google import genai

        self._model = model or settings.GEMINI_LLM_MODEL
        self._client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        logger.info("GeminiLLM initialised with model '%s'.", self._model)

    def answer(self, query: str, context: str) -> str:
        prompt = self._build_prompt(query, context)
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
        )
        return response.text.strip()


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAILLM(BaseLLM):
    def __init__(self, model: Optional[str] = None) -> None:
        from openai import OpenAI

        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Set it in .env or switch LLM_PROVIDER=gemini."
            )
        self._model = model or settings.OPENAI_MODEL
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        logger.info("OpenAILLM initialised with model '%s'.", self._model)

    def answer(self, query: str, context: str) -> str:
        prompt = self._build_prompt(query, context)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()


# ── Anthropic / Claude ────────────────────────────────────────────────────────

class ClaudeLLM(BaseLLM):
    def __init__(self, model: Optional[str] = None) -> None:
        from anthropic import Anthropic

        if not settings.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Set it in .env or switch LLM_PROVIDER to 'gemini' or 'openai'."
            )
        self._model = model or settings.CLAUDE_MODEL
        self._client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        logger.info("ClaudeLLM initialised with model '%s'.", self._model)

    def answer(self, query: str, context: str) -> str:
        prompt = self._build_prompt(query, context)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


# ── Factory ───────────────────────────────────────────────────────────────────

def build_llm(provider: Optional[str] = None) -> BaseLLM:
    """
    Return an LLM instance for the given provider.

    Parameters
    ----------
    provider : "gemini" | "openai" | "claude" | None
        If None, reads from settings.LLM_PROVIDER.
    """
    p = (provider or settings.LLM_PROVIDER).lower()
    if p == "gemini":
        return GeminiLLM()
    elif p == "openai":
        return OpenAILLM()
    elif p == "claude":
        return ClaudeLLM()
    else:
        raise ValueError(
            f"Unknown LLM provider '{p}'. Choose 'gemini', 'openai', or 'claude'."
        )
