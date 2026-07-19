"""Gemini LLM client for chat completions."""

from __future__ import annotations

import json
import logging
from typing import Optional

from google import genai
from google.genai import types

from mnemosyne.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Thin wrapper around Gemini for chat completions."""

    def __init__(self, client: Optional[genai.Client] = None) -> None:
        self._client = client or genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_llm_model

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.7) -> str:
        """Send a chat completion request and return the response text."""
        contents = self._convert_messages(messages)
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
            ),
        )
        return (response.text or "").strip()

    def chat_json(self, messages: list[dict[str, str]], temperature: float = 0.3) -> dict:
        """Send a chat completion request expecting JSON output.

        Parses the response as JSON. Returns empty dict on parse failure.
        """
        contents = self._convert_messages(messages)
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        content = response.text or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON response: %s", content[:200])
            return {}

    @staticmethod
    def _convert_messages(messages: list[dict[str, str]]) -> list[types.Content]:
        """Convert OpenAI-style message dicts to Gemini Content objects."""
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])],
            ))
        return contents
