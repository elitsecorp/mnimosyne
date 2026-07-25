"""LLM service with provider abstraction for multiple LLM backends."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

from mnemosyne.config import settings

logger = logging.getLogger(__name__)

# Provider registry with models
PROVIDERS = {
    "gemini": {
        "name": "Gemini",
        "models": ["gemini-2.0-flash", "gemini-3.5-flash"],
        "requires_key": True,
    },
    "openai": {
        "name": "OpenAI",
        "models": ["gpt-4o", "gpt-4o-mini"],
        "requires_key": True,
    },
    "deepseek": {
        "name": "DeepSeek",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "requires_key": True,
    },
    "grok": {
        "name": "Grok",
        "models": ["grok-2", "grok-3"],
        "requires_key": True,
    },
    "kimi": {
        "name": "Kimi",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k"],
        "requires_key": True,
    },
    "ollama": {
        "name": "Ollama (Offline)",
        "models": ["llama3.2", "mistral", "phi3", "qwen2.5", "gemma2"],
        "requires_key": False,
    },
}


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], temperature: float) -> str:
        """Send a chat completion request and return the response text."""
        ...

    @abstractmethod
    def chat_json(self, messages: list[dict[str, str]], temperature: float) -> dict:
        """Send a chat completion request expecting JSON output."""
        ...

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the provider is accessible."""
        ...


class GeminiProvider(LLMProvider):
    """Google Gemini provider."""

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def chat(self, messages: list[dict[str, str]], temperature: float) -> str:
        from google.genai import types
        contents = self._convert_messages(messages)
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(temperature=temperature),
        )
        return (response.text or "").strip()

    def chat_json(self, messages: list[dict[str, str]], temperature: float) -> dict:
        from google.genai import types
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
            logger.warning("Failed to parse Gemini JSON response: %s", content[:200])
            return {}

    def test_connection(self) -> bool:
        try:
            from google.genai import types
            response = self._client.models.generate_content(
                model=self._model,
                contents=[types.Content(role="user", parts=[types.Part(text="Hi")])],
            )
            return bool(response.text)
        except Exception as e:
            logger.error("Gemini test failed: %s", e)
            return False

    @staticmethod
    def _convert_messages(messages: list[dict[str, str]]) -> list:
        from google.genai import types
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])],
            ))
        return contents


class OpenAIProvider(LLMProvider):
    """OpenAI provider."""

    def __init__(self, api_key: str, model: str) -> None:
        import openai
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def chat(self, messages: list[dict[str, str]], temperature: float) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def chat_json(self, messages: list[dict[str, str]], temperature: float) -> dict:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse OpenAI JSON response: %s", content[:200])
            return {}

    def test_connection(self) -> bool:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return bool(response.choices[0].message.content)
        except Exception as e:
            logger.error("OpenAI test failed: %s", e)
            return False


class DeepSeekProvider(LLMProvider):
    """DeepSeek provider (uses OpenAI-compatible API)."""

    def __init__(self, api_key: str, model: str) -> None:
        import openai
        self._client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self._model = model

    def chat(self, messages: list[dict[str, str]], temperature: float) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def chat_json(self, messages: list[dict[str, str]], temperature: float) -> dict:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse DeepSeek JSON response: %s", content[:200])
            return {}

    def test_connection(self) -> bool:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return bool(response.choices[0].message.content)
        except Exception as e:
            logger.error("DeepSeek test failed: %s", e)
            return False


class GrokProvider(LLMProvider):
    """xAI Grok provider (uses OpenAI-compatible API)."""

    def __init__(self, api_key: str, model: str) -> None:
        import openai
        self._client = openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        self._model = model

    def chat(self, messages: list[dict[str, str]], temperature: float) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def chat_json(self, messages: list[dict[str, str]], temperature: float) -> dict:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Grok JSON response: %s", content[:200])
            return {}

    def test_connection(self) -> bool:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return bool(response.choices[0].message.content)
        except Exception as e:
            logger.error("Grok test failed: %s", e)
            return False


class KimiProvider(LLMProvider):
    """Moonshot Kimi provider (uses OpenAI-compatible API)."""

    def __init__(self, api_key: str, model: str) -> None:
        import openai
        self._client = openai.OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")
        self._model = model

    def chat(self, messages: list[dict[str, str]], temperature: float) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()

    def chat_json(self, messages: list[dict[str, str]], temperature: float) -> dict:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Kimi JSON response: %s", content[:200])
            return {}

    def test_connection(self) -> bool:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            return bool(response.choices[0].message.content)
        except Exception as e:
            logger.error("Kimi test failed: %s", e)
            return False


class OllamaProvider(LLMProvider):
    """Ollama local provider."""

    def __init__(self, base_url: str, model: str) -> None:
        import httpx
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._http = httpx.Client(timeout=60.0)

    def chat(self, messages: list[dict[str, str]], temperature: float) -> str:
        response = self._http.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()

    def chat_json(self, messages: list[dict[str, str]], temperature: float) -> dict:
        json_msg = "Respond ONLY with valid JSON. No markdown, no explanation."
        augmented = [{"role": "system", "content": json_msg}] + messages
        response = self._http.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": augmented,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "{}")
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Ollama JSON response: %s", content[:200])
            return {}

    def test_connection(self) -> bool:
        try:
            response = self._http.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
            response.raise_for_status()
            return bool(response.json().get("message", {}).get("content"))
        except Exception as e:
            logger.error("Ollama test failed: %s", e)
            return False


_PROVIDER_CLASSES = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "deepseek": DeepSeekProvider,
    "grok": GrokProvider,
    "kimi": KimiProvider,
    "ollama": OllamaProvider,
}


class LLMService:
    """Main LLM service - delegates to selected provider."""

    def __init__(self, client=None) -> None:
        if client:
            self._provider = client
            return

        provider_name = settings.llm_provider if hasattr(settings, 'llm_provider') else "gemini"
        api_key = settings.gemini_api_key
        model = settings.gemini_llm_model
        ollama_url = "http://localhost:11434"

        provider_cls = _PROVIDER_CLASSES.get(provider_name, GeminiProvider)
        if provider_name == "ollama":
            self._provider = OllamaProvider(base_url=ollama_url, model=model)
        elif provider_name == "gemini":
            self._provider = GeminiProvider(api_key=api_key, model=model)
        else:
            self._provider = provider_cls(api_key=api_key, model=model)

        logger.info("LLM provider: %s, model: %s", provider_name, model)

    @classmethod
    def from_config(cls, provider: str, api_key: str, model: str, ollama_url: str = "http://localhost:11434") -> "LLMService":
        """Create LLMService from explicit config values."""
        provider_cls = _PROVIDER_CLASSES.get(provider, GeminiProvider)
        if provider == "ollama":
            instance = cls.__new__(cls)
            instance._provider = OllamaProvider(base_url=ollama_url, model=model)
            return instance
        elif provider == "gemini":
            instance = cls.__new__(cls)
            instance._provider = GeminiProvider(api_key=api_key, model=model)
            return instance
        else:
            instance = cls.__new__(cls)
            instance._provider = provider_cls(api_key=api_key, model=model)
            return instance

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.7) -> str:
        """Send a chat completion request and return the response text."""
        return self._provider.chat(messages, temperature)

    def chat_json(self, messages: list[dict[str, str]], temperature: float = 0.3) -> dict:
        """Send a chat completion request expecting JSON output."""
        return self._provider.chat_json(messages, temperature)

    def test_connection(self) -> bool:
        """Test if the current provider is accessible."""
        return self._provider.test_connection()
