"""Configuration management. Reads settings from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Application configuration loaded from environment variables."""

    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_llm_model: str = field(default_factory=lambda: os.getenv("GEMINI_LLM_MODEL", "gemini-2.0-flash"))
    gemini_embedding_model: str = field(default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"))
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///mnemosyne.db"))
    vec_top_k: int = field(default_factory=lambda: int(os.getenv("VEC_TOP_K", "5")))


def load_settings() -> Settings:
    """Load environment variables from .env and return Settings."""
    load_dotenv()
    return Settings()


settings = load_settings()
