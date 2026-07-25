"""Configuration service: reads/writes settings from DB with Fernet encryption."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import text

from mnemosyne.models import Setting

logger = logging.getLogger(__name__)

_KEY_FILE = Path(".fernet_key")


class ConfigService:
    """Read/write settings from DB with Fernet encryption for API keys."""

    def __init__(self, db: Session) -> None:
        self._db = db
        self._fernet = self._init_fernet()

    def _init_fernet(self):
        """Initialize Fernet encryption."""
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            logger.warning("cryptography not installed, encryption disabled")
            return None

        if _KEY_FILE.exists():
            key = _KEY_FILE.read_bytes()
        else:
            key = Fernet.generate_key()
            _KEY_FILE.write_bytes(key)
            logger.info("Generated new Fernet encryption key")

        return Fernet(key)

    def get(self, key: str, default: str = "") -> str:
        """Get a setting value (decrypts if needed)."""
        row = self._db.execute(
            text("SELECT value, encrypted FROM settings WHERE key = :key"),
            {"key": key},
        ).fetchone()
        if not row:
            return default
        value, encrypted = row
        if encrypted and self._fernet and value:
            try:
                return self._fernet.decrypt(value.encode()).decode()
            except Exception:
                return value
        return value

    def set(self, key: str, value: str, encrypted: bool = False) -> None:
        """Set a setting value (encrypts if needed)."""
        store_value = value
        if encrypted and self._fernet and value:
            store_value = self._fernet.encrypt(value.encode()).decode()

        existing = self._db.execute(
            text("SELECT id FROM settings WHERE key = :key"),
            {"key": key},
        ).fetchone()

        if existing:
            self._db.execute(
                text("UPDATE settings SET value = :value, encrypted = :encrypted WHERE key = :key"),
                {"value": store_value, "encrypted": encrypted, "key": key},
            )
        else:
            self._db.execute(
                text("INSERT INTO settings (key, value, encrypted) VALUES (:key, :value, :encrypted)"),
                {"key": key, "value": store_value, "encrypted": encrypted},
            )
        self._db.commit()

    def get_all(self) -> dict:
        """Get all settings (decrypts API keys for use, masks for display)."""
        rows = self._db.execute(text("SELECT key, value, encrypted FROM settings")).fetchall()
        result = {}
        for key, value, encrypted in rows:
            if encrypted and self._fernet and value:
                try:
                    result[key] = self._fernet.decrypt(value.encode()).decode()
                except Exception:
                    result[key] = value
            else:
                result[key] = value
        return result

    def get_masked(self) -> dict:
        """Get all settings with API keys masked for display."""
        rows = self._db.execute(text("SELECT key, value, encrypted FROM settings")).fetchall()
        result = {}
        for key, value, encrypted in rows:
            if encrypted and value:
                result[key] = "••••••••" if len(value) > 4 else "****"
            else:
                result[key] = value
        return result

    def update_many(self, settings: dict) -> None:
        """Update multiple settings at once."""
        for key, value in settings.items():
            if value == "••••••••" or value == "****":
                continue
            existing = self._db.execute(
                text("SELECT encrypted FROM settings WHERE key = :key"),
                {"key": key},
            ).fetchone()
            encrypted = existing[0] if existing else False
            self.set(key, value, encrypted=encrypted)
