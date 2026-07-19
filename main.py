"""Mnemosyne entry point. Run with: uv run main.py"""

from __future__ import annotations

import logging
import sys

import uvicorn


def main() -> None:
    """Start the Mnemosyne FastAPI server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    uvicorn.run(
        "mnemosyne.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
