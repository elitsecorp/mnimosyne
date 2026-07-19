"""Memory extraction: parse LLM output into structured entities, relationships, and facts."""

from __future__ import annotations

import logging
from typing import Optional

from mnemosyne.llm import LLMService
from mnemosyne.prompts import build_extraction_messages
from mnemosyne.schemas import ExtractionResult, EntitySchema, RelationshipSchema, FactSchema

logger = logging.getLogger(__name__)


def extract_memory(
    llm: LLMService,
    user_message: str,
    assistant_response: str,
) -> ExtractionResult:
    """Extract entities, relationships, and facts from a conversation turn.

    Args:
        llm: LLM service instance.
        user_message: The user's message.
        assistant_response: The assistant's response.

    Returns:
        ExtractionResult with extracted knowledge. Empty on failure.
    """
    messages = build_extraction_messages(user_message, assistant_response)
    raw = llm.chat_json(messages)

    if not raw:
        logger.warning("Empty extraction response from LLM")
        return ExtractionResult()

    return _parse_extraction(raw)


def _parse_extraction(data: dict) -> ExtractionResult:
    """Parse raw JSON dict into ExtractionResult.

    Args:
        data: Parsed JSON from the LLM.

    Returns:
        ExtractionResult with validated data.
    """
    entities = []
    for item in data.get("entities", []):
        try:
            entities.append(EntitySchema(
                name=str(item["name"]).strip(),
                type=str(item["type"]).strip().lower(),
                confidence=float(item.get("confidence", 0.9)),
            ))
        except (KeyError, ValueError, TypeError):
            logger.debug("Skipping invalid entity: %s", item)

    relationships = []
    for item in data.get("relationships", []):
        try:
            relationships.append(RelationshipSchema(
                subject=str(item["subject"]).strip(),
                predicate=_normalize_predicate(item["predicate"]),
                object=str(item["object"]).strip(),
                confidence=float(item.get("confidence", 0.9)),
            ))
        except (KeyError, ValueError, TypeError):
            logger.debug("Skipping invalid relationship: %s", item)

    facts = []
    for item in data.get("facts", []):
        try:
            facts.append(FactSchema(
                subject=str(item["subject"]).strip(),
                predicate=_normalize_predicate(item["predicate"]),
                object=str(item["object"]).strip(),
            ))
        except (KeyError, ValueError, TypeError):
            logger.debug("Skipping invalid fact: %s", item)

    return ExtractionResult(entities=entities, relationships=relationships, facts=facts)


def _normalize_predicate(predicate: str) -> str:
    """Normalize a predicate to snake_case."""
    return (
        predicate.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )
