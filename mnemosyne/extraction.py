"""Memory extraction: parse LLM output into structured entities, relationships, and facts."""

from __future__ import annotations

import logging
import re

from mnemosyne.llm import LLMService
from mnemosyne.prompts import build_extraction_messages
from mnemosyne.schemas import ExtractionResult, EntitySchema, RelationshipSchema, FactSchema

logger = logging.getLogger(__name__)

_USER_ALIASES = {"me", "user", "owner", "i"}

_VALID_PREDICATES = {
    "owns", "has", "likes", "dislikes", "works_for", "works_at",
    "lives_in", "located_in", "knows", "uses", "created", "built",
    "visited", "belongs_to", "has_goal", "has_project", "has_skill",
    "has_habit", "has_resource", "speaks", "learning", "interested_in",
    "believes", "has_pet", "has_friend", "attended", "is_a",
    "has_name", "has_role", "discussed", "is_interested_in",
    "sent", "shared", "asked_about", "replied_to",
}

_PREDICATE_SYNONYMS = {
    "works_at": "works_for",
    "employed_by": "works_for",
    "lives_at": "lives_in",
    "is_located_in": "located_in",
    "has_pet": "owns",
    "has_friend": "knows",
    "is_interested_in": "interested_in",
}


def extract_memory(
    llm: LLMService,
    user_message: str,
    assistant_response: str,
) -> ExtractionResult:
    """Extract entities, relationships, and facts from a conversation turn."""
    messages = build_extraction_messages(user_message, assistant_response)
    raw = llm.chat_json(messages)

    if not raw:
        logger.warning("Empty extraction response from LLM")
        return ExtractionResult()

    return _enforce_fact_coverage(_parse_extraction(raw))


def _normalize_entity_name(name: str) -> str:
    """Normalize entity name: strip whitespace, unify user references."""
    name = name.strip()
    if name.lower() in _USER_ALIASES:
        return "Me"
    return name


def _normalize_predicate(predicate: str) -> str:
    """Normalize a predicate to snake_case and map synonyms."""
    result = predicate.strip().lower()
    result = re.sub(r"[^a-z0-9]+", "_", result)
    result = re.sub(r"_+", "_", result).strip("_")
    return _PREDICATE_SYNONYMS.get(result, result)


def _enforce_fact_coverage(result: ExtractionResult) -> ExtractionResult:
    """Ensure every relationship has a corresponding fact."""
    existing_facts = {(f.subject.lower(), f.predicate.lower(), f.object.lower()) for f in result.facts}

    for rel in result.relationships:
        key = (rel.subject.lower(), rel.predicate.lower(), rel.object.lower())
        if key not in existing_facts:
            result.facts.append(FactSchema(
                subject=rel.subject,
                predicate=rel.predicate,
                object=rel.object,
            ))
            existing_facts.add(key)

    return result


def _parse_extraction(data: dict) -> ExtractionResult:
    """Parse raw JSON dict into ExtractionResult."""
    entities = []
    seen_entity_names = set()
    for item in data.get("entities", []):
        try:
            name = _normalize_entity_name(str(item["name"]))
            if not name or name.lower() in seen_entity_names:
                continue
            seen_entity_names.add(name.lower())
            entities.append(EntitySchema(
                name=name,
                type=str(item["type"]).strip().lower(),
                confidence=float(item.get("confidence", 0.9)),
            ))
        except (KeyError, ValueError, TypeError):
            logger.debug("Skipping invalid entity: %s", item)

    relationships = []
    for item in data.get("relationships", []):
        try:
            subject = _normalize_entity_name(str(item["subject"]))
            obj = _normalize_entity_name(str(item["object"]))
            relationships.append(RelationshipSchema(
                subject=subject,
                predicate=_normalize_predicate(item["predicate"]),
                object=obj,
                confidence=float(item.get("confidence", 0.9)),
            ))
        except (KeyError, ValueError, TypeError):
            logger.debug("Skipping invalid relationship: %s", item)

    facts = []
    for item in data.get("facts", []):
        try:
            subject = _normalize_entity_name(str(item["subject"]))
            obj = _normalize_entity_name(str(item["object"]))
            facts.append(FactSchema(
                subject=subject,
                predicate=_normalize_predicate(item["predicate"]),
                object=obj,
            ))
        except (KeyError, ValueError, TypeError):
            logger.debug("Skipping invalid fact: %s", item)

    rel_set = {(r.subject.lower(), r.predicate.lower(), r.object.lower()) for r in relationships}
    matching_facts = [f for f in facts if (f.subject.lower(), f.predicate.lower(), f.object.lower()) in rel_set]
    extra_facts = [f for f in facts if (f.subject.lower(), f.predicate.lower(), f.object.lower()) not in rel_set]
    all_facts = matching_facts + extra_facts if extra_facts else facts

    return ExtractionResult(entities=entities, relationships=relationships, facts=all_facts)