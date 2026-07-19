"""Deterministic entity resolution from user queries."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from mnemosyne.models import Entity


def _tokenize(text: str) -> set[str]:
    """Lowercase, split on non-alphanumeric, return token set."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


@dataclass
class ResolvedEntity:
    """An entity resolved from the user query."""

    id: int
    name: str
    type: str
    confidence: float
    match_method: str
    match_score: float


class EntityResolver:
    """Resolves entity mentions in queries to actual graph entities.

    Uses exact match, substring match, token overlap, and fuzzy matching.
    All deterministic — no LLM calls.
    """

    def __init__(self, db: Session) -> None:
        self._db = db
        self._entities = db.query(Entity).all()
        self._name_map = {e.name.lower(): e for e in self._entities}

    def resolve(self, query: str, limit: int = 10) -> list[ResolvedEntity]:
        """Resolve entity mentions in the query to ranked candidates."""
        candidates: dict[int, ResolvedEntity] = {}
        query_lower = query.lower()
        query_tokens = _tokenize(query)

        for entity in self._entities:
            score, method = self._score_entity(entity, query_lower, query_tokens)
            if score > 0:
                existing = candidates.get(entity.id)
                if not existing or score > existing.match_score:
                    candidates[entity.id] = ResolvedEntity(
                        id=entity.id,
                        name=entity.name,
                        type=entity.type,
                        confidence=entity.confidence,
                        match_method=method,
                        match_score=score,
                    )

        results = sorted(candidates.values(), key=lambda r: r.match_score, reverse=True)
        return results[:limit]

    def _score_entity(self, entity: Entity, query_lower: str, query_tokens: set[str]) -> tuple[float, str]:
        """Score how well an entity matches the query. Returns (score, method)."""
        name_lower = entity.name.lower()

        if name_lower == query_lower:
            return 1.0, "exact"

        if name_lower in query_lower:
            return 0.9, "substring_contained"

        if query_lower in name_lower:
            return 0.85, "substring_contains"

        name_tokens = _tokenize(entity.name)
        relevant_query_tokens = query_tokens - _tokenize("the a an is are was were do does did what who where when why how tell about")
        if relevant_query_tokens:
            overlap = _jaccard(relevant_query_tokens, name_tokens)
            if overlap > 0.5:
                return overlap * 0.8, "token_overlap"

        for qt in query_tokens:
            if len(qt) >= 3:
                dist = _levenshtein(qt, name_lower)
                max_len = max(len(qt), len(name_lower))
                sim = 1 - (dist / max_len) if max_len > 0 else 0
                if sim > 0.7:
                    return sim * 0.6, "fuzzy"

        return 0.0, "none"
