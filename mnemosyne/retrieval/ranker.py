"""Multi-signal ranking for retrieved information."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC


@dataclass
class ScoredItem:
    """A scored piece of retrieved information."""

    item: dict
    item_type: str  # "entity", "relationship", "fact", "memory"
    score: float
    signals: dict[str, float] = field(default_factory=dict)


def _compute_recency(item: dict, now: datetime) -> float:
    """Compute recency score from item timestamp. Returns 0.0-1.0."""
    ts_str = item.get("timestamp", "")
    if not ts_str:
        return 0.5
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        age_hours = max((now - ts).total_seconds() / 3600, 0)
        return max(0.0, 1.0 - min(age_hours / 720, 1.0))
    except (ValueError, TypeError):
        return 0.5


def _compute_evidence_count(item: dict, all_facts: list[dict], all_relationships: list[dict]) -> float:
    """Count supporting evidence for an item. Returns normalized score 0.0-1.0."""
    name = item.get("name", item.get("subject", ""))
    if not name:
        return 0.0

    count = 0
    for f in all_facts:
        if name in (f.get("subject", ""), f.get("object", "")):
            count += 1
    for r in all_relationships:
        if name in (r.get("subject", ""), r.get("object", "")):
            count += 1

    return min(1.0, count / 10.0)


class Ranker:
    """Scores retrieved items using multiple configurable signals.

    Signals: entity_overlap, graph_proximity, semantic_similarity,
    confidence, recency, evidence_count.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or {
            "entity_overlap": 0.3,
            "graph_proximity": 0.2,
            "semantic_similarity": 0.2,
            "confidence": 0.15,
            "recency": 0.1,
            "evidence_count": 0.05,
        }

    def rank_entities(
        self,
        entities: list[dict],
        query_entities: list[str],
        entity_scores: dict[str, float] | None = None,
        all_facts: list[dict] | None = None,
        all_relationships: list[dict] | None = None,
    ) -> list[ScoredItem]:
        """Rank entities by relevance to query."""
        now = datetime.now(UTC)
        facts = all_facts or []
        rels = all_relationships or []
        items = []
        for ent in entities:
            signals = {}
            name = ent.get("name", "")

            signals["entity_overlap"] = 1.0 if name.lower() in [e.lower() for e in query_entities] else 0.0
            signals["graph_proximity"] = entity_scores.get(name, 0) if entity_scores else 0
            signals["confidence"] = ent.get("confidence", 0)
            signals["recency"] = _compute_recency(ent, now)
            signals["semantic_similarity"] = 0.0
            signals["evidence_count"] = _compute_evidence_count(ent, facts, rels)

            score = sum(signals.get(k, 0) * v for k, v in self._weights.items())
            items.append(ScoredItem(item=ent, item_type="entity", score=score, signals=signals))

        return sorted(items, key=lambda x: x.score, reverse=True)

    def rank_relationships(
        self,
        relationships: list[dict],
        query_entities: list[str],
    ) -> list[ScoredItem]:
        """Rank relationships by relevance."""
        now = datetime.now(UTC)
        query_lower = {e.lower() for e in query_entities}
        items = []
        for rel in relationships:
            signals = {}
            subject = rel.get("subject", "").lower()
            obj = rel.get("object", "").lower()

            overlap_count = sum(1 for e in query_lower if e in subject or e in obj)
            signals["entity_overlap"] = min(1.0, overlap_count / max(len(query_lower), 1))
            signals["graph_proximity"] = 1.0 / max(rel.get("distance", 1), 1)
            signals["confidence"] = rel.get("confidence", 0)
            signals["recency"] = _compute_recency(rel, now)
            signals["semantic_similarity"] = rel.get("score", 0)
            signals["evidence_count"] = 0.0

            score = sum(signals.get(k, 0) * v for k, v in self._weights.items())
            items.append(ScoredItem(item=rel, item_type="relationship", score=score, signals=signals))

        return sorted(items, key=lambda x: x.score, reverse=True)

    def rank_facts(
        self,
        facts: list[dict],
        query_entities: list[str],
    ) -> list[ScoredItem]:
        """Rank facts by relevance."""
        now = datetime.now(UTC)
        query_lower = {e.lower() for e in query_entities}
        items = []
        for fact in facts:
            signals = {}
            subject = fact.get("subject", "").lower()
            obj = fact.get("object", "").lower()

            overlap_count = sum(1 for e in query_lower if e in subject or e in obj)
            signals["entity_overlap"] = min(1.0, overlap_count / max(len(query_lower), 1))
            signals["graph_proximity"] = 0.5
            signals["confidence"] = fact.get("confidence", 0.5)
            signals["recency"] = _compute_recency(fact, now)
            signals["semantic_similarity"] = 0.0
            signals["evidence_count"] = 0.0

            score = sum(signals.get(k, 0) * v for k, v in self._weights.items())
            items.append(ScoredItem(item=fact, item_type="fact", score=score, signals=signals))

        return sorted(items, key=lambda x: x.score, reverse=True)

    def rank_memories(
        self,
        memories: list[dict],
        query_entities: list[str],
    ) -> list[ScoredItem]:
        """Rank vector memories by relevance."""
        now = datetime.now(UTC)
        items = []
        for mem in memories:
            signals = {}
            text_lower = mem.get("text", "").lower()

            signals["entity_overlap"] = sum(
                1 for e in query_entities if e.lower() in text_lower
            ) / max(len(query_entities), 1)
            signals["graph_proximity"] = 0.0
            signals["semantic_similarity"] = mem.get("score", 0)
            signals["confidence"] = 0.5
            signals["recency"] = _compute_recency(mem, now)
            signals["evidence_count"] = 0.0

            score = sum(signals.get(k, 0) * v for k, v in self._weights.items())
            items.append(ScoredItem(item=mem, item_type="memory", score=score, signals=signals))

        return sorted(items, key=lambda x: x.score, reverse=True)
