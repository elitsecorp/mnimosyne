"""Deduplication of retrieved information."""

from __future__ import annotations

from mnemosyne.retrieval.ranker import ScoredItem


class Deduplicator:
    """Removes redundant information from scored results.

    Ensures each entity, relationship, and fact appears only once.
    """

    def dedup(self, items: list[ScoredItem]) -> list[ScoredItem]:
        """Deduplicate scored items, keeping highest-scored instances."""
        seen_entities: dict[str, ScoredItem] = {}
        seen_rels: dict[tuple, ScoredItem] = {}
        seen_facts: dict[tuple, ScoredItem] = {}
        seen_memories: dict[int, ScoredItem] = {}

        for item in items:
            if item.item_type == "entity":
                key = item.item.get("name", "").lower()
                existing = seen_entities.get(key)
                if not existing or item.score > existing.score:
                    seen_entities[key] = item

            elif item.item_type == "relationship":
                key = (
                    item.item.get("subject", "").lower(),
                    item.item.get("predicate", "").lower(),
                    item.item.get("object", "").lower(),
                )
                existing = seen_rels.get(key)
                if not existing or item.score > existing.score:
                    seen_rels[key] = item

            elif item.item_type == "fact":
                key = (
                    item.item.get("subject", "").lower(),
                    item.item.get("predicate", "").lower(),
                    item.item.get("object", "").lower(),
                )
                existing = seen_facts.get(key)
                if not existing or item.score > existing.score:
                    seen_facts[key] = item

            elif item.item_type == "memory":
                mid = item.item.get("id", id(item))
                existing = seen_memories.get(mid)
                if not existing or item.score > existing.score:
                    seen_memories[mid] = item

        result = []
        result.extend(seen_entities.values())
        result.extend(seen_rels.values())
        result.extend(seen_facts.values())
        result.extend(seen_memories.values())
        return sorted(result, key=lambda x: x.score, reverse=True)
