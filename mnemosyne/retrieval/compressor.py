"""Context compression to fit within token budget."""

from __future__ import annotations

from mnemosyne.retrieval.ranker import ScoredItem

CHARS_PER_TOKEN = 4


class Compressor:
    """Truncates and prioritizes scored items to fit within a token budget."""

    def __init__(self, token_budget: int = 4000) -> None:
        self._budget = token_budget

    def compress(self, items: list[ScoredItem]) -> list[ScoredItem]:
        """Keep items within the token budget, prioritizing higher-scored items."""
        budget_chars = self._budget * CHARS_PER_TOKEN
        used = 0
        result = []

        for item in items:
            text = self._item_to_text(item)
            cost = len(text) + 20
            if used + cost <= budget_chars:
                result.append(item)
                used += cost
            else:
                break

        return result

    @staticmethod
    def _item_to_text(item: ScoredItem) -> str:
        """Estimate text size of an item."""
        if item.item_type == "entity":
            return f"{item.item.get('name', '')} ({item.item.get('type', '')})"
        if item.item_type == "relationship":
            return (
                f"{item.item.get('subject', '')} "
                f"{item.item.get('predicate', '')} "
                f"{item.item.get('object', '')}"
            )
        if item.item_type == "fact":
            return (
                f"{item.item.get('subject', '')} "
                f"{item.item.get('predicate', '')} "
                f"{item.item.get('object', '')}"
            )
        if item.item_type == "memory":
            return item.item.get("text", "")
        return ""
