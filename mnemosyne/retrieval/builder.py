"""Structured context construction for LLM consumption."""

from __future__ import annotations

from mnemosyne.retrieval.planner import QueryPlan
from mnemosyne.retrieval.ranker import ScoredItem


class ContextBuilder:
    """Assembles scored items into a structured context string for the LLM.

    Produces compact, organized sections — not raw dumps.
    """

    def build(
        self,
        items: list[ScoredItem],
        plan: QueryPlan,
        conversation: list[dict[str, str]] | None = None,
    ) -> str:
        """Build structured context from scored items."""
        sections = []

        entities = [i for i in items if i.item_type == "entity"]
        relationships = [i for i in items if i.item_type == "relationship"]
        facts = [i for i in items if i.item_type == "fact"]
        memories = [i for i in items if i.item_type == "memory"]

        if entities:
            lines = []
            for item in entities[:15]:
                e = item.item
                lines.append(f"- {e['name']} (type: {e.get('type', 'unknown')})")
            sections.append("Entities:\n" + "\n".join(lines))

        if relationships:
            lines = []
            for item in relationships[:20]:
                r = item.item
                lines.append(f"- {r['subject']} {r['predicate']} {r['object']}")
            sections.append("Relationships:\n" + "\n".join(lines))

        if facts:
            lines = []
            for item in facts[:15]:
                f = item.item
                lines.append(f"- {f['subject']} {f['predicate']} {f['object']}")
            sections.append("Facts:\n" + "\n".join(lines))

        if memories:
            lines = []
            for item in memories[:5]:
                m = item.item
                text = m.get("text", "")[:200]
                lines.append(f"- \"{text}\"")
            sections.append("Relevant conversations:\n" + "\n".join(lines))

        if not sections:
            return "No relevant information found in memory."

        header = f"Query type: {plan.query_type}"
        return header + "\n\n" + "\n\n".join(sections)
