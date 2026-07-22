"""Structured context construction for LLM consumption."""

from __future__ import annotations

from mnemosyne.retrieval.planner import QueryPlan
from mnemosyne.retrieval.ranker import ScoredItem
from mnemosyne.retrieval.summarizer import summarize_conversation, extract_key_entities


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

        if conversation:
            summary = summarize_conversation(conversation, max_sentences=4)
            if summary:
                sections.append("Session summary:\n" + summary)

            key_terms = extract_key_entities(conversation)
            if key_terms:
                sections.append("Key terms discussed: " + ", ".join(key_terms[:10]))

        entities = [i for i in items if i.item_type == "entity"]
        relationships = [i for i in items if i.item_type == "relationship"]
        facts = [i for i in items if i.item_type == "fact"]
        memories = [i for i in items if i.item_type == "memory"]

        if entities:
            lines = []
            for item in entities[:25]:
                e = item.item
                conf = e.get('confidence', 0)
                lines.append(f"- {e['name']} (type: {e.get('type', 'unknown')}, confidence: {conf:.2f})")
            sections.append("Entities:\n" + "\n".join(lines))

        if relationships:
            lines = []
            for item in relationships[:30]:
                r = item.item
                conf = r.get('confidence', 0)
                last_seen = r.get('last_seen', '')
                if last_seen:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                        date_str = dt.strftime('%Y-%m-%d')
                        lines.append(f"- {r['subject']} {r['predicate']} {r['object']} (confidence: {conf:.2f}, seen: {date_str})")
                    except (ValueError, TypeError):
                        lines.append(f"- {r['subject']} {r['predicate']} {r['object']} (confidence: {conf:.2f})")
                else:
                    lines.append(f"- {r['subject']} {r['predicate']} {r['object']} (confidence: {conf:.2f})")
            sections.append("Relationships:\n" + "\n".join(lines))

        if facts:
            lines = []
            for item in facts[:20]:
                f = item.item
                lines.append(f"- {f['subject']} {f['predicate']} {f['object']}")
            sections.append("Facts:\n" + "\n".join(lines))

        if memories:
            lines = []
            for item in memories[:10]:
                m = item.item
                score = m.get("score", 0)
                text = m.get("text", "")[:400]
                lines.append(f"- [score: {score:.2f}] {text}")
            sections.append("Relevant conversations:\n" + "\n".join(lines))

        if not sections:
            return "No relevant information found in memory."

        header = f"Query type: {plan.query_type}"
        return header + "\n\n" + "\n\n".join(sections)
