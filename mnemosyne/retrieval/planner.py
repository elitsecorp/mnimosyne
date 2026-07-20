"""Query analysis and retrieval strategy planning."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "about", "tell", "me", "what", "who", "whom", "which", "whose",
    "and", "but", "or", "if", "because", "that", "this", "these", "those",
    "it", "its", "my", "your", "his", "her", "our", "their", "i", "you",
    "he", "she", "we", "they", "me", "him", "us", "them",
})

_ENTITY_PATTERNS = [
    (r"(?:who|whom)\s+(?:is|are|was|were)\s+(.+?)(?:\?|$)", "entity_lookup"),
    (r"(?:what|which)\s+(?:is|are|was|were)\s+(.+?)(?:\?|$)", "entity_lookup"),
    (r"tell\s+me\s+about\s+(.+?)(?:\?|$)", "entity_lookup"),
    (r"(?:what|how|where|when|why)\s+(?:does|do|did|is|are|was|were)\s+(\w+)", "relationship_query"),
    (r"(.+?)\s+(?:works|lives|likes|knows|has|is|was|created|made|built|owns|flies|visits)", "relationship_query"),
    (r"(?:what|which)\s+.*?(?:name|type|kind)", "fact_query"),
    (r"(?:remember|recall|last\s+time|before|previously|earlier)", "conversation"),
    (r"(?:something|anything|about|related)", "semantic_query"),
]

_QUERY_TYPE_KEYWORDS = {
    "entity_lookup": {"who", "whom", "whose", "about"},
    "relationship_query": {"does", "do", "did", "works", "lives", "likes", "knows", "has"},
    "fact_query": {"what", "which", "name", "type", "kind"},
    "conversation": {"remember", "recall", "last", "before", "previously", "earlier", "discussed", "said"},
    "semantic_query": {"something", "related", "similar", "about"},
    "summarization": {"summarize", "summary", "overview", "summarise"},
}


@dataclass
class QueryPlan:
    """Determines how a query should be answered."""

    query_type: str = "general"
    detected_entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    max_hops: int = 2
    vector_enabled: bool = True
    graph_enabled: bool = True
    direct_answer_possible: bool = False


class QueryPlanner:
    """Analyzes user queries and determines retrieval strategy.

    Uses deterministic pattern matching — no LLM calls.
    """

    def __init__(self, known_entities: list[str] | None = None) -> None:
        self._known_entities = [e.lower() for e in (known_entities or [])]

    def analyze(self, query: str) -> QueryPlan:
        """Analyze a query and return a retrieval plan."""
        query_lower = query.lower().strip()
        keywords = self._extract_keywords(query_lower)
        query_type = self._classify_query(query_lower, keywords)
        detected = self._detect_entities(query_lower)
        max_hops = self._determine_hops(query_type)
        vector_enabled = self._needs_vector_search(query_type, detected)
        graph_enabled = self._needs_graph_search(query_type)
        direct = self._can_answer_directly(query_type, detected)

        return QueryPlan(
            query_type=query_type,
            detected_entities=detected,
            keywords=keywords,
            max_hops=max_hops,
            vector_enabled=vector_enabled,
            graph_enabled=graph_enabled,
            direct_answer_possible=direct,
        )

    def _extract_keywords(self, query: str) -> list[str]:
        """Remove stopwords and return meaningful keywords."""
        tokens = re.findall(r"[a-z0-9]+", query.lower())
        return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    _QUESTION_WORDS = frozenset({"what", "where", "when", "how", "why", "which", "who", "whom", "whose"})
    _ENTITY_KEYWORDS = frozenset({"who", "whom", "whose", "about"})
    _SCORING_KEYWORDS = frozenset({"does", "do", "did", "works", "lives", "likes", "knows", "has", "remember", "recall", "last", "before", "previously", "earlier", "discussed", "said"})

    def _classify_query(self, query: str, keywords: list[str]) -> str:
        """Classify query type using pattern matching and keywords."""
        all_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
        has_question = bool(all_tokens & self._QUESTION_WORDS)

        scoring_tokens = keywords.copy()
        scoring_tokens.extend(t for t in all_tokens if t in self._QUESTION_WORDS)
        scoring_tokens.extend(t for t in all_tokens if t in self._SCORING_KEYWORDS)

        keyword_scores: dict[str, int] = {}
        for qtype, trigger_words in _QUERY_TYPE_KEYWORDS.items():
            score = sum(1 for kw in scoring_tokens if kw in trigger_words)
            if score > 0:
                keyword_scores[qtype] = score

        if keyword_scores:
            return max(keyword_scores, key=keyword_scores.get)

        if has_question and self._known_entities:
            if any(e.lower() in " ".join(all_tokens) for e in self._known_entities):
                has_entity_keyword = bool(all_tokens & self._ENTITY_KEYWORDS)
                if has_entity_keyword:
                    return "entity_lookup"
                verb_tokens = all_tokens - self._QUESTION_WORDS - {"the", "a", "an", "is", "are", "was", "were", "of", "in", "for", "at", "to", "with", "about"}
                if verb_tokens - {e.lower() for e in self._known_entities}:
                    return "relationship_query"

        if has_question:
            return "general"

        for pattern, qtype in _ENTITY_PATTERNS:
            if re.search(pattern, query):
                return qtype

        if any(e in all_tokens for e in self._known_entities):
            return "entity_lookup"

        return "general"

    def _detect_entities(self, query: str) -> list[str]:
        """Detect known entity names mentioned in the query."""
        found = []
        query_words = set(query.split())
        for entity in self._known_entities:
            if entity in query:
                found.append(entity)
            elif entity.split()[0] in query_words and len(entity.split()) > 1:
                found.append(entity)
        return found

    def _determine_hops(self, query_type: str) -> int:
        """Determine graph traversal depth based on query type."""
        if query_type == "entity_lookup":
            return 1
        if query_type == "relationship_query":
            return 2
        if query_type == "fact_query":
            return 2
        return 2

    def _needs_vector_search(self, query_type: str, entities: list[str]) -> bool:
        """Determine if vector search should be used."""
        if query_type in ("conversation", "semantic_query"):
            return True
        if query_type in ("entity_lookup", "relationship_query", "fact_query"):
            return len(entities) == 0
        return True

    def _needs_graph_search(self, query_type: str) -> bool:
        """Determine if graph search should be used."""
        if query_type == "conversation":
            return False
        return True

    def _can_answer_directly(self, query_type: str, entities: list[str]) -> bool:
        """Check if a simple entity lookup can be answered without LLM."""
        return query_type == "entity_lookup" and len(entities) > 0
