"""Open Information Extraction: offline triple extraction from text.

Rule-based extraction using regex patterns and simple NLP.
No external model downloads needed. Fully offline, zero cost.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_SVO_PATTERNS = [
    re.compile(r"(\b\w+\b)\s+(\w+)\s+(\b\w+\b)", re.IGNORECASE),
]

_BE_PATTERNS = [
    re.compile(r"(\b\w+\b)\s+is\s+(?:a|an|the)\s+(\b\w+\b)", re.IGNORECASE),
    re.compile(r"(\b\w+\b)\s+are\s+(?:a|an|the)\s+(\b\w+\b)", re.IGNORECASE),
]

_HAS_PATTERNS = [
    re.compile(r"(\b\w+\b)\s+has\s+(?:a|an|the)?\s*(\b\w+\b)", re.IGNORECASE),
    re.compile(r"(\b\w+\b)\s+haves?\s+(?:a|an|the)?\s*(\b\w+\b)", re.IGNORECASE),
]

_OWN_PATTERNS = [
    re.compile(r"(\b\w+\b)\s+owns?\s+(?:a|an|the)?\s*(\b\w+\b)", re.IGNORECASE),
    re.compile(r"(\b\w+\b)\s+has\s+(?:a|an|the)?\s*(\b\w+\b)\s+named\s+(\b\w+\b)", re.IGNORECASE),
]

_WORK_PATTERNS = [
    re.compile(r"(\b\w+\b)\s+works?\s+(?:at|for|in)\s+(?:a|an|the)?\s*(\b\w+\b)", re.IGNORECASE),
    re.compile(r"(\b\w+\b)\s+is\s+(?:a|an)\s+employee\s+of\s+(\b\w+\b)", re.IGNORECASE),
]

_LIKE_PATTERNS = [
    re.compile(r"(\b\w+\b)\s+likes?\s+(?:a|an|the)?\s*(\b\w+\b)", re.IGNORECASE),
    re.compile(r"(\b\w+\b)\s+loves?\s+(?:a|an|the)?\s*(\b\w+\b)", re.IGNORECASE),
]

_LOCATED_PATTERNS = [
    re.compile(r"(\b\w+\b)\s+(?:is|are)\s+located\s+in\s+(\b\w+\b)", re.IGNORECASE),
    re.compile(r"(\b\w+\b)\s+lives?\s+in\s+(\b\w+\b)", re.IGNORECASE),
    re.compile(r"(\b\w+\b)\s+is\s+in\s+(\b\w+\b)", re.IGNORECASE),
]

_COMMON_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
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
    "he", "she", "we", "they", "me", "him", "us", "them", "also", "like",
    "think", "know", "want", "said", "say", "going", "got", "get", "well",
    "back", "even", "still", "new", "way", "use", "thing", "make", "many",
})


@dataclass
class Triple:
    """An extracted (subject, predicate, object) triple."""

    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    source_sentence: str = ""


@dataclass
class OpenIEResult:
    """Result of OpenIE extraction from a text chunk."""

    triples: list[Triple] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)


def extract_triples(text: str) -> OpenIEResult:
    """Extract (subject, predicate, object) triples from text.

    Uses regex patterns and simple heuristics.
    Fully offline, no API calls, no model downloads.
    """
    triples = []
    entities = []
    seen_entities = set()

    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 5:
            continue

        sent_triples = _extract_from_sentence(sent)
        triples.extend(sent_triples)

        for t in sent_triples:
            _add_entity(entities, seen_entities, t.subject)
            _add_entity(entities, seen_entities, t.object)

    deduped = _dedup_triples(triples)

    return OpenIEResult(triples=deduped, entities=entities)


def _extract_from_sentence(sent: str) -> list[Triple]:
    """Extract triples from a single sentence."""
    triples = []

    for pattern, predicate in [
        (_OWN_PATTERNS[0], "owns"),
        (_OWN_PATTERNS[1], "owns"),
        (_HAS_PATTERNS[0], "has"),
        (_HAS_PATTERNS[1], "has"),
        (_WORK_PATTERNS[0], "works_for"),
        (_WORK_PATTERNS[1], "works_for"),
        (_LIKE_PATTERNS[0], "likes"),
        (_LIKE_PATTERNS[1], "likes"),
        (_LOCATED_PATTERNS[0], "located_in"),
        (_LOCATED_PATTERNS[1], "located_in"),
        (_LOCATED_PATTERNS[2], "located_in"),
        (_BE_PATTERNS[0], "is_a"),
        (_BE_PATTERNS[1], "is_a"),
    ]:
        for match in pattern.finditer(sent):
            groups = match.groups()
            if len(groups) >= 2:
                subj = _clean_token(groups[0])
                obj = _clean_token(groups[1])
                if subj and obj and subj != obj:
                    triples.append(Triple(
                        subject=subj,
                        predicate=predicate,
                        object=obj,
                        confidence=0.75,
                        source_sentence=sent,
                    ))

    return triples


def _clean_token(token: str) -> str:
    """Clean and validate a token."""
    token = token.strip()
    if not token or len(token) < 2:
        return ""
    if token.lower() in _COMMON_WORDS:
        return ""
    if not token[0].isalpha():
        return ""
    return token


def _add_entity(entities: list[dict], seen: set, name: str) -> None:
    """Add an entity to the list if not already seen."""
    key = name.lower()
    if key not in seen and name:
        seen.add(key)
        entities.append({
            "name": name,
            "type": "entity",
            "confidence": 0.7,
        })


def _dedup_triples(triples: list[Triple]) -> list[Triple]:
    """Remove duplicate triples, keeping highest confidence."""
    seen: dict[tuple, Triple] = {}
    for t in triples:
        key = (t.subject.lower(), t.predicate.lower(), t.object.lower())
        existing = seen.get(key)
        if not existing or t.confidence > existing.confidence:
            seen[key] = t
    return list(seen.values())
