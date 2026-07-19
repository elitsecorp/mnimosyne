"""Offline extractive summarization — no LLM calls."""

from __future__ import annotations

import math
import re
from collections import Counter

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
    "he", "she", "we", "they", "me", "him", "us", "them", "like", "think",
    "know", "want", "said", "say", "going", "got", "get", "also", "well",
    "back", "even", "still", "new", "way", "use", "thing", "make", "many",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _sentence_split(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _word_frequencies(text: str) -> dict[str, float]:
    """Compute TF-weighted word frequencies."""
    words = _tokenize(text)
    words = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    if not words:
        return {}
    counts = Counter(words)
    max_count = max(counts.values())
    return {w: c / max_count for w, c in counts.items()}


def _sentence_score(sentence: str, word_freqs: dict[str, float]) -> float:
    """Score a sentence based on word frequency overlap."""
    words = _tokenize(sentence)
    words = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    if not words:
        return 0.0
    score = sum(word_freqs.get(w, 0) for w in words)
    return score / len(words)


def summarize(text: str, max_sentences: int = 3) -> str:
    """Extractive summary: pick the most important sentences.

    Uses TF-based scoring. No LLM calls.
    """
    if not text or not text.strip():
        return ""

    sentences = _sentence_split(text)
    if len(sentences) <= max_sentences:
        return text.strip()

    word_freqs = _word_frequencies(text)
    scored = [(s, _sentence_score(s, word_freqs)) for s in sentences]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:max_sentences]

    ordered = sorted(top, key=lambda x: sentences.index(x[0]))
    return " ".join(s for s, _ in ordered)


def summarize_conversation(messages: list[dict[str, str]], max_sentences: int = 5) -> str:
    """Summarize a conversation into key points.

    Args:
        messages: List of {"role": "user"|"assistant", "content": "..."} dicts.
        max_sentences: Maximum sentences in the summary.

    Returns:
        Extractive summary of the conversation.
    """
    if not messages:
        return ""

    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if content:
            parts.append(f"{role}: {content}")

    full_text = " ".join(parts)
    return summarize(full_text, max_sentences=max_sentences)


def extract_key_entities(messages: list[dict[str, str]]) -> list[str]:
    """Extract frequently mentioned terms from conversation.

    Returns terms sorted by frequency, excluding stopwords.
    """
    word_counts: dict[str, int] = {}
    for msg in messages:
        content = msg.get("content", "")
        words = _tokenize(content)
        for w in words:
            if w not in _STOPWORDS and len(w) > 2:
                word_counts[w] = word_counts.get(w, 0) + 1

    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:20]]
