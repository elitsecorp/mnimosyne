"""Vector/semantic memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from mnemosyne.embeddings import EmbeddingService


@dataclass
class MemoryResult:
    """Result of vector memory retrieval."""

    memories: list[dict] = field(default_factory=list)


class MemoryRetriever:
    """Retrieves semantically similar past messages using vector search.

    Always returns top_k results, prioritizing those above the similarity
    threshold but filling with best remaining if fewer match.
    """

    def __init__(self, embeddings: EmbeddingService) -> None:
        self._embeddings = embeddings

    def retrieve(
        self,
        db: Session,
        query: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        query_vector: list[float] | None = None,
    ) -> MemoryResult:
        """Search for semantically similar past messages.

        Always returns up to top_k results. Prioritizes results above
        min_similarity, but fills remaining slots with best available.
        """
        if not query.strip():
            return MemoryResult()

        raw_results = self._embeddings.search(db, query, top_k=top_k * 2, query_vector=query_vector)

        above_threshold = [r for r in raw_results if r.get("score", 0) >= min_similarity]

        if len(above_threshold) >= top_k:
            selected = above_threshold[:top_k]
        else:
            remaining = [r for r in raw_results if r not in above_threshold]
            selected = above_threshold + remaining[:top_k - len(above_threshold)]

        memories = []
        for r in selected:
            text = r.get("text", "")
            if text.strip().lower() == query.strip().lower():
                continue
            memories.append({
                "id": r["id"],
                "message_id": r["message_id"],
                "text": text,
                "score": r.get("score", 0),
            })

        return MemoryResult(memories=memories)
