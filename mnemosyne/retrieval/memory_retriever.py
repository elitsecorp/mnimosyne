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

    Only used when graph retrieval alone is insufficient.
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

        Args:
            query_vector: Pre-computed embedding to avoid duplicate API call.
        """
        if not query.strip():
            return MemoryResult()

        raw_results = self._embeddings.search(db, query, top_k=top_k, query_vector=query_vector)

        memories = []
        for r in raw_results:
            score = r.get("score", 0)
            if score >= min_similarity:
                memories.append({
                    "id": r["id"],
                    "message_id": r["message_id"],
                    "text": r["text"],
                    "score": score,
                })

        return MemoryResult(memories=memories)
