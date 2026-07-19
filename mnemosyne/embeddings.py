"""Embedding service: Gemini embeddings with sqlite-vec or FAISS vector storage."""

from __future__ import annotations

import logging
import struct
from typing import Optional

import numpy as np
from google import genai
from google.genai import types
from sqlalchemy.orm import Session

from mnemosyne.config import settings
from mnemosyne.models import Embedding

logger = logging.getLogger(__name__)

_USE_SQLITE_VEC = False
_USE_FAISS = False

try:
    import sqlite_vec  # noqa: F401
    _USE_SQLITE_VEC = True
except ImportError:
    pass

if not _USE_SQLITE_VEC:
    try:
        import faiss  # noqa: F401
        _USE_FAISS = True
    except ImportError:
        pass

logger.info("Vector backend: %s", "sqlite-vec" if _USE_SQLITE_VEC else ("FAISS" if _USE_FAISS else "numpy brute-force"))


def _serialize_vector(vec: list[float]) -> bytes:
    """Pack a float list into bytes for BLOB storage."""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_vector(data: bytes, dims: int) -> list[float]:
    """Unpack bytes back into a float list."""
    return list(struct.unpack(f"{dims}f", data))


class EmbeddingService:
    """Handles embedding generation and vector similarity search.

    Uses Gemini embeddings. Attempts sqlite-vec first, falls back to FAISS,
    then brute-force numpy.
    """

    def __init__(self, client: Optional[genai.Client] = None) -> None:
        self._client = client or genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_embedding_model
        self._dims: int = 0
        self._faiss_index = None
        self._faiss_ids: list[int] = []

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text string."""
        response = self._client.models.embed_content(
            model=self._model,
            contents=text,
        )
        vector = response.embeddings[0].values
        if self._dims == 0:
            self._dims = len(vector)
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        response = self._client.models.embed_content(
            model=self._model,
            contents=texts,
        )
        vectors = [e.values for e in response.embeddings]
        if self._dims == 0 and vectors:
            self._dims = len(vectors[0])
        return vectors

    def store_embedding(self, db: Session, message_id: int, text: str, embedding: list[float]) -> Embedding:
        """Persist an embedding to the database."""
        record = Embedding(
            message_id=message_id,
            text=text,
            embedding=_serialize_vector(embedding),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    def search(self, db: Session, query: str, top_k: int = 5, query_vector: list[float] | None = None) -> list[dict]:
        """Search for the most similar stored embeddings.

        Args:
            query_vector: Pre-computed embedding. If None, will embed the query.

        Returns list of dicts with keys: id, message_id, text, score.
        """
        query_vec = query_vector if query_vector is not None else self.embed(query)
        if _USE_SQLITE_VEC:
            return self._search_sqlite_vec(db, query_vec, top_k)
        if _USE_FAISS:
            return self._search_faiss(db, query_vec, top_k)
        return self._search_bruteforce(db, query_vec, top_k)

    def _search_sqlite_vec(self, db: Session, query_vec: list[float], top_k: int) -> list[dict]:
        """Use sqlite-vec for vector similarity search."""
        try:
            result = db.execute(
                """
                SELECT e.id, e.message_id, e.text,
                       vec_distance_cosine(e.embedding, ?) AS score
                FROM embeddings e
                ORDER BY score ASC
                LIMIT ?
                """,
                (_serialize_vector(query_vec), top_k),
            )
            return [
                {"id": row[0], "message_id": row[1], "text": row[2], "score": float(row[3])}
                for row in result.fetchall()
            ]
        except Exception:
            logger.debug("sqlite-vec query failed, falling back to brute-force")
            return self._search_bruteforce(db, query_vec, top_k)

    def _search_faiss(self, db: Session, query_vec: list[float], top_k: int) -> list[dict]:
        """Use FAISS for vector similarity search."""
        try:
            if self._faiss_index is None:
                self._rebuild_faiss_index(db)
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                return []

            query_np = np.array([query_vec], dtype=np.float32)
            distances, indices = self._faiss_index.search(query_np, min(top_k, self._faiss_index.ntotal))

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self._faiss_ids):
                    continue
                emb_id = self._faiss_ids[idx]
                emb = db.get(Embedding, emb_id)
                if emb:
                    results.append({
                        "id": emb.id,
                        "message_id": emb.message_id,
                        "text": emb.text,
                        "score": float(dist),
                    })
            return results
        except Exception:
            logger.debug("FAISS search failed, falling back to brute-force")
            return self._search_bruteforce(db, query_vec, top_k)

    def _rebuild_faiss_index(self, db: Session) -> None:
        """Rebuild the FAISS index from all stored embeddings."""
        import faiss

        rows = db.query(Embedding).all()
        if not rows:
            return

        dims = self._dims or 768
        index = faiss.IndexFlatIP(dims)
        vectors = []
        ids = []
        for row in rows:
            vec = _deserialize_vector(row.embedding, dims)
            vectors.append(vec)
            ids.append(row.id)

        index.add(np.array(vectors, dtype=np.float32))
        self._faiss_index = index
        self._faiss_ids = ids

    def _search_bruteforce(self, db: Session, query_vec: list[float], top_k: int) -> list[dict]:
        """Brute-force cosine similarity search over all embeddings."""
        rows = db.query(Embedding).all()
        if not rows:
            return []

        dims = self._dims or len(query_vec)
        query_np = np.array(query_vec, dtype=np.float32)
        scores = []

        for row in rows:
            vec = _deserialize_vector(row.embedding, dims)
            vec_np = np.array(vec, dtype=np.float32)
            sim = float(np.dot(query_np, vec_np) / (np.linalg.norm(query_np) * np.linalg.norm(vec_np) + 1e-10))
            scores.append((row, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"id": r.id, "message_id": r.message_id, "text": r.text, "score": s}
            for r, s in scores[:top_k]
        ]
