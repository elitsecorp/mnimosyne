"""Embedding service with Gemini and local (sentence-transformers) backends."""

from __future__ import annotations

import logging
import struct
from typing import Optional

import numpy as np
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


class LocalEmbeddingBackend:
    """Local embedding backend using sentence-transformers.

    Downloads model on first use (~80MB for all-MiniLM-L6-v2).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._dims = 0
        self._loading = False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_loading(self) -> bool:
        return self._loading

    def _load_model(self):
        """Load the sentence-transformers model (called eagerly on startup)."""
        if self._model is not None:
            return
        if self._loading:
            return
        self._loading = True
        try:
            logger.info("Loading local embedding model: %s", self._model_name)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._dims = self._model.get_embedding_dimension()
            logger.info("Local model loaded: %d dimensions", self._dims)
        finally:
            self._loading = False

    def warmup(self) -> None:
        """Eagerly load the model. Call this on startup."""
        self._load_model()

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text string."""
        self._load_model()
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        self._load_model()
        vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vecs]


class GeminiEmbeddingBackend:
    """Gemini embedding backend using Google's API."""

    def __init__(self) -> None:
        from google import genai
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_embedding_model
        self._dims = 0

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


def _create_backend():
    """Create the appropriate embedding backend based on config."""
    if settings.embedding_backend == "local":
        return LocalEmbeddingBackend(settings.local_embedding_model)
    return GeminiEmbeddingBackend()


class EmbeddingService:
    """Handles embedding generation and vector similarity search.

    Supports Gemini and local (sentence-transformers) backends.
    Attempts sqlite-vec first, falls back to FAISS, then brute-force numpy.
    """

    def __init__(self) -> None:
        self._backend = _create_backend()
        self._dims: int = 0
        self._faiss_index = None
        self._faiss_ids: list[int] = []

        backend_name = type(self._backend).__name__
        logger.info("Embedding backend: %s", backend_name)

    @property
    def is_ready(self) -> bool:
        """Check if the embedding backend is ready to use."""
        if isinstance(self._backend, LocalEmbeddingBackend):
            return self._backend.is_loaded
        return True  # Gemini is always ready

    def warmup(self) -> None:
        """Eagerly load the embedding model. Call this on startup."""
        if isinstance(self._backend, LocalEmbeddingBackend):
            logger.info("Warming up local embedding model...")
            self._backend.warmup()
            logger.info("Embedding model ready.")

    def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single text string."""
        vec = self._backend.embed(text)
        if self._dims == 0:
            self._dims = len(vec)
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        vecs = self._backend.embed_batch(texts)
        if self._dims == 0 and vecs:
            self._dims = len(vecs[0])
        return vecs

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

    def search(self, db: Session, query: str, top_k: int = 10, query_vector: list[float] | None = None) -> list[dict]:
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

        dims = self._dims or 384
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

        query_np = np.array(query_vec, dtype=np.float32)
        query_norm = np.linalg.norm(query_np)
        scores = []

        for row in rows:
            try:
                stored_dims = len(row.embedding) // 4
                if stored_dims != len(query_vec):
                    continue
                vec = _deserialize_vector(row.embedding, stored_dims)
                vec_np = np.array(vec, dtype=np.float32)
                sim = float(np.dot(query_np, vec_np) / (query_norm * np.linalg.norm(vec_np) + 1e-10))
                scores.append((row, sim))
            except Exception:
                continue

        scores.sort(key=lambda x: x[1], reverse=True)
        return [
            {"id": r.id, "message_id": r.message_id, "text": r.text, "score": s}
            for r, s in scores[:top_k]
        ]
