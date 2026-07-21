"""Data ingestion service for importing text and documents into Mnemosyne."""

from __future__ import annotations

import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, UTC

from mnemosyne.database import get_session_factory
from mnemosyne.embeddings import EmbeddingService
from mnemosyne.extraction import extract_memory
from mnemosyne.graph import GraphService
from mnemosyne.llm import LLMService
from mnemosyne.memory import MemoryEngine
from mnemosyne.models import Embedding, Entity, Fact, Message, Relationship

logger = logging.getLogger(__name__)

MAX_CHUNK_SIZE = 500
MIN_CHUNK_SIZE = 50
_paragraph_split = re.compile(r"\n\s*\n")
_sentence_split = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ImportJob:
    """Tracks the state of a background import job."""

    id: str
    status: str = "running"
    total_chunks: int = 0
    processed_chunks: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    embeddings_created: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    completed_at: str | None = None
    _cancelled: bool = field(default=False, repr=False)


class ImportService:
    """Handles text ingestion, chunking, embedding, and knowledge extraction.

    Uses background threads for long-running imports.
    """

    def __init__(self, engine: MemoryEngine | None = None) -> None:
        self._engine = engine or MemoryEngine()
        self._jobs: dict[str, ImportJob] = {}
        self._executor = ThreadPoolExecutor(max_workers=2)

    def import_text(self, text: str) -> str:
        """Start a background import job for the given text.

        Returns the job ID immediately.
        """
        job_id = str(uuid.uuid4())[:8]
        job = ImportJob(id=job_id)
        self._jobs[job_id] = job
        self._executor.submit(self._run_import, job, text)
        return job_id

    def import_file(self, filename: str, content: str, content_type: str) -> str:
        """Import a file. Delegates to import_text after parsing."""
        return self.import_text(content)

    def get_job(self, job_id: str) -> dict | None:
        """Get job status and progress."""
        job = self._jobs.get(job_id)
        if not job:
            return None
        return {
            "id": job.id,
            "status": job.status,
            "total_chunks": job.total_chunks,
            "processed_chunks": job.processed_chunks,
            "entities_extracted": job.entities_extracted,
            "relationships_extracted": job.relationships_extracted,
            "embeddings_created": job.embeddings_created,
            "errors": job.errors,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
        }

    def cancel_job(self, job_id: str) -> bool:
        """Request cancellation of a running job."""
        job = self._jobs.get(job_id)
        if not job or job.status != "running":
            return False
        job._cancelled = True
        return True

    def _run_import(self, job: ImportJob, text: str) -> None:
        """Execute the full import pipeline in a background thread."""
        try:
            chunks = self._chunk_text(text)
            job.total_chunks = len(chunks)
            logger.info("Import %s: %d chunks to process", job.id, len(chunks))

            for chunk in chunks:
                if job._cancelled:
                    job.status = "cancelled"
                    job.completed_at = datetime.now(UTC).isoformat()
                    return

                try:
                    self._process_chunk(job, chunk)
                    job.processed_chunks += 1
                except Exception as e:
                    error_msg = f"Chunk {job.processed_chunks}: {str(e)[:200]}"
                    job.errors.append(error_msg)
                    logger.error("Import chunk error: %s", error_msg)
                    job.processed_chunks += 1

            job.status = "completed"
            job.completed_at = datetime.now(UTC).isoformat()
            logger.info(
                "Import %s completed: %d chunks, %d entities, %d relationships",
                job.id,
                job.processed_chunks,
                job.entities_extracted,
                job.relationships_extracted,
            )
        except Exception as e:
            job.status = "error"
            job.errors.append(str(e)[:500])
            job.completed_at = datetime.now(UTC).isoformat()
            logger.error("Import %s failed: %s", job.id, e)

    def _process_chunk(self, job: ImportJob, chunk: str) -> None:
        """Process a single text chunk: store, embed, extract using OpenIE (zero cost)."""
        db = self._engine._session_factory()
        try:
            msg = Message(role="imported", content=chunk)
            db.add(msg)
            db.commit()
            db.refresh(msg)

            embedding = self._engine._embeddings.embed(chunk)
            self._engine._embeddings.store_embedding(db, msg.id, chunk, embedding)
            job.embeddings_created += 1

            from mnemosyne.services.openie import extract_triples
            result = extract_triples(chunk)

            for ent in result.entities:
                existing = db.query(Entity).filter_by(name=ent["name"]).first()
                if existing:
                    if ent["confidence"] > existing.confidence:
                        existing.type = ent["type"]
                        existing.confidence = ent["confidence"]
                else:
                    db.add(Entity(name=ent["name"], type=ent["type"], confidence=ent["confidence"]))
                job.entities_extracted += 1

            for triple in result.triples:
                existing = db.query(Relationship).filter_by(
                    subject=triple.subject, predicate=triple.predicate, object=triple.object,
                ).first()
                if not existing:
                    db.add(Relationship(
                        subject=triple.subject,
                        predicate=triple.predicate,
                        object=triple.object,
                        confidence=triple.confidence,
                    ))
                    db.add(Fact(
                        subject=triple.subject,
                        predicate=triple.predicate,
                        object=triple.object,
                        source_message=triple.source_sentence[:500] if triple.source_sentence else chunk[:500],
                    ))
                job.relationships_extracted += 1

            db.commit()
        finally:
            db.close()

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        """Split text into chunks: paragraphs first, then sentences if too long."""
        paragraphs = _paragraph_split.split(text)
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) <= MAX_CHUNK_SIZE:
                if len(current) + len(para) + 2 <= MAX_CHUNK_SIZE:
                    current = f"{current}\n\n{para}" if current else para
                else:
                    if current:
                        chunks.append(current)
                    current = para
            else:
                if current:
                    chunks.append(current)
                    current = ""
                sentences = _sentence_split.split(para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if len(current) + len(sent) + 1 <= MAX_CHUNK_SIZE:
                        current = f"{current} {sent}" if current else sent
                    else:
                        if current:
                            chunks.append(current)
                        if len(sent) > MAX_CHUNK_SIZE:
                            for i in range(0, len(sent), MAX_CHUNK_SIZE):
                                chunks.append(sent[i : i + MAX_CHUNK_SIZE])
                            current = ""
                        else:
                            current = sent

        if current and len(current) >= MIN_CHUNK_SIZE:
            chunks.append(current)

        return chunks if chunks else [text[:MAX_CHUNK_SIZE]]
