"""Memory seeding: generate and ingest test memories for recall optimization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from mnemosyne.embeddings import EmbeddingService
from mnemosyne.extraction import extract_memory
from mnemosyne.graph import GraphService
from mnemosyne.llm import LLMService
from mnemosyne.models import Entity, Fact, Message, Relationship

logger = logging.getLogger(__name__)


@dataclass
class SeedResult:
    """Result of seeding a single memory."""
    text: str
    entities_found: int
    relationships_found: int
    facts_found: int
    success: bool
    error: str = ""


@dataclass
class SeedReport:
    """Summary of memory seeding."""
    total: int
    successful: int
    failed: int
    total_entities: int
    total_relationships: int
    total_facts: int
    results: list[SeedResult] = field(default_factory=list)


# Rich test memories covering different relationship types and depths
SEED_MEMORIES = [
    # Personal identity
    "My name is Alice. I'm a senior software engineer.",
    "I work at Google on the Gemini team.",
    "I live in San Francisco, California.",
    "My dog Max is a golden retriever. He's 3 years old.",
    "I have a cat named Luna who is very playful.",

    # Work and projects
    "I'm building Mnemosyne, a persistent memory system for LLMs.",
    "Mnemosyne uses Python, SQLAlchemy, and NetworkX for the knowledge graph.",
    "The project has a web UI built with vanilla HTML and Cytoscape.js for graph visualization.",
    "I work primarily on the retrieval pipeline and context generation.",
    "My team lead is David. He manages the Gemini integration.",

    # Colleagues and relationships
    "My colleague Bob works on the search ranking team.",
    "Alice knows Carol from college. They studied computer science together.",
    "David manages a team of 8 engineers including Alice and Bob.",
    "Bob's wife Sarah is a data scientist at Meta.",
    "Carol moved to New York last year.",

    # Technical knowledge
    "Python is a high-level programming language known for its readability.",
    "SQLAlchemy is a Python SQL toolkit and ORM for database management.",
    "NetworkX is a Python library for studying graphs and networks.",
    "FastAPI is a modern web framework for building APIs with Python.",
    "SQLite is a lightweight, serverless database engine.",

    # Projects and interests
    "I'm interested in knowledge graphs and semantic search.",
    "I've been reading about existentialism and philosophy lately.",
    "I enjoy hiking in Marin County on weekends.",
    "I'm learning to play the guitar. I practice every evening.",
    "My favorite programming language is Python, but I also know Rust.",

    # Events and timeline
    "I started working at Google in January 2023.",
    "Mnemosyne project began in March 2024.",
    "I adopted Max from a shelter in October 2022.",
    "I graduated from Stanford University in 2018.",
    "I attended a Python conference last month.",

    # Preferences and habits
    "I prefer dark mode for all my applications.",
    "I drink coffee every morning. I like it black.",
    "I usually wake up at 6 AM and go for a run.",
    "I read before bed every night. Currently reading Sartre.",
    "I prefer working in quiet environments.",

    # Health and wellbeing
    "I go to the gym three times a week.",
    "I have a meditation practice. I meditate for 10 minutes daily.",
    "I try to eat healthy. I cook most of my meals at home.",
    "I get about 7 hours of sleep each night.",
    "I practice yoga on Sunday mornings.",

    # Family and friends
    "My parents live in Portland, Oregon.",
    "I have a younger sister named Emma. She's a doctor.",
    "My best friend from high school is Jake. He's a lawyer.",
    "I call my parents every Sunday evening.",
    "My family has a tradition of hiking together during holidays.",

    # Goals and aspirations
    "My goal this year is to launch Mnemosyne publicly.",
    "I want to write a book about AI and memory systems.",
    "I'm planning to learn Rust this year.",
    "I hope to travel to Japan next spring.",
    "I want to improve my public speaking skills.",

    # Technical details about Mnemosyne
    "Mnemosyne uses sentence-transformers for local embeddings.",
    "The knowledge graph stores entities, relationships, and facts.",
    "The retrieval pipeline does 3-hop BFS traversal for graph queries.",
    "Vector search uses cosine similarity with a minimum threshold of 0.6.",
    "The system consolidates knowledge every 5 messages automatically.",
]


class MemorySeeder:
    """Seeds the database with test memories for recall optimization."""

    def __init__(
        self,
        llm: LLMService | None = None,
        embeddings: EmbeddingService | None = None,
        graph: GraphService | None = None,
    ) -> None:
        self._llm = llm or LLMService()
        self._embeddings = embeddings or EmbeddingService()
        self._graph = graph or GraphService()

    def seed(self, db: Session, memories: list[str] | None = None) -> SeedReport:
        """Seed the database with test memories.

        Args:
            db: Database session.
            memories: List of memory texts. Uses SEED_MEMORIES if None.

        Returns:
            SeedReport with statistics.
        """
        memories = memories or SEED_MEMORIES
        report = SeedReport(
            total=len(memories),
            successful=0,
            failed=0,
            total_entities=0,
            total_relationships=0,
            total_facts=0,
        )

        for i, text in enumerate(memories):
            logger.info("Seeding memory %d/%d: %s", i + 1, len(memories), text[:50])
            result = self._seed_one(db, text)
            report.results.append(result)

            if result.success:
                report.successful += 1
                report.total_entities += result.entities_found
                report.total_relationships += result.relationships_found
                report.total_facts += result.facts_found
            else:
                report.failed += 1
                logger.warning("Failed to seed: %s - %s", text[:30], result.error)

        # Sync graph to DB
        self._graph.save(db)

        logger.info(
            "Seeding complete: %d/%d successful, %d entities, %d relationships, %d facts",
            report.successful, report.total,
            report.total_entities, report.total_relationships, report.total_facts,
        )
        return report

    def _seed_one(self, db: Session, text: str) -> SeedResult:
        """Seed a single memory into the database."""
        try:
            # Store as a user message
            msg = Message(role="user", content=text)
            db.add(msg)
            db.commit()
            db.refresh(msg)

            # Embed and store
            embedding = self._embeddings.embed(text)
            self._embeddings.store_embedding(db, msg.id, text, embedding)

            # Extract knowledge
            extraction = extract_memory(self._llm, text, text)

            entities_found = len(extraction.entities)
            relationships_found = len(extraction.relationships)
            facts_found = len(extraction.facts)

            # Store entities
            for ent in extraction.entities:
                existing = db.query(Entity).filter_by(name=ent.name).first()
                if existing:
                    if ent.confidence > existing.confidence:
                        existing.type = ent.type
                        existing.confidence = ent.confidence
                else:
                    db.add(Entity(name=ent.name, type=ent.type, confidence=ent.confidence))
                self._graph.add_entity(ent.name, ent.type, ent.confidence)

            # Store relationships
            for rel in extraction.relationships:
                existing = db.query(Relationship).filter_by(
                    subject=rel.subject, predicate=rel.predicate, object=rel.object,
                ).first()
                if not existing:
                    db.add(Relationship(
                        subject=rel.subject,
                        predicate=rel.predicate,
                        object=rel.object,
                        confidence=rel.confidence,
                    ))
                self._graph.add_relationship(rel.subject, rel.predicate, rel.object, rel.confidence)

            # Store facts
            for fact in extraction.facts:
                existing = db.query(Fact).filter_by(
                    subject=fact.subject, predicate=fact.predicate, object=fact.object,
                ).first()
                if not existing:
                    db.add(Fact(
                        subject=fact.subject,
                        predicate=fact.predicate,
                        object=fact.object,
                        source_message=text,
                    ))

            db.commit()

            return SeedResult(
                text=text,
                entities_found=entities_found,
                relationships_found=relationships_found,
                facts_found=facts_found,
                success=True,
            )

        except Exception as e:
            return SeedResult(
                text=text,
                entities_found=0,
                relationships_found=0,
                facts_found=0,
                success=False,
                error=str(e),
            )

    def clear_memories(self, db: Session) -> None:
        """Clear all seeded memories (keeps Me entity and owner relationships)."""
        from sqlalchemy import text

        # Delete embeddings
        db.execute(text("DELETE FROM embeddings"))

        # Delete facts
        db.execute(text("DELETE FROM facts"))

        # Delete non-owner relationships
        db.execute(text("DELETE FROM relationships WHERE is_owner = 0"))

        # Delete non-owner entities (keep Me and its connections)
        db.execute(text("DELETE FROM entities WHERE name != 'Me'"))

        # Delete messages
        db.execute(text("DELETE FROM messages"))

        db.commit()

        # Reload graph
        self._graph.load_from_db(db)

        logger.info("Seeded memories cleared")
