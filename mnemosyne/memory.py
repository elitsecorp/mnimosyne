"""Memory engine: orchestrates the full chat pipeline."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from mnemosyne.config import settings
from mnemosyne.database import get_session_factory
from mnemosyne.embeddings import EmbeddingService
from mnemosyne.extraction import extract_memory
from mnemosyne.graph import GraphService
from mnemosyne.llm import LLMService
from mnemosyne.models import Fact, Message
from mnemosyne.prompts import build_chat_messages
from mnemosyne.retrieval import RetrievalService

logger = logging.getLogger(__name__)


class MemoryEngine:
    """Core memory engine. Coordinates all services through the chat pipeline.

    This is the main entry point for the memory system. It orchestrates:
    message storage, embedding, retrieval, LLM calls, and knowledge extraction.
    """

    def __init__(
        self,
        embeddings: EmbeddingService | None = None,
        graph: GraphService | None = None,
        llm: LLMService | None = None,
        retrieval: RetrievalService | None = None,
    ) -> None:
        self._embeddings = embeddings or EmbeddingService()
        self._graph = graph or GraphService()
        self._llm = llm or LLMService()
        self._retrieval = retrieval or RetrievalService(self._embeddings, self._graph)
        self._session_factory = get_session_factory()
        self._message_count = 0
        self._auto_consolidate_interval = 5

    def chat(self, message: str, session_id: int | None = None) -> dict:
        """Process a user message through the full memory pipeline.

        Args:
            message: User message text.
            session_id: Chat session ID. Creates new session if None.

        Returns dict with 'response', 'pipeline', and 'session_id' keys.
        """
        db = self._session_factory()
        try:
            return self._run_pipeline(db, message, session_id)
        finally:
            db.close()

    def _run_pipeline(self, db: Session, message: str, session_id: int | None = None) -> str:
        """Execute the full chat pipeline within a single database session."""
        # 0. Resolve session
        if session_id is None:
            from mnemosyne.models import ChatSession
            session = ChatSession(title=message[:50])
            db.add(session)
            db.commit()
            db.refresh(session)
            session_id = session.id
        else:
            from mnemosyne.models import ChatSession
            session = db.get(ChatSession, session_id)
            if not session:
                session = ChatSession(title=message[:50])
                db.add(session)
                db.commit()
                db.refresh(session)
                session_id = session.id

        # 1. Store user message
        user_msg = Message(session_id=session_id, role="user", content=message)
        db.add(user_msg)
        db.commit()
        db.refresh(user_msg)
        logger.debug("Stored user message id=%d in session %d", user_msg.id, session_id)

        # 2. Embed and store user message for future vector search
        user_embedding = self._embeddings.embed(message)
        self._embeddings.store_embedding(db, user_msg.id, message, user_embedding)

        # 3. Load recent conversation history for this session
        recent_msgs = (
            db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.timestamp.desc())
            .limit(20)
            .all()
        )
        recent_msgs.reverse()
        conversation = [{"role": m.role, "content": m.content} for m in recent_msgs[:-1]]

        # 3. Run deterministic context pipeline
        from mnemosyne.retrieval.pipeline import ContextPipeline
        pipeline = ContextPipeline(
            self._embeddings,
            self._graph,
            max_hops=2,
            min_confidence=0.0,
            token_budget=8000,
        )
        result = pipeline.run(db, message, conversation, query_vector=user_embedding)

        # 4. Extract vector memories for prompt
        vector_context = ""
        if result.memory_result and result.memory_result.get("memories"):
            memories = result.memory_result["memories"]
            vector_context = "\n".join(
                f"- [{m.get('score', 0):.2f}] {m.get('text', '')[:300]}"
                for m in memories[:10]
            )

        # 5. Build LLM messages with pipeline context
        messages = build_chat_messages(
            conversation=conversation,
            vector_memories=vector_context,
            ontology_facts=result.context,
            user_message=message,
        )

        # 6. Call LLM
        response = self._llm.chat(messages)

        # 6. Store assistant message
        assistant_msg = Message(session_id=session_id, role="assistant", content=response)
        db.add(assistant_msg)
        db.commit()
        db.refresh(assistant_msg)
        logger.debug("Stored assistant message id=%d in session %d", assistant_msg.id, session_id)

        # 7. Extract new knowledge (skip for short or non-informative messages)
        extraction = None
        if self._should_extract(message, response):
            extraction = extract_memory(self._llm, message, response)

        # 8. Store ontology (entities, relationships, facts)
        if extraction:
            self._store_ontology(db, extraction, source_message=message)

        # 9. Embed assistant response
        assistant_embedding = self._embeddings.embed(response)
        self._embeddings.store_embedding(db, assistant_msg.id, response, assistant_embedding)

        # 10. Sync graph to DB
        self._graph.save(db)

        # 11. Auto-consolidation every N prompts
        self._message_count += 1
        if self._message_count % self._auto_consolidate_interval == 0:
            self._auto_consolidate(db)

        if extraction:
            logger.info(
                "Chat processed: %d entities, %d relationships, %d facts extracted",
                len(extraction.entities),
                len(extraction.relationships),
                len(extraction.facts),
            )
        else:
            logger.info("Chat processed (extraction skipped)")

        pipeline_meta = {
            "query_plan": {
                "type": result.plan.query_type,
                "detected_entities": result.plan.detected_entities,
                "keywords": result.plan.keywords,
                "max_hops": result.plan.max_hops,
                "graph_enabled": result.plan.graph_enabled,
                "vector_enabled": result.plan.vector_enabled,
            },
            "resolved_entities": [
                {"name": e.name, "type": e.type, "confidence": e.confidence, "method": e.match_method, "score": round(e.match_score, 3)}
                for e in result.resolved_entities
            ],
            "graph_result": result.graph_result,
            "memory_result": {
                "memories": result.memory_result.get("memories", [])[:5] if result.memory_result else [],
            },
            "context_sent": result.context,
            "stats": result.stats,
            "extraction": {
                "entities": [{"name": e.name, "type": e.type, "confidence": e.confidence} for e in extraction.entities] if extraction else [],
                "relationships": [{"subject": r.subject, "predicate": r.predicate, "object": r.object} for r in extraction.relationships] if extraction else [],
                "facts": [{"subject": f.subject, "predicate": f.predicate, "object": f.object} for f in extraction.facts] if extraction else [],
            },
        }

        return {"response": response, "pipeline": pipeline_meta, "session_id": session_id}

    _SKIP_PHRASES = {
        "ok", "okay", "yes", "no", "thanks", "thank you", "bye", "hello",
        "hi", "hey", "sure", "got it", "understood", "right", "cool",
        "great", "nice", "awesome", "haha", "lol", "ok.", "yes.", "no.",
    }

    @classmethod
    def _should_extract(cls, user_message: str, assistant_response: str) -> bool:
        """Decide whether to run knowledge extraction.

        Skips extraction for short greetings, acknowledgments, and trivial exchanges.
        """
        msg_lower = user_message.strip().lower().rstrip(".!?")
        if msg_lower in cls._SKIP_PHRASES:
            return False
        if len(user_message.strip()) < 8:
            return False
        if len(assistant_response.strip()) < 10:
            return False
        return True

    def _store_ontology(self, db: Session, extraction, source_message: str = "") -> None:
        """Store extracted entities, relationships, and facts to the database."""
        from mnemosyne.models import Entity, Relationship, Fact as FactModel

        for ent in extraction.entities:
            existing = db.query(Entity).filter_by(name=ent.name).first()
            if existing:
                if ent.confidence > existing.confidence:
                    existing.type = ent.type
                    existing.confidence = ent.confidence
            else:
                db.add(Entity(name=ent.name, type=ent.type, confidence=ent.confidence))
            self._graph.add_entity(ent.name, ent.type, ent.confidence)

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

        for fact in extraction.facts:
            existing = db.query(FactModel).filter_by(
                subject=fact.subject, predicate=fact.predicate, object=fact.object,
            ).first()
            if not existing:
                db.add(FactModel(
                    subject=fact.subject,
                    predicate=fact.predicate,
                    object=fact.object,
                    source_message=source_message,
                ))

        db.commit()

    def search_memory(self, query: str, top_k: int = 10) -> list[dict]:
        """Search vector memory for relevant past messages."""
        db = self._session_factory()
        try:
            return self._embeddings.search(db, query, top_k=top_k)
        finally:
            db.close()

    def search_graph(self, entity_name: str) -> dict:
        """Search the ontology graph for an entity and its neighborhood."""
        return self._graph.get_neighbors(entity_name, hops=1)

    def get_neighbors(self, entity: str, hops: int = 1) -> dict:
        """Get neighbors of an entity in the knowledge graph."""
        return self._graph.get_neighbors(entity, hops=hops)

    def _auto_consolidate(self, db: Session) -> None:
        """Run consolidation and auto-apply all approved changes."""
        try:
            from mnemosyne.services.consolidation import ConsolidationService
            svc = ConsolidationService()
            report = svc.analyze()

            total = len(report.get("recommendations", []))
            if total == 0:
                logger.info("Auto-consolidation: no issues found")
                return

            actions = [{"index": i, "action": "approve"} for i in range(total)]
            result = svc.apply_recommendations(actions)

            logger.info(
                "Auto-consolidation: applied=%d, rejected=%d, errors=%d",
                result.get("applied", 0),
                result.get("rejected", 0),
                len(result.get("errors", [])),
            )

            self._graph.load_from_db(db)
        except Exception as e:
            logger.error("Auto-consolidation failed: %s", e)
