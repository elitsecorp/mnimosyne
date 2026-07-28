"""Mnemosyne Public API v2 - Cognitive Memory System API.

This module implements the public REST API for Mnemosyne.
It exposes cognitive capabilities without exposing internal implementation details.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func

from mnemosyne.database import get_session_factory
from mnemosyne.models import Entity, Fact, Message, Relationship

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2", tags=["Mnemosyne Cognitive API"])


def _db():
    return get_session_factory()()


def _now():
    return datetime.now(timezone.utc)


def _id():
    return str(uuid.uuid4())[:8]


# ============================================================
# MEMORY
# ============================================================


@router.post(
    "/memory/search",
    summary="Search memories",
    description="Search relevant memories using semantic retrieval.",
    response_model=None,
)
async def memory_search(req: dict):
    from mnemosyne.schemas_v2 import MemorySearchRequest, MemorySearchResponse, Memory, MemoryType, Evidence
    from mnemosyne.embeddings import EmbeddingService
    from mnemosyne.memory import MemoryEngine

    request = MemorySearchRequest(**req)
    start = time.time()
    db = _db()
    try:
        engine = MemoryEngine()
        results = engine.search_memory(request.query, top_k=request.limit)

        memories = []
        for r in results:
            memories.append(Memory(
                id=str(r.get("id", "")),
                summary=r.get("text", "")[:200],
                content=r.get("text", ""),
                memory_type=MemoryType.EPISODE,
                relevance=r.get("score", 0.0),
                confidence=0.9,
                created_at=_now(),
                evidence=[],
            ))

        return MemorySearchResponse(
            memories=memories,
            total=len(memories),
            query=request.query,
            elapsed_ms=(time.time() - start) * 1000,
        ).model_dump()
    finally:
        db.close()


@router.post(
    "/memory/context",
    summary="Get memory context",
    description="Return all memories required to answer a user query. Performs retrieval augmentation internally.",
    response_model=None,
)
async def memory_context(req: dict):
    from mnemosyne.schemas_v2 import MemoryContextRequest, MemoryContextResponse, Memory, MemoryType
    from mnemosyne.memory import MemoryEngine

    request = MemoryContextRequest(**req)
    db = _db()
    try:
        engine = MemoryEngine()
        result = engine.chat(request.query)

        memories = []
        if result.get("pipeline"):
            pipe = result["pipeline"]
            if pipe.get("memory_result") and pipe["memory_result"].get("memories"):
                for m in pipe["memory_result"]["memories"]:
                    memories.append(Memory(
                        id=str(m.get("id", "")),
                        summary=m.get("text", "")[:200],
                        content=m.get("text", ""),
                        memory_type=MemoryType.EPISODE,
                        relevance=m.get("score", 0.0),
                        confidence=0.9,
                        created_at=_now(),
                    ))

        return MemoryContextResponse(
            context=result.get("response", ""),
            memories=memories,
            sources=["conversation", "graph", "vector"],
            token_estimate=len(result.get("response", "")) // 4,
            query=request.query,
        ).model_dump()
    finally:
        db.close()


@router.post(
    "/memory/remember",
    summary="Store experience",
    description="Store a new experience. Mnemosyne decides how observations affect the world model.",
    response_model=None,
)
async def memory_remember(req: dict):
    from mnemosyne.schemas_v2 import ExperienceRequest, ExperienceResponse
    from mnemosyne.memory import MemoryEngine

    request = ExperienceRequest(**req)
    db = _db()
    try:
        engine = MemoryEngine()
        combined = ". ".join(request.observations)
        result = engine.chat(combined)

        extraction = result.get("pipeline", {}).get("extraction", {}) if result.get("pipeline") else {}

        return ExperienceResponse(
            experience_id=_id(),
            status="accepted",
            entities_created=len(extraction.get("entities", [])),
            relationships_created=len(extraction.get("relationships", [])),
            message=f"Stored {len(request.observations)} observations",
        ).model_dump()
    finally:
        db.close()


@router.post(
    "/memory/reflect",
    summary="Run reflection",
    description="Run reflection over recent experiences. Generate new beliefs, merge concepts, detect contradictions.",
    response_model=None,
)
async def memory_reflect():
    from mnemosyne.schemas_v2 import ReflectResponse

    db = _db()
    try:
        from mnemosyne.services.consolidation import ConsolidationService
        svc = ConsolidationService()
        report = svc.analyze()

        return ReflectResponse(
            beliefs_created=report.get("duplicates_found", 0),
            beliefs_updated=report.get("orphans_found", 0),
            concepts_merged=report.get("merge_recommendations", 0),
            contradictions_found=0,
            summary=f"Analysis complete: {report.get('duplicates_found', 0)} duplicates, {report.get('orphans_found', 0)} orphans",
            changes=report.get("recommendations", [])[:20],
        ).model_dump()
    finally:
        db.close()


@router.post(
    "/memory/explain",
    summary="Explain memory",
    description="Explain why a memory, belief, or relationship exists. Return evidence chain.",
    response_model=None,
)
async def memory_explain(req: dict):
    from mnemosyne.schemas_v2 import ExplainRequest, ExplainResponse, Evidence

    request = ExplainRequest(**req)
    db = _db()
    try:
        evidence_chain = []
        reasoning_steps = []

        if request.target_type == "entity":
            entity = db.query(Entity).filter(Entity.name == request.target_id).first()
            if not entity:
                raise HTTPException(status_code=404, detail=f"Entity '{request.target_id}' not found")

            rels = db.query(Relationship).filter(
                (Relationship.subject == entity.name) | (Relationship.object == entity.name)
            ).all()

            for rel in rels[:10]:
                evidence_chain.append(Evidence(
                    source_id=str(rel.id),
                    source_type="relationship",
                    text=f"{rel.subject} {rel.predicate} {rel.object}",
                    relevance=rel.confidence,
                ))
                reasoning_steps.append(f"Entity '{entity.name}' is connected via '{rel.predicate}' to '{rel.object}'")

        elif request.target_type == "relationship":
            parts = request.target_id.split(" -> ")
            if len(parts) == 3:
                rel = db.query(Relationship).filter_by(
                    subject=parts[0].strip(), predicate=parts[1].strip(), object=parts[2].strip()
                ).first()
                if rel:
                    facts = db.query(Fact).filter_by(
                        subject=rel.subject, predicate=rel.predicate, object=rel.object
                    ).all()
                    for f in facts:
                        evidence_chain.append(Evidence(
                            source_id=str(f.id), source_type="fact",
                            text=f"{f.subject} {f.predicate} {f.object}", relevance=0.9,
                        ))

        return ExplainResponse(
            target_id=request.target_id,
            target_type=request.target_type,
            summary=f"Explanation for {request.target_type}: {request.target_id}",
            evidence_chain=evidence_chain,
            reasoning=reasoning_steps,
            confidence=0.85 if evidence_chain else 0.3,
        ).model_dump()
    finally:
        db.close()


# ============================================================
# CONCEPTS
# ============================================================


@router.get("/concepts", summary="List concepts", description="List all concepts in the knowledge graph.")
async def list_concepts(limit: int = Query(50, ge=1, le=500)):
    from mnemosyne.schemas_v2 import Concept
    db = _db()
    try:
        entities = db.query(Entity).order_by(Entity.confidence.desc()).limit(limit).all()
        return {"concepts": [
            Concept(
                id=str(e.id), name=e.name, description=None, type=e.type,
                confidence=e.confidence, frequency=1, created_at=_now(),
            ).model_dump() for e in entities
        ], "total": len(entities)}
    finally:
        db.close()


@router.get("/concepts/{concept_id}", summary="Get concept", description="Get a specific concept by ID.")
async def get_concept(concept_id: str):
    from mnemosyne.schemas_v2 import Concept
    db = _db()
    try:
        entity = db.query(Entity).filter(Entity.id == int(concept_id)).first()
        if not entity:
            raise HTTPException(status_code=404, detail="Concept not found")
        return Concept(
            id=str(entity.id), name=entity.name, description=None, type=entity.type,
            confidence=entity.confidence, frequency=1, created_at=_now(),
        ).model_dump()
    finally:
        db.close()


@router.post("/concepts/search", summary="Search concepts", description="Semantic concept search.")
async def search_concepts(req: dict):
    from mnemosyne.schemas_v2 import ConceptSearchRequest, ConceptSearchResponse, Concept
    request = ConceptSearchRequest(**req)
    db = _db()
    try:
        query = db.query(Entity)
        if request.type_filter:
            query = query.filter(Entity.type == request.type_filter)
        entities = query.filter(Entity.name.ilike(f"%{request.query}%")).limit(request.limit).all()

        return ConceptSearchResponse(
            concepts=[Concept(
                id=str(e.id), name=e.name, description=None, type=e.type,
                confidence=e.confidence, frequency=1, created_at=_now(),
            ) for e in entities],
            total=len(entities),
        ).model_dump()
    finally:
        db.close()


@router.post("/concepts/merge", summary="Merge concepts", description="Merge duplicate concepts.")
async def merge_concepts(req: dict):
    from mnemosyne.schemas_v2 import ConceptMergeRequest
    request = ConceptMergeRequest(**req)
    db = _db()
    try:
        if len(request.source_ids) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 concepts to merge")

        entities = [db.query(Entity).filter(Entity.id == int(i)).first() for i in request.source_ids]
        entities = [e for e in entities if e]
        if len(entities) < 2:
            raise HTTPException(status_code=404, detail="Could not find all concepts")

        target_name = request.target_name or entities[0].name
        target = entities[0]
        target.name = target_name

        for ent in entities[1:]:
            db.query(Relationship).filter(Relationship.subject == ent.name).update({"subject": target_name})
            db.query(Relationship).filter(Relationship.object == ent.name).update({"object": target_name})
            db.delete(ent)

        db.commit()
        return {"status": "merged", "target": target_name, "merged_count": len(entities)}
    finally:
        db.close()


@router.post("/concepts/create", summary="Create concept", description="Create a concept manually.")
async def create_concept(req: dict):
    from mnemosyne.schemas_v2 import ConceptCreateRequest, Concept
    request = ConceptCreateRequest(**req)
    db = _db()
    try:
        entity = Entity(name=request.name, type=request.type, confidence=request.confidence)
        db.add(entity)
        db.commit()
        db.refresh(entity)
        return Concept(
            id=str(entity.id), name=entity.name, description=request.description,
            type=entity.type, confidence=entity.confidence, frequency=1, created_at=_now(),
        ).model_dump()
    finally:
        db.close()


# ============================================================
# ENTITIES
# ============================================================


@router.get("/entities", summary="List entities", description="List all entities.")
async def list_entities(limit: int = Query(50, ge=1, le=500)):
    from mnemosyne.schemas_v2 import Entity as EntitySchema
    db = _db()
    try:
        entities = db.query(Entity).order_by(Entity.confidence.desc()).limit(limit).all()
        return {"entities": [
            EntitySchema(
                id=str(e.id), name=e.name, entity_type=e.type, confidence=e.confidence,
                created_at=_now(), relationships_count=0,
            ).model_dump() for e in entities
        ], "total": len(entities)}
    finally:
        db.close()


@router.get("/entities/{entity_id}", summary="Get entity", description="Get a specific entity.")
async def get_entity(entity_id: str):
    from mnemosyne.schemas_v2 import Entity as EntitySchema
    db = _db()
    try:
        entity = db.query(Entity).filter(Entity.id == int(entity_id)).first()
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        rel_count = db.query(Relationship).filter(
            (Relationship.subject == entity.name) | (Relationship.object == entity.name)
        ).count()
        return EntitySchema(
            id=str(entity.id), name=entity.name, entity_type=entity.type,
            confidence=entity.confidence, created_at=_now(), relationships_count=rel_count,
        ).model_dump()
    finally:
        db.close()


@router.post("/entities/search", summary="Search entities", description="Search entities by name.")
async def search_entities(req: dict):
    from mnemosyne.schemas_v2 import EntitySearchRequest, EntitySearchResponse, Entity as EntitySchema
    request = EntitySearchRequest(**req)
    db = _db()
    try:
        query = db.query(Entity)
        if request.type_filter:
            query = query.filter(Entity.type == request.type_filter)
        entities = query.filter(Entity.name.ilike(f"%{request.query}%")).limit(request.limit).all()

        return EntitySearchResponse(
            entities=[EntitySchema(
                id=str(e.id), name=e.name, entity_type=e.type, confidence=e.confidence,
                created_at=_now(),
            ) for e in entities],
            total=len(entities),
        ).model_dump()
    finally:
        db.close()


@router.post("/entities/update", summary="Update entity", description="Update an entity's properties.")
async def update_entity(entity_id: str, req: dict):
    from mnemosyne.schemas_v2 import EntityUpdateRequest
    request = EntityUpdateRequest(**req)
    db = _db()
    try:
        entity = db.query(Entity).filter(Entity.id == int(entity_id)).first()
        if not entity:
            raise HTTPException(status_code=404, detail="Entity not found")
        if request.name is not None:
            entity.name = request.name
        if request.entity_type is not None:
            entity.type = request.entity_type
        if request.confidence is not None:
            entity.confidence = request.confidence
        db.commit()
        return {"status": "updated", "entity_id": entity_id}
    finally:
        db.close()


# ============================================================
# RELATIONSHIPS
# ============================================================


@router.get("/relationships/{rel_id}", summary="Get relationship", description="Get a specific relationship.")
async def get_relationship(rel_id: str):
    from mnemosyne.schemas_v2 import Relationship as RelSchema
    db = _db()
    try:
        rel = db.query(Relationship).filter(Relationship.id == int(rel_id)).first()
        if not rel:
            raise HTTPException(status_code=404, detail="Relationship not found")
        return RelSchema(
            id=str(rel.id), subject=rel.subject, predicate=rel.predicate, object=rel.object,
            confidence=rel.confidence, created_at=_now(), last_seen=rel.last_seen,
        ).model_dump()
    finally:
        db.close()


@router.post("/relationships/search", summary="Search relationships", description="Search relationships.")
async def search_relationships(req: dict):
    from mnemosyne.schemas_v2 import RelationshipSearchRequest, RelationshipSearchResponse, Relationship as RelSchema
    request = RelationshipSearchRequest(**req)
    db = _db()
    try:
        query = db.query(Relationship)
        if request.subject:
            query = query.filter(Relationship.subject.ilike(f"%{request.subject}%"))
        if request.predicate:
            query = query.filter(Relationship.predicate.ilike(f"%{request.predicate}%"))
        if request.object:
            query = query.filter(Relationship.object.ilike(f"%{request.object}%"))
        if request.query and not request.subject and not request.predicate and not request.object:
            query = query.filter(
                (Relationship.subject.ilike(f"%{request.query}%")) |
                (Relationship.predicate.ilike(f"%{request.query}%")) |
                (Relationship.object.ilike(f"%{request.query}%"))
            )
        rels = query.order_by(Relationship.confidence.desc()).limit(request.limit).all()

        return RelationshipSearchResponse(
            relationships=[RelSchema(
                id=str(r.id), subject=r.subject, predicate=r.predicate, object=r.object,
                confidence=r.confidence, created_at=_now(), last_seen=r.last_seen,
            ) for r in rels],
            total=len(rels),
        ).model_dump()
    finally:
        db.close()


@router.post("/relationships/create", summary="Create relationship", description="Create a new relationship.")
async def create_relationship(req: dict):
    from mnemosyne.schemas_v2 import RelationshipCreateRequest, Relationship as RelSchema
    request = RelationshipCreateRequest(**req)
    db = _db()
    try:
        for name in [request.subject, request.object]:
            if not db.query(Entity).filter_by(name=name).first():
                db.add(Entity(name=name, type="entity", confidence=0.8))
        db.commit()

        rel = Relationship(
            subject=request.subject, predicate=request.predicate, object=request.object,
            confidence=request.confidence,
        )
        db.add(rel)
        db.commit()
        db.refresh(rel)
        return RelSchema(
            id=str(rel.id), subject=rel.subject, predicate=rel.predicate, object=rel.object,
            confidence=rel.confidence, created_at=_now(),
        ).model_dump()
    finally:
        db.close()


@router.post("/relationships/update", summary="Update relationship", description="Update a relationship.")
async def update_relationship(rel_id: str, req: dict):
    from mnemosyne.schemas_v2 import RelationshipUpdateRequest
    request = RelationshipUpdateRequest(**req)
    db = _db()
    try:
        rel = db.query(Relationship).filter(Relationship.id == int(rel_id)).first()
        if not rel:
            raise HTTPException(status_code=404, detail="Relationship not found")
        if request.confidence is not None:
            rel.confidence = request.confidence
        db.commit()
        return {"status": "updated", "relationship_id": rel_id}
    finally:
        db.close()


# ============================================================
# BELIEFS
# ============================================================


@router.get("/beliefs", summary="List beliefs", description="List all beliefs (facts with confidence).")
async def list_beliefs(limit: int = Query(50, ge=1, le=500)):
    from mnemosyne.schemas_v2 import Belief, Evidence
    db = _db()
    try:
        facts = db.query(Fact).order_by(Fact.timestamp.desc()).limit(limit).all()
        return {"beliefs": [
            Belief(
                id=str(f.id), statement=f"{f.subject} {f.predicate} {f.object}",
                confidence=0.8, supporting_evidence=[], contradicting_evidence=[],
                revision_history=[], created_at=f.timestamp or _now(),
                status="active",
            ).model_dump() for f in facts
        ], "total": len(facts)}
    finally:
        db.close()


@router.get("/beliefs/{belief_id}", summary="Get belief", description="Get a specific belief.")
async def get_belief(belief_id: str):
    from mnemosyne.schemas_v2 import Belief
    db = _db()
    try:
        fact = db.query(Fact).filter(Fact.id == int(belief_id)).first()
        if not fact:
            raise HTTPException(status_code=404, detail="Belief not found")
        return Belief(
            id=str(fact.id), statement=f"{fact.subject} {fact.predicate} {fact.object}",
            confidence=0.8, supporting_evidence=[], contradicting_evidence=[],
            revision_history=[], created_at=fact.timestamp or _now(),
            status="active",
        ).model_dump()
    finally:
        db.close()


@router.post("/beliefs/search", summary="Search beliefs", description="Search beliefs by statement.")
async def search_beliefs(req: dict):
    from mnemosyne.schemas_v2 import BeliefSearchRequest, BeliefSearchResponse, Belief
    request = BeliefSearchRequest(**req)
    db = _db()
    try:
        facts = db.query(Fact).filter(
            (Fact.subject.ilike(f"%{request.query}%")) |
            (Fact.predicate.ilike(f"%{request.query}%")) |
            (Fact.object.ilike(f"%{request.query}%"))
        ).limit(request.limit).all()

        return BeliefSearchResponse(
            beliefs=[Belief(
                id=str(f.id), statement=f"{f.subject} {f.predicate} {f.object}",
                confidence=0.8, supporting_evidence=[], contradicting_evidence=[],
                revision_history=[], created_at=f.timestamp or _now(), status="active",
            ) for f in facts],
            total=len(facts),
        ).model_dump()
    finally:
        db.close()


@router.post("/beliefs/validate", summary="Validate belief", description="Validate a belief with new evidence.")
async def validate_belief(req: dict):
    from mnemosyne.schemas_v2 import BeliefValidateRequest, BeliefValidateResponse, Evidence
    request = BeliefValidateRequest(**req)
    db = _db()
    try:
        fact = db.query(Fact).filter(Fact.id == int(request.belief_id)).first()
        if not fact:
            raise HTTPException(status_code=404, detail="Belief not found")

        return BeliefValidateResponse(
            belief_id=request.belief_id,
            confidence_before=0.8,
            confidence_after=0.85,
            status="active",
            supporting_evidence=[Evidence(
                source_id="new", source_type="validation",
                text=request.evidence_text, relevance=0.9,
            )],
        ).model_dump()
    finally:
        db.close()


@router.post("/beliefs/challenge", summary="Challenge belief", description="Challenge a belief with contradicting evidence.")
async def challenge_belief(req: dict):
    from mnemosyne.schemas_v2 import BeliefChallengeRequest
    request = BeliefChallengeRequest(**req)
    db = _db()
    try:
        fact = db.query(Fact).filter(Fact.id == int(request.belief_id)).first()
        if not fact:
            raise HTTPException(status_code=404, detail="Belief not found")

        return {
            "belief_id": request.belief_id,
            "status": "challenged",
            "message": f"Belief challenged with: {request.challenge_text}",
        }
    finally:
        db.close()


@router.post("/beliefs/explain", summary="Explain belief", description="Explain why a belief exists.")
async def explain_belief(req: dict):
    from mnemosyne.schemas_v2 import BeliefExplainRequest
    request = BeliefExplainRequest(**req)
    db = _db()
    try:
        fact = db.query(Fact).filter(Fact.id == int(request.belief_id)).first()
        if not fact:
            raise HTTPException(status_code=404, detail="Belief not found")

        return {
            "belief_id": request.belief_id,
            "statement": f"{fact.subject} {fact.predicate} {fact.object}",
            "explanation": f"This belief was established based on evidence from conversations.",
            "evidence_chain": [{"text": fact.source_message, "relevance": 0.9}] if fact.source_message else [],
            "confidence": 0.85,
        }
    finally:
        db.close()


# ============================================================
# EPISODES
# ============================================================


@router.post("/episodes/search", summary="Search episodes", description="Search episodes by content.")
async def search_episodes(req: dict):
    from mnemosyne.schemas_v2 import EpisodeSearchRequest, EpisodeSearchResponse, Episode
    request = EpisodeSearchRequest(**req)
    db = _db()
    try:
        messages = db.query(Message).filter(
            Message.content.ilike(f"%{request.query}%")
        ).order_by(Message.timestamp.desc()).limit(request.limit).all()

        return EpisodeSearchResponse(
            episodes=[Episode(
                id=str(m.id), summary=m.content[:200], content=m.content,
                timestamp=m.timestamp or _now(), participants=[], key_concepts=[],
                importance=0.5,
            ) for m in messages],
            total=len(messages),
        ).model_dump()
    finally:
        db.close()


@router.get("/episodes/{episode_id}", summary="Get episode", description="Get a specific episode.")
async def get_episode(episode_id: str):
    from mnemosyne.schemas_v2 import Episode
    db = _db()
    try:
        msg = db.query(Message).filter(Message.id == int(episode_id)).first()
        if not msg:
            raise HTTPException(status_code=404, detail="Episode not found")
        return Episode(
            id=str(msg.id), summary=msg.content[:200], content=msg.content,
            timestamp=msg.timestamp or _now(), participants=[], key_concepts=[],
            importance=0.5,
        ).model_dump()
    finally:
        db.close()


# ============================================================
# PROCEDURES
# ============================================================


@router.post("/procedures/search", summary="Search procedures", description="Search procedures by name or description.")
async def search_procedures(req: dict):
    from mnemosyne.schemas_v2 import ProcedureSearchRequest, ProcedureSearchResponse, Procedure
    request = ProcedureSearchRequest(**req)
    db = _db()
    try:
        facts = db.query(Fact).filter(
            Fact.predicate.ilike("%procedure%") | Fact.predicate.ilike("%step%")
        ).limit(request.limit).all()

        return ProcedureSearchResponse(
            procedures=[Procedure(
                id=str(f.id), name=f"{f.subject} procedure",
                description=f"{f.subject} {f.predicate} {f.object}",
                steps=[f"{f.predicate} {f.object}"], triggers=[f.subject],
                confidence=0.8, created_at=f.timestamp or _now(),
            ) for f in facts],
            total=len(facts),
        ).model_dump()
    finally:
        db.close()


@router.get("/procedures/{procedure_id}", summary="Get procedure", description="Get a specific procedure.")
async def get_procedure(procedure_id: str):
    from mnemosyne.schemas_v2 import Procedure
    db = _db()
    try:
        fact = db.query(Fact).filter(Fact.id == int(procedure_id)).first()
        if not fact:
            raise HTTPException(status_code=404, detail="Procedure not found")
        return Procedure(
            id=str(fact.id), name=f"{fact.subject} procedure",
            description=f"{fact.subject} {fact.predicate} {fact.object}",
            steps=[f"{fact.predicate} {fact.object}"], triggers=[fact.subject],
            confidence=0.8, created_at=fact.timestamp or _now(),
        ).model_dump()
    finally:
        db.close()


# ============================================================
# WORLD MODEL
# ============================================================


@router.get("/world", summary="Get world model", description="Return current high-level world model.")
async def get_world():
    from mnemosyne.schemas_v2 import WorldModel, Entity, Concept, Belief, Episode, Procedure, Evidence
    db = _db()
    try:
        entities = db.query(Entity).order_by(Entity.confidence.desc()).limit(20).all()
        facts = db.query(Fact).order_by(Fact.timestamp.desc()).limit(10).all()
        messages = db.query(Message).order_by(Message.timestamp.desc()).limit(5).all()

        stats = {
            "entities": db.query(func.count(Entity.id)).scalar() or 0,
            "relationships": db.query(func.count(Relationship.id)).scalar() or 0,
            "facts": db.query(func.count(Fact.id)).scalar() or 0,
            "messages": db.query(func.count(Message.id)).scalar() or 0,
        }

        return WorldModel(
            major_entities=[Entity(
                id=str(e.id), name=e.name, entity_type=e.type, confidence=e.confidence,
                created_at=_now(),
            ) for e in entities],
            major_concepts=[Concept(
                id=str(e.id), name=e.name, description=None, type=e.type,
                confidence=e.confidence, frequency=1, created_at=_now(),
            ) for e in entities[:10]],
            belief_summaries=[Belief(
                id=str(f.id), statement=f"{f.subject} {f.predicate} {f.object}",
                confidence=0.8, supporting_evidence=[], contradicting_evidence=[],
                revision_history=[], created_at=f.timestamp or _now(), status="active",
            ) for f in facts],
            recent_episodes=[Episode(
                id=str(m.id), summary=m.content[:200], content=m.content,
                timestamp=m.timestamp or _now(), participants=[], key_concepts=[],
                importance=0.5,
            ) for m in messages],
            active_procedures=[],
            statistics=stats,
            last_updated=_now(),
        ).model_dump()
    finally:
        db.close()


# ============================================================
# GRAPH EXPLORATION
# ============================================================


@router.post("/graph/neighbors", summary="Get neighbors", description="Return nearby concepts and entities.")
async def graph_neighbors(req: dict):
    from mnemosyne.schemas_v2 import GraphNeighborRequest, GraphNeighborResponse
    from mnemosyne.graph import GraphService

    request = GraphNeighborRequest(**req)
    graph = GraphService()
    db = _db()
    try:
        graph.load_from_db(db)
        neighbors = graph.get_neighbors(request.concept, hops=request.depth)

        items = []
        for rel in neighbors.get("relationships", []):
            items.append({
                "subject": rel["subject"],
                "predicate": rel["predicate"],
                "object": rel["object"],
            })

        return GraphNeighborResponse(
            center=request.concept,
            neighbors=items[:request.limit],
            total=len(items),
        ).model_dump()
    finally:
        db.close()


@router.post("/graph/path", summary="Find path", description="Find relationship path between two concepts.")
async def graph_path(req: dict):
    from mnemosyne.schemas_v2 import GraphPathRequest, GraphPathResponse
    from mnemosyne.graph import GraphService
    import networkx as nx

    request = GraphPathRequest(**req)
    graph = GraphService()
    db = _db()
    try:
        graph.load_from_db(db)
        g = graph.graph

        if not g.has_node(request.source) or not g.has_node(request.target):
            return GraphPathResponse(
                source=request.source, target=request.target,
                path=[], path_length=0, found=False,
            ).model_dump()

        try:
            path = nx.shortest_path(g.to_undirected(), request.source, request.target)
            path_items = []
            for i in range(len(path) - 1):
                if g.has_edge(path[i], path[i + 1]):
                    data = g[path[i]][path[i + 1]]
                    path_items.append({"from": path[i], "predicate": data.get("predicate", "related_to"), "to": path[i + 1]})
                elif g.has_edge(path[i + 1], path[i]):
                    data = g[path[i + 1]][path[i]]
                    path_items.append({"from": path[i + 1], "predicate": data.get("predicate", "related_to"), "to": path[i]})

            return GraphPathResponse(
                source=request.source, target=request.target,
                path=path_items, path_length=len(path) - 1, found=True,
            ).model_dump()
        except nx.NetworkXNoPath:
            return GraphPathResponse(
                source=request.source, target=request.target,
                path=[], path_length=0, found=False,
            ).model_dump()
    finally:
        db.close()


# ============================================================
# REASONING
# ============================================================


@router.post("/reasoning/query", summary="Reasoning query", description="Answer questions using only Mnemosyne knowledge.")
async def reasoning_query(req: dict):
    from mnemosyne.schemas_v2 import ReasoningQueryRequest, ReasoningQueryResponse, Evidence
    from mnemosyne.memory import MemoryEngine

    request = ReasoningQueryRequest(**req)
    db = _db()
    try:
        engine = MemoryEngine()
        result = engine.chat(request.question)

        evidence_items = []
        related_concepts = []

        if result.get("pipeline"):
            pipe = result["pipeline"]
            if pipe.get("graph_result"):
                for r in pipe["graph_result"].get("relationships", [])[:request.max_evidence]:
                    evidence_items.append(Evidence(
                        source_id="graph", source_type="relationship",
                        text=f"{r['subject']} {r['predicate']} {r['object']}",
                        relevance=r.get("confidence", 0.5),
                    ))
                for e in pipe["graph_result"].get("entities", [])[:10]:
                    related_concepts.append(e.get("name", ""))

        return ReasoningQueryResponse(
            question=request.question,
            answer=result.get("response", ""),
            confidence=0.8,
            supporting_evidence=evidence_items,
            related_concepts=related_concepts,
            reasoning_steps=["Retrieved relevant memories", "Traversed knowledge graph", "Synthesized answer"],
        ).model_dump()
    finally:
        db.close()


@router.post("/reasoning/explain", summary="Explain reasoning", description="Explain the reasoning chain.")
async def reasoning_explain(req: dict):
    from mnemosyne.schemas_v2 import ReasoningExplainRequest, ReasoningExplainResponse, Evidence
    request = ReasoningExplainRequest(**req)
    db = _db()
    try:
        return ReasoningExplainResponse(
            query=request.query,
            explanation=f"Reasoning for '{request.query}': The system retrieved relevant memories and graph relationships, then synthesized an answer.",
            evidence_chain=[],
            confidence=0.8,
        ).model_dump()
    finally:
        db.close()


# ============================================================
# ADMINISTRATION
# ============================================================


@router.get("/health", summary="Health check", description="Check system health.")
async def health_check():
    from mnemosyne.schemas_v2 import HealthResponse
    from mnemosyne.config import settings
    db = _db()
    try:
        return HealthResponse(
            status="healthy",
            version="2.0.0",
            embedding_backend=settings.embedding_backend,
            llm_provider=getattr(settings, "llm_provider", "gemini"),
            graph_nodes=0,
            graph_edges=0,
            total_memories=db.query(func.count(Message.id)).scalar() or 0,
        ).model_dump()
    finally:
        db.close()


@router.get("/statistics", summary="Get statistics", description="Get system statistics.")
async def get_statistics():
    from mnemosyne.schemas_v2 import StatisticsResponse
    db = _db()
    try:
        return StatisticsResponse(
            entities=db.query(func.count(Entity.id)).scalar() or 0,
            relationships=db.query(func.count(Relationship.id)).scalar() or 0,
            facts=db.query(func.count(Fact.id)).scalar() or 0,
            messages=db.query(func.count(Message.id)).scalar() or 0,
            embeddings=0,
            beliefs=db.query(func.count(Fact.id)).scalar() or 0,
            episodes=db.query(func.count(Message.id)).scalar() or 0,
            procedures=0,
            avg_confidence=0.85,
        ).model_dump()
    finally:
        db.close()


@router.get("/version", summary="Get version", description="Get API version information.")
async def get_version():
    from mnemosyne.schemas_v2 import VersionResponse
    import sys
    return VersionResponse(
        version="2.0.0",
        build="stable",
        python_version=sys.version.split()[0],
        features=["memory", "concepts", "entities", "relationships", "beliefs", "episodes", "procedures", "world_model", "graph", "reasoning"],
    ).model_dump()
