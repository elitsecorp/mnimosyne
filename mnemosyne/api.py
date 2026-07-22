"""FastAPI application and route definitions."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, Response

from mnemosyne.memory import MemoryEngine
from mnemosyne.schemas import (
    ApplyRequest,
    ApplyResponse,
    ChatRequest,
    ChatResponse,
    ConsolidationReport,
    EntityOut,
    FactOut,
    GraphEdgeDetail,
    GraphEntity,
    GraphNodeDetail,
    GraphRelationship,
    GraphResponse,
    GraphSeedResponse,
    GraphStatsResponse,
    GraphTypesResponse,
    HealthResponse,
    ImportJobStatus,
    ImportRequest,
    ImportResponse,
    MessageOut,
    RelationshipOut,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SessionCreate,
    SessionOut,
    StatsResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mnemosyne",
    description="Persistent memory system for Large Language Models",
    version="0.2.0",
)

_engine: MemoryEngine | None = None
_import_service = None


def get_engine() -> MemoryEngine:
    """Return the singleton MemoryEngine, creating it on first call."""
    global _engine
    if _engine is None:
        _engine = MemoryEngine()
    return _engine


def get_import_service():
    """Return the singleton ImportService, creating it on first call."""
    global _import_service
    if _import_service is None:
        from mnemosyne.services.importer import ImportService
        _import_service = ImportService(engine=get_engine())
    return _import_service


@app.on_event("startup")
def _startup() -> None:
    """Initialize database, graph, and embedding model on startup."""
    from mnemosyne.database import init_db, get_session_factory
    init_db()
    engine = get_engine()
    # Load the full graph from DB into NetworkX
    db = get_session_factory()()
    try:
        engine._graph.load_from_db(db)
    finally:
        db.close()
    # Eagerly load the embedding model
    engine.warmup()
    logger.info("Mnemosyne started successfully.")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the chat UI."""
    html_path = Path(__file__).parent / "static.html"
    resp = HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.get("/memories", response_class=HTMLResponse)
def memories_page() -> HTMLResponse:
    """Serve the memory explorer UI."""
    html_path = Path(__file__).parent / "memories.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse()


@app.get("/api/status")
def system_status():
    """Check if the system is ready (embedding model loaded, etc.)."""
    engine = get_engine()
    return {
        "ready": engine.is_ready,
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.local_embedding_model if settings.embedding_backend == "local" else settings.gemini_embedding_model,
    }


@app.post("/api/database/reset")
def reset_database():
    """Delete all data from the database."""
    from mnemosyne.database import get_session_factory
    from sqlalchemy import text

    try:
        db = get_session_factory()()
        try:
            for table in ["embeddings", "facts", "relationships", "entities", "messages", "chat_sessions"]:
                db.execute(text(f"DELETE FROM {table}"))
            db.commit()
        finally:
            db.close()

        global _engine, _import_service
        if _engine:
            db2 = get_session_factory()()
            try:
                _engine._graph.load_from_db(db2)
            finally:
                db2.close()

        return {"status": "reset", "message": "All data deleted."}
    except Exception as e:
        logger.error("Reset failed: %s", e)
        return {"status": "error", "message": str(e)}


@app.post("/api/sessions", response_model=SessionOut)
def create_session(req: SessionCreate):
    """Create a new chat session."""
    from mnemosyne.database import get_session_factory
    from mnemosyne.models import ChatSession
    db = get_session_factory()()
    try:
        session = ChatSession(title=req.title)
        db.add(session)
        db.commit()
        db.refresh(session)
        return SessionOut(
            id=session.id,
            title=session.title,
            created_at=str(session.created_at),
            updated_at=str(session.updated_at),
            message_count=0,
        )
    finally:
        db.close()


@app.get("/api/sessions", response_model=list[SessionOut])
def list_sessions():
    """List all chat sessions."""
    from mnemosyne.database import get_session_factory
    from mnemosyne.models import ChatSession, Message
    from sqlalchemy import func
    db = get_session_factory()()
    try:
        sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
        result = []
        for s in sessions:
            count = db.query(func.count(Message.id)).filter(Message.session_id == s.id).scalar() or 0
            result.append(SessionOut(
                id=s.id,
                title=s.title,
                created_at=str(s.created_at),
                updated_at=str(s.updated_at),
                message_count=count,
            ))
        return result
    finally:
        db.close()


@app.get("/api/sessions/{session_id}", response_model=SessionOut)
def get_session(session_id: int):
    """Get a single chat session."""
    from mnemosyne.database import get_session_factory
    from mnemosyne.models import ChatSession, Message
    from sqlalchemy import func
    db = get_session_factory()()
    try:
        session = db.get(ChatSession, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        count = db.query(func.count(Message.id)).filter(Message.session_id == session_id).scalar() or 0
        return SessionOut(
            id=session.id,
            title=session.title,
            created_at=str(session.created_at),
            updated_at=str(session.updated_at),
            message_count=count,
        )
    finally:
        db.close()


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: int):
    """Delete a chat session, its messages, and their embeddings."""
    from mnemosyne.database import get_session_factory
    from mnemosyne.models import ChatSession, Message, Embedding
    db = get_session_factory()()
    try:
        session = db.get(ChatSession, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        msg_ids = [m.id for m in db.query(Message.id).filter(Message.session_id == session_id).all()]
        if msg_ids:
            db.query(Embedding).filter(Embedding.message_id.in_(msg_ids)).delete(synchronize_session="fetch")
        db.query(Message).filter(Message.session_id == session_id).delete(synchronize_session="fetch")
        db.delete(session)
        db.commit()
        return {"status": "deleted"}
    finally:
        db.close()


@app.put("/api/sessions/{session_id}")
def update_session_title(session_id: int, req: SessionCreate):
    """Update session title."""
    from mnemosyne.database import get_session_factory
    from mnemosyne.models import ChatSession
    db = get_session_factory()()
    try:
        session = db.get(ChatSession, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        session.title = req.title
        db.commit()
        return {"status": "updated"}
    finally:
        db.close()


@app.get("/api/sessions/{session_id}/messages", response_model=list[MessageOut])
def get_session_messages(session_id: int):
    """Get all messages in a session."""
    from mnemosyne.database import get_session_factory
    from mnemosyne.models import Message
    db = get_session_factory()()
    try:
        messages = (
            db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.timestamp.asc())
            .all()
        )
        return [
            MessageOut(id=m.id, role=m.role, content=m.content, timestamp=str(m.timestamp))
            for m in messages
        ]
    finally:
        db.close()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Process a user message through the memory pipeline."""
    try:
        engine = get_engine()
        result = engine.chat(req.message, session_id=req.session_id)
        return ChatResponse(
            response=result["response"],
            pipeline=result.get("pipeline"),
            session_id=result.get("session_id"),
        )
    except Exception as e:
        logger.error("Chat error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
def search_memory(req: SearchRequest) -> SearchResponse:
    """Search vector memory for relevant past messages."""
    try:
        engine = get_engine()
        results = engine.search_memory(req.query, top_k=req.top_k)
        return SearchResponse(
            results=[
                SearchResult(
                    id=r["id"],
                    message_id=r["message_id"],
                    text=r["text"],
                    score=r["score"],
                )
                for r in results
            ]
        )
    except Exception as e:
        logger.error("Search error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/{entity_name}", response_model=GraphResponse)
def search_graph(entity_name: str) -> GraphResponse:
    """Search the ontology graph for an entity and its neighborhood."""
    try:
        engine = get_engine()
        result = engine.search_graph(entity_name)
        return GraphResponse(
            entity=result["entity"],
            relationships=[
                GraphRelationship(**r) for r in result["relationships"]
            ],
            connected_entities=[
                GraphEntity(**e) for e in result["connected_entities"]
            ],
        )
    except Exception as e:
        logger.error("Graph error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/{entity_name}/neighbors", response_model=GraphResponse)
def get_neighbors(entity_name: str, hops: int = 1) -> GraphResponse:
    """Get neighbors of an entity with configurable hop distance."""
    try:
        engine = get_engine()
        result = engine.get_neighbors(entity_name, hops=hops)
        return GraphResponse(
            entity=result["entity"],
            relationships=[
                GraphRelationship(**r) for r in result["relationships"]
            ],
            connected_entities=[
                GraphEntity(**e) for e in result["connected_entities"]
            ],
        )
    except Exception as e:
        logger.error("Graph neighbors error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/entities")
def list_entities() -> list[EntityOut]:
    """List all stored entities."""
    from mnemosyne.database import get_session_factory
    from mnemosyne.models import Entity
    db = get_session_factory()()
    try:
        rows = db.query(Entity).order_by(Entity.confidence.desc()).all()
        return [EntityOut(id=r.id, name=r.name, type=r.type, confidence=r.confidence) for r in rows]
    finally:
        db.close()


@app.get("/api/relationships")
def list_relationships() -> list[RelationshipOut]:
    """List all stored relationships."""
    from mnemosyne.database import get_session_factory
    from mnemosyne.models import Relationship
    db = get_session_factory()()
    try:
        rows = db.query(Relationship).order_by(Relationship.confidence.desc()).all()
        return [
            RelationshipOut(id=r.id, subject=r.subject, predicate=r.predicate, object=r.object, confidence=r.confidence)
            for r in rows
        ]
    finally:
        db.close()


@app.get("/api/facts")
def list_facts() -> list[FactOut]:
    """List all stored facts with source messages."""
    from mnemosyne.database import get_session_factory
    from mnemosyne.models import Fact
    db = get_session_factory()()
    try:
        rows = db.query(Fact).order_by(Fact.timestamp.desc()).all()
        return [
            FactOut(
                id=r.id, subject=r.subject, predicate=r.predicate, object=r.object,
                source_message=r.source_message or "", timestamp=str(r.timestamp),
            )
            for r in rows
        ]
    finally:
        db.close()


@app.get("/api/messages")
def list_messages() -> list[MessageOut]:
    """List all conversation messages."""
    from mnemosyne.database import get_session_factory
    from mnemosyne.models import Message
    db = get_session_factory()()
    try:
        rows = db.query(Message).order_by(Message.timestamp.asc()).all()
        return [
            MessageOut(id=r.id, role=r.role, content=r.content, timestamp=str(r.timestamp))
            for r in rows
        ]
    finally:
        db.close()


@app.get("/api/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    """Get summary counts of all stored data."""
    from mnemosyne.database import get_session_factory
    from mnemosyne.models import Entity, Relationship, Fact, Message, Embedding
    db = get_session_factory()()
    try:
        return StatsResponse(
            messages=db.query(Message).count(),
            embeddings=db.query(Embedding).count(),
            entities=db.query(Entity).count(),
            relationships=db.query(Relationship).count(),
            facts=db.query(Fact).count(),
        )
    finally:
        db.close()


# --- Page routes ---


@app.get("/graph-explorer", response_class=HTMLResponse)
def graph_explorer_page() -> HTMLResponse:
    """Serve the knowledge graph explorer UI."""
    html_path = Path(__file__).parent / "graph_explorer.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/importer", response_class=HTMLResponse)
def importer_page() -> HTMLResponse:
    """Serve the data importer UI."""
    html_path = Path(__file__).parent / "importer.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/consolidate", response_class=HTMLResponse)
def consolidate_page() -> HTMLResponse:
    """Serve the memory consolidator UI."""
    html_path = Path(__file__).parent / "consolidator.html"
    resp = HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.get("/static/cytoscape.min.js")
def serve_cytoscape():
    """Serve the Cytoscape.js library."""
    from fastapi.responses import FileResponse
    js_path = Path(__file__).parent / "static" / "cytoscape.min.js"
    return FileResponse(js_path, media_type="application/javascript")


# --- Me Graph API ---


@app.get("/api/me/graph")
def get_me_graph():
    """Get the Me subgraph for visualization."""
    from mnemosyne.services.owner_compiler import OwnerCompiler
    from mnemosyne.database import get_session_factory
    db = get_session_factory()()
    try:
        compiler = OwnerCompiler()
        return compiler.get_me_graph(db)
    finally:
        db.close()


@app.get("/api/me/entity")
def get_me_entity():
    """Get the Me entity details."""
    from mnemosyne.models import Entity
    from mnemosyne.database import get_session_factory
    db = get_session_factory()()
    try:
        me = db.query(Entity).filter(Entity.name == "Me").first()
        if not me:
            return {"found": False}
        return {
            "found": True,
            "id": me.id,
            "name": me.name,
            "type": me.type,
            "confidence": me.confidence,
        }
    finally:
        db.close()


@app.get("/api/me/onboarding/status")
def onboarding_status():
    """Check if onboarding is needed."""
    from mnemosyne.services.owner_compiler import OwnerCompiler
    from mnemosyne.database import get_session_factory
    db = get_session_factory()()
    try:
        compiler = OwnerCompiler()
        return compiler.get_onboarding_status(db)
    finally:
        db.close()


@app.post("/api/me/onboarding/answer")
def onboarding_answer(req: dict):
    """Submit an onboarding answer."""
    from mnemosyne.services.owner_compiler import OwnerCompiler
    from mnemosyne.database import get_session_factory
    key = req.get("key", "")
    answer = req.get("answer", "")
    if not key or not answer:
        raise HTTPException(status_code=400, detail="key and answer required")
    db = get_session_factory()()
    try:
        compiler = OwnerCompiler()
        return compiler.answer_onboarding(db, key, answer)
    finally:
        db.close()


@app.get("/api/me/profile")
def me_profile():
    """Get Me's full profile."""
    from mnemosyne.services.owner_compiler import OwnerCompiler
    from mnemosyne.database import get_session_factory
    db = get_session_factory()()
    try:
        compiler = OwnerCompiler()
        return compiler.get_me_profile(db)
    finally:
        db.close()


# --- Graph Explorer API ---


@app.get("/api/graph/seed")
def graph_seed(limit: int = 30, after_date: str | None = None, before_date: str | None = None):
    """Get initial graph seed for Cytoscape.js.

    Args:
        limit: Maximum entities to return.
        after_date: Only include entities with facts after this date (YYYY-MM-DD).
        before_date: Only include entities with facts before this date (YYYY-MM-DD).
    """
    from mnemosyne.services.graph_explorer import GraphExplorerService
    svc = GraphExplorerService()
    return svc.get_initial_graph(limit=limit, after_date=after_date, before_date=before_date)


@app.get("/api/graph/search")
def graph_search(q: str = "", limit: int = 20):
    """Search entities by name."""
    from mnemosyne.services.graph_explorer import GraphExplorerService
    svc = GraphExplorerService()
    return svc.search_entities(q, limit=limit)


@app.get("/api/graph/node/{entity_id}")
def graph_node_detail(entity_id: int):
    """Get full detail for a single node."""
    from mnemosyne.services.graph_explorer import GraphExplorerService
    svc = GraphExplorerService()
    result = svc.get_node_detail(entity_id)
    if not result:
        raise HTTPException(status_code=404, detail="Entity not found")
    return result


@app.get("/api/graph/edge/{relationship_id}")
def graph_edge_detail(relationship_id: int):
    """Get full detail for a single edge."""
    from mnemosyne.services.graph_explorer import GraphExplorerService
    svc = GraphExplorerService()
    result = svc.get_edge_detail(relationship_id)
    if not result:
        raise HTTPException(status_code=404, detail="Relationship not found")
    return result


@app.get("/api/graph/neighbors/{entity_id}")
def graph_neighbors(entity_id: int, hops: int = 1):
    """Get neighbors for lazy expansion."""
    from mnemosyne.services.graph_explorer import GraphExplorerService
    svc = GraphExplorerService()
    return svc.get_neighbors(entity_id, hops=hops)


@app.get("/api/graph/types")
def graph_types():
    """Get all entity types with counts."""
    from mnemosyne.services.graph_explorer import GraphExplorerService
    svc = GraphExplorerService()
    return {"types": svc.get_types()}


@app.get("/api/graph/stats")
def graph_stats():
    """Get graph statistics."""
    from mnemosyne.services.graph_explorer import GraphExplorerService
    svc = GraphExplorerService()
    return svc.get_statistics()


# --- Data Importer API ---


@app.post("/api/import/text")
def import_text(req: ImportRequest):
    """Start a text import job."""
    svc = get_import_service()
    job_id = svc.import_text(req.text)
    return {"job_id": job_id, "status": "running"}


@app.post("/api/import/file")
async def import_file(file: UploadFile = File(...)):
    """Start a file import job."""
    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    svc = get_import_service()
    job_id = svc.import_file(file.filename or "unknown", text, file.content_type or "text/plain")
    return {"job_id": job_id, "status": "running"}


@app.get("/api/import/job/{job_id}")
def import_job_status(job_id: str):
    """Get import job status and progress."""
    svc = get_import_service()
    result = svc.get_job(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job not found")
    return result


@app.post("/api/import/cancel/{job_id}")
def import_cancel(job_id: str):
    """Cancel a running import job."""
    svc = get_import_service()
    cancelled = svc.cancel_job(job_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="Job not found or already completed")
    return {"status": "cancelled"}


# --- Memory Consolidator API ---


@app.post("/api/consolidate/analyze")
def consolidate_analyze():
    """Run all consolidation analyses and return report."""
    from mnemosyne.services.consolidation import ConsolidationService
    svc = ConsolidationService()
    return svc.analyze()


@app.post("/api/consolidate/apply")
def consolidate_apply(req: ApplyRequest):
    """Apply or reject consolidation recommendations."""
    from mnemosyne.services.consolidation import ConsolidationService
    svc = ConsolidationService()
    actions = [{"index": a.id, "action": a.action} for a in req.actions]
    return svc.apply_recommendations(actions)
