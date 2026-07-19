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
    StatsResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mnemosyne",
    description="Persistent memory system for Large Language Models",
    version="0.2.0",
)

_engine: MemoryEngine | None = None


def get_engine() -> MemoryEngine:
    """Return the singleton MemoryEngine, creating it on first call."""
    global _engine
    if _engine is None:
        _engine = MemoryEngine()
    return _engine


@app.on_event("startup")
def _startup() -> None:
    """Initialize database and graph on startup."""
    from mnemosyne.database import init_db, get_session_factory
    init_db()
    engine = get_engine()
    # Load the full graph from DB into NetworkX
    db = get_session_factory()()
    try:
        engine._graph.load_from_db(db)
    finally:
        db.close()
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


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Process a user message through the memory pipeline."""
    try:
        engine = get_engine()
        result = engine.chat(req.message)
        return ChatResponse(response=result["response"], pipeline=result.get("pipeline"))
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


# --- Graph Explorer API ---


@app.get("/api/graph/seed")
def graph_seed(limit: int = 30):
    """Get initial graph seed for Cytoscape.js."""
    from mnemosyne.services.graph_explorer import GraphExplorerService
    svc = GraphExplorerService()
    return svc.get_initial_graph(limit=limit)


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
    from mnemosyne.services.importer import ImportService
    svc = ImportService(engine=get_engine())
    job_id = svc.import_text(req.text)
    return {"job_id": job_id, "status": "running"}


@app.post("/api/import/file")
async def import_file(file: UploadFile = File(...)):
    """Start a file import job."""
    from mnemosyne.services.importer import ImportService
    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    svc = ImportService(engine=get_engine())
    job_id = svc.import_file(file.filename or "unknown", text, file.content_type or "text/plain")
    return {"job_id": job_id, "status": "running"}


@app.get("/api/import/job/{job_id}")
def import_job_status(job_id: str):
    """Get import job status and progress."""
    from mnemosyne.services.importer import ImportService
    svc = ImportService(engine=get_engine())
    result = svc.get_job(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job not found")
    return result


@app.post("/api/import/cancel/{job_id}")
def import_cancel(job_id: str):
    """Cancel a running import job."""
    from mnemosyne.services.importer import ImportService
    svc = ImportService(engine=get_engine())
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
