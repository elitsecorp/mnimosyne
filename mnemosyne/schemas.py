"""Pydantic schemas for API input/output and internal validation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request."""

    message: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    """Outgoing chat response."""

    response: str = Field(..., description="Assistant response text")
    pipeline: dict | None = Field(default=None, description="Pipeline metadata")


class SearchRequest(BaseModel):
    """Vector similarity search request."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class SearchResult(BaseModel):
    """Single search result."""

    id: int
    message_id: int
    text: str
    score: float


class SearchResponse(BaseModel):
    """Vector search response."""

    results: list[SearchResult]


class GraphEntity(BaseModel):
    """Entity in graph response."""

    name: str
    type: str
    confidence: float = 0.0


class GraphRelationship(BaseModel):
    """Relationship in graph response."""

    subject: str
    predicate: str
    object: str


class GraphResponse(BaseModel):
    """Graph neighborhood response."""

    entity: str
    relationships: list[GraphRelationship]
    connected_entities: list[GraphEntity]


class EntitySchema(BaseModel):
    """Extracted entity."""

    name: str
    type: str
    confidence: float = 0.9


class RelationshipSchema(BaseModel):
    """Extracted relationship."""

    subject: str
    predicate: str
    object: str
    confidence: float = 0.9


class FactSchema(BaseModel):
    """Extracted fact."""

    subject: str
    predicate: str
    object: str


class ExtractionResult(BaseModel):
    """Complete extraction result from a conversation turn."""

    entities: list[EntitySchema] = []
    relationships: list[RelationshipSchema] = []
    facts: list[FactSchema] = []


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"


class EntityOut(BaseModel):
    """Entity for explorer output."""

    id: int
    name: str
    type: str
    confidence: float


class RelationshipOut(BaseModel):
    """Relationship for explorer output."""

    id: int
    subject: str
    predicate: str
    object: str
    confidence: float


class FactOut(BaseModel):
    """Fact for explorer output."""

    id: int
    subject: str
    predicate: str
    object: str
    source_message: str
    timestamp: str


class MessageOut(BaseModel):
    """Message for explorer output."""

    id: int
    role: str
    content: str
    timestamp: str


class StatsResponse(BaseModel):
    """Summary counts."""

    messages: int = 0
    embeddings: int = 0
    entities: int = 0
    relationships: int = 0
    facts: int = 0


# --- Graph Explorer ---


class GraphNode(BaseModel):
    """Node for Cytoscape.js graph visualization."""

    id: int
    name: str
    type: str
    confidence: float
    degree: int = 0


class GraphEdge(BaseModel):
    """Edge for Cytoscape.js graph visualization."""

    id: int
    source: int
    target: int
    predicate: str
    confidence: float


class GraphSeedResponse(BaseModel):
    """Initial graph data for Cytoscape.js."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]


class GraphNodeRelationship(BaseModel):
    """Relationship in node detail view."""

    id: int
    subject: str
    predicate: str
    object: str
    confidence: float


class GraphNodeEvidence(BaseModel):
    """Evidence message supporting a node."""

    id: int
    content: str
    timestamp: str


class GraphNodeDetail(BaseModel):
    """Full detail for a single node."""

    id: int
    name: str
    type: str
    confidence: float
    relationships: list[GraphNodeRelationship]
    evidence: list[GraphNodeEvidence]


class GraphEdgeDetail(BaseModel):
    """Full detail for a single edge."""

    id: int
    predicate: str
    confidence: float
    subject: str
    object: str
    evidence: list[GraphNodeEvidence]


class GraphTypeCount(BaseModel):
    """Entity type with count."""

    type: str
    count: int


class GraphTypesResponse(BaseModel):
    """All entity types with counts."""

    types: list[GraphTypeCount]


class GraphStatsResponse(BaseModel):
    """Graph statistics."""

    nodes: int
    edges: int
    types: list[GraphTypeCount]
    avg_confidence: float


# --- Data Ingester ---


class ImportRequest(BaseModel):
    """Text import request."""

    text: str = Field(..., min_length=1)


class ImportResponse(BaseModel):
    """Import job started."""

    job_id: str
    status: str = "running"


class ImportJobStatus(BaseModel):
    """Import job progress."""

    id: str
    status: str
    total_chunks: int
    processed_chunks: int
    entities_extracted: int
    relationships_extracted: int
    embeddings_created: int
    errors: list[str]
    started_at: str
    completed_at: str | None = None


# --- Memory Consolidator ---


class ConsolidationRecommendation(BaseModel):
    """Single consolidation recommendation."""

    id: str
    type: str
    reason: str
    confidence: float
    evidence: list[str]
    affected_nodes: list[str]
    proposed_action: dict
    status: str = "pending"


class ConsolidationSummary(BaseModel):
    """Summary of consolidation analysis."""

    duplicate_entities: int
    normalize_relationships: int
    duplicate_relationships: int
    orphans: int
    confidence_changes: int


class ConsolidationReport(BaseModel):
    """Full consolidation report."""

    recommendations: list[ConsolidationRecommendation]
    summary: ConsolidationSummary


class ApplyAction(BaseModel):
    """Single apply/reject action."""

    id: int  # recommendation index
    action: str  # "approve" or "reject"


class ApplyRequest(BaseModel):
    """Apply approved recommendations."""

    actions: list[ApplyAction]


class ApplyResponse(BaseModel):
    """Result of applying recommendations."""

    applied: int
    rejected: int
    errors: list[str]
