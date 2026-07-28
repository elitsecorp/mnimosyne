"""Pydantic models for the Mnemosyne Public API v2."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# --- Enums ---


class MemoryType(str, Enum):
    EPISODE = "episode"
    BELIEF = "belief"
    FACT = "fact"
    PROCEDURE = "procedure"
    CONCEPT = "concept"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


# --- Memory ---


class Evidence(BaseModel):
    source_id: str = Field(description="ID of the source memory or entity")
    source_type: str = Field(description="Type of source (message, entity, relationship)")
    text: str = Field(description="Evidence text or summary")
    relevance: float = Field(description="Relevance score 0.0-1.0")
    timestamp: datetime | None = Field(default=None, description="When the evidence was created")


class Memory(BaseModel):
    id: str = Field(description="Unique memory identifier")
    summary: str = Field(description="Human-readable summary of the memory")
    content: str = Field(description="Full content of the memory")
    memory_type: MemoryType = Field(description="Type of memory")
    relevance: float = Field(description="Relevance score 0.0-1.0")
    confidence: float = Field(description="Confidence in this memory 0.0-1.0")
    created_at: datetime = Field(description="When the memory was created")
    updated_at: datetime | None = Field(default=None, description="When the memory was last updated")
    evidence: list[Evidence] = Field(default_factory=list, description="Supporting evidence")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class MemorySearchRequest(BaseModel):
    query: str = Field(description="Search query in natural language")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context for search")
    include_evidence: bool = Field(default=True, description="Include supporting evidence")
    memory_types: list[MemoryType] | None = Field(default=None, description="Filter by memory types")
    min_relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum relevance threshold")


class MemorySearchResponse(BaseModel):
    memories: list[Memory] = Field(description="List of relevant memories")
    total: int = Field(description="Total number of matching memories")
    query: str = Field(description="Original search query")
    elapsed_ms: float = Field(description="Search time in milliseconds")


class MemoryContextRequest(BaseModel):
    query: str = Field(description="User query to get context for")
    max_tokens: int = Field(default=4000, ge=100, le=32000, description="Maximum tokens for context")
    include_conversation: bool = Field(default=True, description="Include conversation history")
    include_graph: bool = Field(default=True, description="Include graph context")
    include_memories: bool = Field(default=True, description="Include vector memories")


class MemoryContextResponse(BaseModel):
    context: str = Field(description="Assembled context for the query")
    memories: list[Memory] = Field(description="Memories included in context")
    sources: list[str] = Field(description="Sources used (conversation, graph, vector)")
    token_estimate: int = Field(description="Estimated token count")
    query: str = Field(description="Original query")


class ExperienceRequest(BaseModel):
    observations: list[str] = Field(description="List of observations to remember")
    source: str | None = Field(default=None, description="Source of the experience")
    context: dict[str, Any] = Field(default_factory=dict, description="Additional context")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance level")


class ExperienceResponse(BaseModel):
    experience_id: str = Field(description="ID of the stored experience")
    status: str = Field(description="Status (accepted, processed, rejected)")
    entities_created: int = Field(description="Number of entities created")
    relationships_created: int = Field(description="Number of relationships created")
    message: str = Field(description="Status message")


class ReflectResponse(BaseModel):
    beliefs_created: int = Field(description="Number of new beliefs generated")
    beliefs_updated: int = Field(description="Number of beliefs updated")
    concepts_merged: int = Field(description="Number of concepts merged")
    contradictions_found: int = Field(description="Number of contradictions detected")
    summary: str = Field(description="Reflection summary")
    changes: list[dict[str, Any]] = Field(description="List of changes made")


class ExplainRequest(BaseModel):
    target_id: str = Field(description="ID of memory, belief, or relationship to explain")
    target_type: str = Field(description="Type (memory, belief, relationship, entity)")
    depth: int = Field(default=3, ge=1, le=10, description="Explanation depth")


class ExplainResponse(BaseModel):
    target_id: str = Field(description="ID of the explained item")
    target_type: str = Field(description="Type of the explained item")
    summary: str = Field(description="Human-readable explanation")
    evidence_chain: list[Evidence] = Field(description="Chain of evidence")
    reasoning: list[str] = Field(description="Step-by-step reasoning")
    confidence: float = Field(description="Confidence in the explanation")


# --- Concepts ---


class Concept(BaseModel):
    id: str = Field(description="Unique concept identifier")
    name: str = Field(description="Concept name")
    description: str | None = Field(default=None, description="Concept description")
    type: str = Field(description="Concept type")
    confidence: float = Field(description="Confidence 0.0-1.0")
    frequency: int = Field(description="How often this concept appears")
    created_at: datetime = Field(description="Creation timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConceptSearchRequest(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=10, ge=1, le=100)
    type_filter: str | None = Field(default=None, description="Filter by concept type")


class ConceptSearchResponse(BaseModel):
    concepts: list[Concept] = Field(description="Matching concepts")
    total: int = Field(description="Total matches")


class ConceptMergeRequest(BaseModel):
    source_ids: list[str] = Field(description="IDs of concepts to merge")
    target_name: str | None = Field(default=None, description="Name for merged concept")
    strategy: str = Field(default="keep_highest_confidence", description="Merge strategy")


class ConceptCreateRequest(BaseModel):
    name: str = Field(description="Concept name")
    description: str | None = Field(default=None)
    type: str = Field(default="general", description="Concept type")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)


# --- Entities ---


class Entity(BaseModel):
    id: str = Field(description="Unique entity identifier")
    name: str = Field(description="Entity name")
    entity_type: str = Field(description="Entity type (person, place, concept, etc.)")
    confidence: float = Field(description="Confidence 0.0-1.0")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime | None = Field(default=None)
    relationships_count: int = Field(default=0, description="Number of relationships")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EntitySearchRequest(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=10, ge=1, le=100)
    type_filter: str | None = Field(default=None)


class EntitySearchResponse(BaseModel):
    entities: list[Entity] = Field(description="Matching entities")
    total: int = Field(description="Total matches")


class EntityUpdateRequest(BaseModel):
    name: str | None = Field(default=None)
    entity_type: str | None = Field(default=None)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] | None = Field(default=None)


# --- Relationships ---


class Relationship(BaseModel):
    id: str = Field(description="Unique relationship identifier")
    subject: str = Field(description="Subject entity name")
    predicate: str = Field(description="Relationship type")
    object: str = Field(description="Object entity name")
    confidence: float = Field(description="Confidence 0.0-1.0")
    created_at: datetime = Field(description="Creation timestamp")
    last_seen: datetime | None = Field(default=None)
    evidence_count: int = Field(default=0, description="Number of supporting evidence")
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationshipSearchRequest(BaseModel):
    query: str = Field(description="Search query or subject name")
    subject: str | None = Field(default=None, description="Filter by subject")
    predicate: str | None = Field(default=None, description="Filter by predicate")
    object: str | None = Field(default=None, description="Filter by object")
    limit: int = Field(default=10, ge=1, le=100)


class RelationshipSearchResponse(BaseModel):
    relationships: list[Relationship] = Field(description="Matching relationships")
    total: int = Field(description="Total matches")


class RelationshipCreateRequest(BaseModel):
    subject: str = Field(description="Subject entity name")
    predicate: str = Field(description="Relationship type")
    object: str = Field(description="Object entity name")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list, description="Supporting evidence texts")


class RelationshipUpdateRequest(BaseModel):
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] | None = Field(default=None)


# --- Beliefs ---


class Belief(BaseModel):
    id: str = Field(description="Unique belief identifier")
    statement: str = Field(description="The belief statement")
    confidence: float = Field(description="Confidence 0.0-1.0")
    supporting_evidence: list[Evidence] = Field(description="Evidence supporting this belief")
    contradicting_evidence: list[Evidence] = Field(description="Evidence contradicting this belief")
    revision_history: list[dict[str, Any]] = Field(description="History of confidence revisions")
    created_at: datetime = Field(description="Creation timestamp")
    last_validated: datetime | None = Field(default=None)
    status: str = Field(description="Status (active, challenged, invalidated)")


class BeliefSearchRequest(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=10, ge=1, le=100)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: str | None = Field(default=None, description="Filter by status")


class BeliefSearchResponse(BaseModel):
    beliefs: list[Belief] = Field(description="Matching beliefs")
    total: int = Field(description="Total matches")


class BeliefValidateRequest(BaseModel):
    belief_id: str = Field(description="ID of belief to validate")
    evidence_text: str = Field(description="New evidence to validate against")


class BeliefValidateResponse(BaseModel):
    belief_id: str = Field(description="Belief ID")
    confidence_before: float = Field(description="Confidence before validation")
    confidence_after: float = Field(description="Confidence after validation")
    status: str = Field(description="New status")
    supporting_evidence: list[Evidence] = Field(description="Updated supporting evidence")


class BeliefChallengeRequest(BaseModel):
    belief_id: str = Field(description="ID of belief to challenge")
    challenge_text: str = Field(description="Challenge or contradicting evidence")


class BeliefExplainRequest(BaseModel):
    belief_id: str = Field(description="ID of belief to explain")
    depth: int = Field(default=3, ge=1, le=10)


# --- Episodes ---


class Episode(BaseModel):
    id: str = Field(description="Unique episode identifier")
    summary: str = Field(description="Episode summary")
    content: str = Field(description="Full episode content")
    timestamp: datetime = Field(description="When the episode occurred")
    participants: list[str] = Field(description="Entities involved")
    key_concepts: list[str] = Field(description="Key concepts mentioned")
    importance: float = Field(description="Importance 0.0-1.0")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeSearchRequest(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=10, ge=1, le=100)
    after: datetime | None = Field(default=None, description="Filter after date")
    before: datetime | None = Field(default=None, description="Filter before date")


class EpisodeSearchResponse(BaseModel):
    episodes: list[Episode] = Field(description="Matching episodes")
    total: int = Field(description="Total matches")


# --- Procedures ---


class Procedure(BaseModel):
    id: str = Field(description="Unique procedure identifier")
    name: str = Field(description="Procedure name")
    description: str = Field(description="What this procedure does")
    steps: list[str] = Field(description="Steps to follow")
    triggers: list[str] = Field(description="When to use this procedure")
    confidence: float = Field(description="Confidence in this procedure")
    created_at: datetime = Field(description="Creation timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcedureSearchRequest(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=10, ge=1, le=100)


class ProcedureSearchResponse(BaseModel):
    procedures: list[Procedure] = Field(description="Matching procedures")
    total: int = Field(description="Total matches")


# --- World Model ---


class WorldModel(BaseModel):
    major_entities: list[Entity] = Field(description="Most important entities")
    major_concepts: list[Concept] = Field(description="Most important concepts")
    belief_summaries: list[Belief] = Field(description="Key beliefs")
    recent_episodes: list[Episode] = Field(description="Recent episodes")
    active_procedures: list[Procedure] = Field(description="Active procedures")
    statistics: dict[str, Any] = Field(description="World model statistics")
    last_updated: datetime = Field(description="When the world model was last updated")


# --- Graph Exploration ---


class GraphNeighborRequest(BaseModel):
    concept: str = Field(description="Concept or entity name to explore")
    depth: int = Field(default=1, ge=1, le=5, description="Traversal depth")
    limit: int = Field(default=20, ge=1, le=100)


class GraphNeighborResponse(BaseModel):
    center: str = Field(description="Center concept")
    neighbors: list[dict[str, Any]] = Field(description="Neighbor nodes and edges")
    total: int = Field(description="Total neighbors found")


class GraphPathRequest(BaseModel):
    source: str = Field(description="Source concept or entity")
    target: str = Field(description="Target concept or entity")
    max_depth: int = Field(default=5, ge=1, le=10)


class GraphPathResponse(BaseModel):
    source: str = Field(description="Source concept")
    target: str = Field(description="Target concept")
    path: list[dict[str, Any]] = Field(description="Path of relationships")
    path_length: int = Field(description="Number of hops")
    found: bool = Field(description="Whether a path was found")


# --- Reasoning ---


class ReasoningQueryRequest(BaseModel):
    question: str = Field(description="Question to answer from knowledge base")
    max_evidence: int = Field(default=10, ge=1, le=50, description="Maximum evidence items")


class ReasoningQueryResponse(BaseModel):
    question: str = Field(description="Original question")
    answer: str = Field(description="Answer from knowledge base")
    confidence: float = Field(description="Confidence in answer 0.0-1.0")
    supporting_evidence: list[Evidence] = Field(description="Evidence used")
    related_concepts: list[str] = Field(description="Related concepts")
    reasoning_steps: list[str] = Field(description="Step-by-step reasoning")


class ReasoningExplainRequest(BaseModel):
    query: str = Field(description="What to explain")
    depth: int = Field(default=3, ge=1, le=10)


class ReasoningExplainResponse(BaseModel):
    query: str = Field(description="Original query")
    explanation: str = Field(description="Full explanation")
    evidence_chain: list[Evidence] = Field(description="Evidence chain")
    confidence: float = Field(description="Confidence in explanation")


# --- Administration ---


class HealthResponse(BaseModel):
    status: str = Field(description="System status")
    version: str = Field(description="API version")
    embedding_backend: str = Field(description="Embedding backend")
    llm_provider: str = Field(description="LLM provider")
    graph_nodes: int = Field(description="Number of graph nodes")
    graph_edges: int = Field(description="Number of graph edges")
    total_memories: int = Field(description="Total memories stored")


class StatisticsResponse(BaseModel):
    entities: int = Field(description="Total entities")
    relationships: int = Field(description="Total relationships")
    facts: int = Field(description="Total facts")
    messages: int = Field(description="Total messages")
    embeddings: int = Field(description="Total embeddings")
    beliefs: int = Field(description="Total beliefs")
    episodes: int = Field(description="Total episodes")
    procedures: int = Field(description="Total procedures")
    avg_confidence: float = Field(description="Average confidence across all items")


class VersionResponse(BaseModel):
    version: str = Field(description="API version")
    build: str = Field(description="Build identifier")
    python_version: str = Field(description="Python version")
    features: list[str] = Field(description="Enabled features")


# --- Errors ---


class ErrorResponse(BaseModel):
    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    details: dict[str, Any] | None = Field(default=None, description="Additional error details")
    request_id: str | None = Field(default=None, description="Request ID for debugging")
