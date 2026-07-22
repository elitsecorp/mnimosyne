"""Orchestrates the full deterministic context generation pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from mnemosyne.embeddings import EmbeddingService
from mnemosyne.graph import GraphService
from mnemosyne.retrieval.builder import ContextBuilder
from mnemosyne.retrieval.compressor import Compressor
from mnemosyne.retrieval.deduplicator import Deduplicator
from mnemosyne.retrieval.graph_retriever import GraphRetriever
from mnemosyne.retrieval.memory_retriever import MemoryRetriever
from mnemosyne.retrieval.planner import QueryPlan, QueryPlanner
from mnemosyne.retrieval.ranker import Ranker, ScoredItem
from mnemosyne.retrieval.resolver import EntityResolver, ResolvedEntity

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of the full context generation pipeline."""

    context: str
    plan: QueryPlan
    resolved_entities: list[ResolvedEntity]
    scored_items: list[ScoredItem]
    graph_result: dict | None = None
    memory_result: dict | None = None
    stats: dict = field(default_factory=dict)
    user_message: str = ""
    conversation_length: int = 0


class ContextPipeline:
    """Deterministic context generation pipeline.

    Replaces the old "retrieve everything and let LLM figure it out" approach
    with a structured pipeline that performs query analysis, entity resolution,
    ranked graph traversal, selective vector search, deduplication, compression,
    and context construction.
    """

    def __init__(
        self,
        embeddings: EmbeddingService,
        graph: GraphService,
        max_hops: int = 2,
        min_confidence: float = 0.0,
        token_budget: int = 4000,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._graph = graph
        self._max_hops = max_hops
        self._min_confidence = min_confidence
        self._token_budget = token_budget
        self._weights = weights

    def _extract_entities_from_memories(self, memories: list[dict]) -> list[str]:
        """Extract entity names from vector memory results.

        Uses existing graph entities as a vocabulary to identify mentions.
        """
        known_entities = set(self._graph.graph.nodes())
        entity_mentions: dict[str, int] = {}

        for mem in memories:
            text = mem.get("text", "").lower()
            for entity in known_entities:
                if entity.lower() in text:
                    entity_mentions[entity] = entity_mentions.get(entity, 0) + 1

        sorted_entities = sorted(entity_mentions.items(), key=lambda x: x[1], reverse=True)
        return [e[0] for e in sorted_entities[:10]]

    def run(
        self,
        db: Session,
        query: str,
        conversation: list[dict[str, str]] | None = None,
        query_vector: list[float] | None = None,
    ) -> PipelineResult:
        """Execute the full deterministic pipeline.

        Args:
            query_vector: Pre-computed embedding to avoid duplicate API call.
        """
        import time
        start = time.time()

        known_entities = [n for n in self._graph.graph.nodes()]
        planner = QueryPlanner(known_entities=known_entities)
        plan = planner.analyze(query)

        logger.info("Query plan: type=%s, entities=%s, graph=%s, vector=%s",
                     plan.query_type, plan.detected_entities, plan.graph_enabled, plan.vector_enabled)

        resolved = []
        if plan.graph_enabled:
            resolver = EntityResolver(db)
            resolved = resolver.resolve(query, limit=10)

        graph_result = None
        entity_scores: dict[str, float] = {}
        all_items: list[ScoredItem] = []
        vector_to_graph_info = {
            "entities_from_message": plan.detected_entities,
            "entities_from_vector": [],
            "relationships_found": [],
            "entities_fed": [],
        }

        if plan.graph_enabled and resolved:
            retriever = GraphRetriever(self._graph.graph)
            graph_result = retriever.retrieve(
                resolved,
                max_hops=plan.max_hops,
                min_confidence=self._min_confidence,
            )
            entity_scores = graph_result.scores

            ranker = Ranker(self._weights)
            ent_items = ranker.rank_entities(graph_result.entities, plan.detected_entities, entity_scores)
            rel_items = ranker.rank_relationships(graph_result.relationships, plan.detected_entities)
            fact_items = ranker.rank_facts(graph_result.facts, plan.detected_entities)
            all_items.extend(ent_items)
            all_items.extend(rel_items)
            all_items.extend(fact_items)

        memory_result = None
        if plan.vector_enabled:
            query_len = len(query.split())
            vec_top_k = 10 if query_len > 3 else 5
            min_sim = 0.6

            mem_retriever = MemoryRetriever(self._embeddings)
            memory_result = mem_retriever.retrieve(
                db, query, top_k=vec_top_k, min_similarity=min_sim, query_vector=query_vector,
            )
            ranker = Ranker(self._weights)
            mem_items = ranker.rank_memories(memory_result.memories, plan.detected_entities)
            all_items.extend(mem_items)

            if memory_result and memory_result.memories:
                vector_entities = self._extract_entities_from_memories(memory_result.memories)
                vector_to_graph_info["entities_from_vector"] = vector_entities
                if vector_entities and plan.graph_enabled:
                    resolver = EntityResolver(db)
                    vector_resolved = resolver.resolve(" ".join(vector_entities), limit=10)
                    seen_names = {e.name for e in resolved}
                    new_resolved = [e for e in vector_resolved if e.name not in seen_names]
                    vector_to_graph_info["entities_fed"] = [e.name for e in new_resolved]
                    if new_resolved:
                        resolved.extend(new_resolved)
                        retriever = GraphRetriever(self._graph.graph)
                        extra_result = retriever.retrieve(
                            new_resolved,
                            max_hops=1,
                            min_confidence=self._min_confidence,
                        )
                        entity_scores.update(extra_result.scores)
                        ranker = Ranker(self._weights)
                        ent_items = ranker.rank_entities(extra_result.entities, plan.detected_entities, entity_scores)
                        rel_items = ranker.rank_relationships(extra_result.relationships, plan.detected_entities)
                        all_items.extend(ent_items)
                        all_items.extend(rel_items)
                        vector_to_graph_info["relationships_found"] = [
                            {"subject": r["subject"], "predicate": r["predicate"], "object": r["object"]}
                            for r in extra_result.relationships[:10]
                        ]

        deduped = Deduplicator().dedup(all_items)
        compressed = Compressor(self._token_budget).compress(deduped)
        context = ContextBuilder().build(compressed, plan, conversation)

        elapsed = time.time() - start
        stats = {
            "resolved_entities": len(resolved),
            "graph_items": len([i for i in all_items if i.item_type != "memory"]),
            "memory_items": len([i for i in all_items if i.item_type == "memory"]),
            "after_dedup": len(deduped),
            "after_compress": len(compressed),
            "context_chars": len(context),
            "elapsed_ms": round(elapsed * 1000, 1),
            "vector_to_graph": vector_to_graph_info,
        }

        logger.info("Pipeline complete: %s", stats)

        return PipelineResult(
            context=context,
            plan=plan,
            resolved_entities=resolved,
            scored_items=compressed,
            graph_result=graph_result.__dict__ if graph_result else None,
            memory_result=memory_result.__dict__ if memory_result else None,
            stats=stats,
            user_message=query,
            conversation_length=len(conversation) if conversation else 0,
        )
