"""Backward-compatible retrieval wrapper using the new deterministic pipeline."""

from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from mnemosyne.embeddings import EmbeddingService
from mnemosyne.graph import GraphService
from mnemosyne.retrieval.pipeline import ContextPipeline

logger = logging.getLogger(__name__)


class RetrievalService:
    """Backward-compatible interface to the new deterministic context pipeline.

    Delegates to ContextPipeline for all retrieval logic.
    """

    def __init__(self, embeddings: EmbeddingService, graph: GraphService) -> None:
        self._embeddings = embeddings
        self._graph = graph
        self._pipeline = ContextPipeline(embeddings, graph)

    def retrieve(self, db: Session, query: str, top_k: int = 5) -> dict:
        """Run the deterministic pipeline and return results in legacy format."""
        result = self._pipeline.run(db, query)

        vector_results = []
        if result.memory_result:
            vector_results = result.memory_result.get("memories", [])

        graph_context = {
            "entities": [],
            "relationships": [],
            "facts": [],
            "connected_entities": [],
        }
        if result.graph_result:
            graph_context["entities"] = result.graph_result.get("entities", [])
            graph_context["relationships"] = result.graph_result.get("relationships", [])
            graph_context["facts"] = result.graph_result.get("facts", [])

        return {
            "vector_results": vector_results,
            "graph_context": graph_context,
            "merged_context": result.context,
        }
