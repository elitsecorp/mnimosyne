"""Deterministic graph traversal and fact retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from mnemosyne.retrieval.resolver import ResolvedEntity


@dataclass
class GraphResult:
    """Result of graph retrieval."""

    entities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    facts: list[dict] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


class GraphRetriever:
    """Traverses the knowledge graph from resolved entities.

    Uses BFS with configurable depth. Scores results by distance and confidence.
    All deterministic — no LLM calls.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self._graph = graph

    def retrieve(
        self,
        resolved: list[ResolvedEntity],
        max_hops: int = 2,
        min_confidence: float = 0.0,
    ) -> GraphResult:
        """Traverse graph from resolved entities and collect relevant knowledge."""
        result = GraphResult()
        visited_edges: set[tuple] = set()
        entity_scores: dict[str, float] = {}

        for ent in resolved:
            if not self._graph.has_node(ent.name):
                continue
            entity_scores[ent.name] = max(
                entity_scores.get(ent.name, 0),
                ent.match_score * ent.confidence,
            )

        start_names = {ent.name for ent in resolved}

        for ent in resolved:
            if not self._graph.has_node(ent.name):
                continue
            self._bfs_collect(
                ent.name,
                max_hops,
                min_confidence,
                entity_scores,
                visited_edges,
                result,
                start_names,
            )

        for name, score in entity_scores.items():
            result.scores[name] = score

        return result

    def _bfs_collect(
        self,
        start: str,
        max_hops: int,
        min_confidence: float,
        entity_scores: dict[str, float],
        visited_edges: set[tuple],
        result: GraphResult,
        start_names: set[str] | None = None,
    ) -> None:
        """BFS from start entity, collecting relationships and scoring by distance."""
        if start_names is None:
            start_names = {start}
        queue: list[tuple[str, int]] = [(start, 0)]
        visited: set[str] = {start}

        while queue:
            node, depth = queue.pop(0)
            if depth >= max_hops:
                continue

            for _, neighbor, data in self._graph.out_edges(node, data=True):
                pred = data.get("predicate", "related_to")
                conf = data.get("confidence", 0)
                edge_key = (node, pred, neighbor)

                if edge_key in visited_edges:
                    continue
                if conf < min_confidence:
                    continue

                visited_edges.add(edge_key)
                distance = depth + 1
                edge_score = (1 / distance) * conf

                result.relationships.append({
                    "subject": node,
                    "predicate": pred,
                    "object": neighbor,
                    "confidence": conf,
                    "distance": distance,
                    "score": edge_score,
                })

                entity_scores[neighbor] = max(
                    entity_scores.get(neighbor, 0),
                    edge_score,
                )

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance))

            for neighbor, _, data in self._graph.in_edges(node, data=True):
                pred = data.get("predicate", "related_to")
                conf = data.get("confidence", 0)
                edge_key = (neighbor, pred, node)

                if edge_key in visited_edges:
                    continue
                if conf < min_confidence:
                    continue

                visited_edges.add(edge_key)
                distance = depth + 1
                edge_score = (1 / distance) * conf

                result.relationships.append({
                    "subject": neighbor,
                    "predicate": pred,
                    "object": node,
                    "confidence": conf,
                    "distance": distance,
                    "score": edge_score,
                })

                entity_scores[neighbor] = max(
                    entity_scores.get(neighbor, 0),
                    edge_score,
                )

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance))

        for name in visited:
            if self._graph.has_node(name):
                attrs = self._graph.nodes[name]
                result.entities.append({
                    "name": name,
                    "type": attrs.get("type", ""),
                    "confidence": attrs.get("confidence", 0),
                    "distance": 0 if name in start_names else 1,
                })

    def get_direct_facts(self, entity_name: str) -> list[dict]:
        """Get all facts directly involving an entity (1-hop only)."""
        if not self._graph.has_node(entity_name):
            return []
        facts = []
        for _, neighbor, data in self._graph.out_edges(entity_name, data=True):
            facts.append({
                "subject": entity_name,
                "predicate": data.get("predicate", "related_to"),
                "object": neighbor,
                "confidence": data.get("confidence", 0),
            })
        for neighbor, _, data in self._graph.in_edges(entity_name, data=True):
            facts.append({
                "subject": neighbor,
                "predicate": data.get("predicate", "related_to"),
                "object": entity_name,
                "confidence": data.get("confidence", 0),
            })
        return facts
