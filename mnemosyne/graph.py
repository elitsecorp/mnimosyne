"""NetworkX graph service for entity-relationship ontology."""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
from sqlalchemy.orm import Session

from mnemosyne.models import Entity, Fact, Relationship

logger = logging.getLogger(__name__)


class GraphService:
    """Manages the in-memory knowledge graph backed by SQLite.

    Entities are nodes, relationships are edges. The graph is loaded from
    the database on init and synced back on save.
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    @property
    def graph(self) -> nx.DiGraph:
        """Direct access to the underlying NetworkX graph."""
        return self._graph

    def load_from_db(self, db: Session) -> None:
        """Populate the graph from database tables, including temporal fields."""
        self._graph.clear()

        entities = db.query(Entity).all()
        for ent in entities:
            self._graph.add_node(ent.name, type=ent.type, confidence=ent.confidence)

        relationships = db.query(Relationship).all()
        for rel in relationships:
            if rel.subject in self._graph and rel.object in self._graph:
                self._graph.add_edge(
                    rel.subject,
                    rel.object,
                    predicate=rel.predicate,
                    confidence=rel.confidence,
                    last_seen=rel.last_seen.isoformat() if rel.last_seen else None,
                    valid_from=rel.valid_from.isoformat() if rel.valid_from else None,
                    valid_to=rel.valid_to.isoformat() if rel.valid_to else None,
                )
        logger.info("Graph loaded: %d nodes, %d edges", self._graph.number_of_nodes(), self._graph.number_of_edges())

    def add_entity(self, name: str, entity_type: str, confidence: float = 0.0) -> None:
        """Add or update an entity node in the graph."""
        if self._graph.has_node(name):
            existing = self._graph.nodes[name]
            if confidence > existing.get("confidence", 0):
                existing["type"] = entity_type
                existing["confidence"] = confidence
        else:
            self._graph.add_node(name, type=entity_type, confidence=confidence)

    def add_relationship(self, subject: str, predicate: str, obj: str, confidence: float = 0.0, last_seen: str | None = None) -> None:
        """Add a relationship edge between two entities."""
        self.add_entity(subject, "entity", confidence)
        self.add_entity(obj, "entity", confidence)
        self._graph.add_edge(subject, obj, predicate=predicate, confidence=confidence, last_seen=last_seen)

    def search_entity(self, name: str) -> list[dict]:
        """Find nodes whose name contains the query string (case-insensitive)."""
        query_lower = name.lower()
        results = []
        for node, attrs in self._graph.nodes(data=True):
            if query_lower in node.lower():
                results.append({"name": node, "type": attrs.get("type", ""), "confidence": attrs.get("confidence", 0)})
        return results

    def get_neighbors(self, entity: str, hops: int = 1) -> dict:
        """BFS expansion from an entity node.

        Returns dict with keys: entity, relationships, connected_entities.
        """
        if not self._graph.has_node(entity):
            return {"entity": entity, "relationships": [], "connected_entities": []}

        visited = set()
        relationships = []
        connected = set()

        current_level = {entity}
        for _ in range(hops):
            next_level = set()
            for node in current_level:
                if node in visited:
                    continue
                visited.add(node)

                for _, neighbor, data in self._graph.out_edges(node, data=True):
                    predicate = data.get("predicate", "related_to")
                    relationships.append({
                        "subject": node,
                        "predicate": predicate,
                        "object": neighbor,
                    })
                    connected.add(neighbor)
                    next_level.add(neighbor)

                for neighbor, _, data in self._graph.in_edges(node, data=True):
                    predicate = data.get("predicate", "related_to")
                    relationships.append({
                        "subject": neighbor,
                        "predicate": predicate,
                        "object": node,
                    })
                    connected.add(neighbor)
                    next_level.add(neighbor)

            current_level = next_level - visited

        connected_entities = [
            {"name": n, "type": self._graph.nodes[n].get("type", "")}
            for n in connected
            if self._graph.has_node(n)
        ]

        return {
            "entity": entity,
            "relationships": relationships,
            "connected_entities": connected_entities,
        }

    def find_related(self, query: str) -> list[dict]:
        """Search for entities and relationships matching query keywords."""
        query_lower = query.lower()
        keywords = query_lower.split()

        matched_entities = set()
        for node, attrs in self._graph.nodes(data=True):
            if any(kw in node.lower() for kw in keywords):
                matched_entities.add(node)

        matched_edges = []
        for u, v, data in self._graph.edges(data=True):
            predicate = data.get("predicate", "")
            if any(kw in predicate.lower() for kw in keywords):
                matched_edges.append({"subject": u, "predicate": predicate, "object": v})
                matched_entities.add(u)
                matched_entities.add(v)

        return [
            {"name": n, "type": self._graph.nodes[n].get("type", "")}
            for n in matched_entities
            if self._graph.has_node(n)
        ]

    def get_facts_for_entities(self, entity_names: list[str]) -> list[dict]:
        """Retrieve all relationships involving the given entities."""
        results = []
        for u, v, data in self._graph.edges(data=True):
            if u in entity_names or v in entity_names:
                results.append({
                    "subject": u,
                    "predicate": data.get("predicate", "related_to"),
                    "object": v,
                })
        return results

    def save(self, db: Session) -> None:
        """Persist the current graph state to the database, including deletions."""
        graph_nodes = set()
        for node, attrs in self._graph.nodes(data=True):
            graph_nodes.add(node)
            existing = db.query(Entity).filter_by(name=node).first()
            if existing:
                existing.type = attrs.get("type", "entity")
                existing.confidence = attrs.get("confidence", 0)
            else:
                db.add(Entity(
                    name=node,
                    type=attrs.get("type", "entity"),
                    confidence=attrs.get("confidence", 0),
                ))

        graph_edges = set()
        for u, v, data in self._graph.edges(data=True):
            predicate = data.get("predicate", "related_to")
            edge_key = (u, predicate, v)
            graph_edges.add(edge_key)
            existing = db.query(Relationship).filter_by(
                subject=u, predicate=predicate, object=v,
            ).first()
            if not existing:
                db.add(Relationship(
                    subject=u,
                    predicate=predicate,
                    object=v,
                    confidence=data.get("confidence", 0),
                ))

        all_entities = db.query(Entity).all()
        for ent in all_entities:
            if ent.name not in graph_nodes:
                db.delete(ent)

        all_rels = db.query(Relationship).filter(Relationship.is_owner == 0).all()
        for rel in all_rels:
            edge_key = (rel.subject, rel.predicate, rel.object)
            if edge_key not in graph_edges:
                db.delete(rel)

        db.commit()

    def to_context(self, entities: list[dict], relationships: list[dict], facts: list[dict]) -> str:
        """Format graph data as context text for the LLM prompt."""
        lines = []
        if entities:
            lines.append("Known entities:")
            for e in entities:
                conf = e.get("confidence", 0)
                lines.append(f"  - {e['name']} (type: {e.get('type', 'unknown')}, confidence: {conf:.2f})")
        if relationships:
            lines.append("Relationships:")
            seen = set()
            for r in relationships:
                key = (r["subject"], r["predicate"], r["object"])
                if key not in seen:
                    seen.add(key)
                    lines.append(f"  - {r['subject']} {r['predicate']} {r['object']}")
        if facts:
            lines.append("Known facts:")
            seen = set()
            for f in facts:
                key = (f["subject"], f["predicate"], f["object"])
                if key not in seen:
                    seen.add(key)
                    lines.append(f"  - {f['subject']} {f['predicate']} {f['object']}")
        return "\n".join(lines) if lines else "No relevant ontology data found."
