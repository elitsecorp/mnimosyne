"""Knowledge graph exploration service for Cytoscape.js visualization."""

from __future__ import annotations

import logging
from collections import defaultdict

from sqlalchemy.orm import Session

from mnemosyne.database import get_session_factory
from mnemosyne.models import Entity, Fact, Message, Relationship

logger = logging.getLogger(__name__)


class GraphExplorerService:
    """Provides graph data for interactive Cytoscape.js visualization.

    Never loads the full graph. Uses lazy expansion from seed nodes.
    """

    def __init__(self) -> None:
        self._session_factory = get_session_factory()

    def get_initial_graph(self, limit: int = 30, after_date: str | None = None, before_date: str | None = None) -> dict:
        """Return a seed graph of top entities by confidence and edges between them.

        Args:
            limit: Maximum number of entities to include.
            after_date: Only include entities with facts after this date (YYYY-MM-DD).
            before_date: Only include entities with facts before this date (YYYY-MM-DD).
        """
        db = self._session_factory()
        try:
            if after_date or before_date:
                entities = self._get_entities_by_date(db, limit, after_date, before_date)
            else:
                entities = (
                    db.query(Entity)
                    .order_by(Entity.confidence.desc())
                    .limit(limit)
                    .all()
                )
            entity_names = {e.name for e in entities}
            entity_id_map = {e.name: e.id for e in entities}

            relationships = (
                db.query(Relationship)
                .filter(
                    Relationship.subject.in_(entity_names),
                    Relationship.object.in_(entity_names),
                )
                .all()
            )

            degree = defaultdict(int)
            for rel in relationships:
                degree[rel.subject] += 1
                degree[rel.object] += 1

            nodes = [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type,
                    "confidence": e.confidence,
                    "degree": degree.get(e.name, 0),
                }
                for e in entities
            ]

            edges = [
                {
                    "id": rel.id,
                    "source": entity_id_map[rel.subject],
                    "target": entity_id_map[rel.object],
                    "predicate": rel.predicate,
                    "confidence": rel.confidence,
                }
                for rel in relationships
            ]

            return {"nodes": nodes, "edges": edges}
        finally:
            db.close()

    def _get_entities_by_date(self, db: Session, limit: int, after_date: str | None, before_date: str | None) -> list:
        """Get entities that have facts within the specified date range."""
        from datetime import datetime

        fact_query = db.query(Fact.subject).distinct()

        if after_date:
            try:
                after_dt = datetime.fromisoformat(after_date)
                fact_query = fact_query.filter(Fact.timestamp >= after_dt)
            except ValueError:
                pass

        if before_date:
            try:
                before_dt = datetime.fromisoformat(before_date)
                fact_query = fact_query.filter(Fact.timestamp <= before_dt)
            except ValueError:
                pass

        entity_names = {row[0] for row in fact_query.all()}

        if not entity_names:
            return []

        return (
            db.query(Entity)
            .filter(Entity.name.in_(entity_names))
            .order_by(Entity.confidence.desc())
            .limit(limit)
            .all()
        )

    def search_entities(self, query: str, limit: int = 20) -> list[dict]:
        """Search entities by name substring (case-insensitive)."""
        db = self._session_factory()
        try:
            rows = (
                db.query(Entity)
                .filter(Entity.name.ilike(f"%{query}%"))
                .order_by(Entity.confidence.desc())
                .limit(limit)
                .all()
            )
            return [
                {"id": r.id, "name": r.name, "type": r.type, "confidence": r.confidence}
                for r in rows
            ]
        finally:
            db.close()

    def get_node_detail(self, entity_id: int) -> dict | None:
        """Get full detail for a single entity: relationships + evidence messages."""
        db = self._session_factory()
        try:
            entity = db.get(Entity, entity_id)
            if not entity:
                return None

            relationships = (
                db.query(Relationship)
                .filter(
                    (Relationship.subject == entity.name)
                    | (Relationship.object == entity.name)
                )
                .all()
            )

            facts = (
                db.query(Fact)
                .filter(
                    (Fact.subject == entity.name) | (Fact.object == entity.name)
                )
                .all()
            )

            evidence_ids = set()
            for fact in facts:
                if fact.source_message:
                    msg = (
                        db.query(Message)
                        .filter(Message.content == fact.source_message)
                        .first()
                    )
                    if msg:
                        evidence_ids.add(msg.id)

            if not evidence_ids:
                entity_msgs = (
                    db.query(Message)
                    .filter(Message.content.ilike(f"%{entity.name}%"))
                    .order_by(Message.timestamp.desc())
                    .limit(20)
                    .all()
                )
                for msg in entity_msgs:
                    evidence_ids.add(msg.id)

            evidence = []
            for mid in evidence_ids:
                msg = db.get(Message, mid)
                if msg:
                    evidence.append({
                        "id": msg.id,
                        "content": msg.content,
                        "timestamp": str(msg.timestamp),
                    })

            return {
                "id": entity.id,
                "name": entity.name,
                "type": entity.type,
                "confidence": entity.confidence,
                "relationships": [
                    {
                        "id": r.id,
                        "subject": r.subject,
                        "predicate": r.predicate,
                        "object": r.object,
                        "confidence": r.confidence,
                    }
                    for r in relationships
                ],
                "evidence": evidence,
            }
        finally:
            db.close()

    def get_edge_detail(self, relationship_id: int) -> dict | None:
        """Get full detail for a single edge: predicate + evidence."""
        db = self._session_factory()
        try:
            rel = db.get(Relationship, relationship_id)
            if not rel:
                return None

            facts = (
                db.query(Fact)
                .filter(
                    Fact.subject == rel.subject,
                    Fact.predicate == rel.predicate,
                    Fact.object == rel.object,
                )
                .all()
            )

            evidence = []
            for fact in facts[:10]:
                if fact.source_message:
                    msg = (
                        db.query(Message)
                        .filter(Message.content == fact.source_message)
                        .first()
                    )
                    if msg:
                        evidence.append({
                            "id": msg.id,
                            "content": msg.content,
                            "timestamp": str(msg.timestamp),
                        })

            return {
                "id": rel.id,
                "predicate": rel.predicate,
                "confidence": rel.confidence,
                "subject": rel.subject,
                "object": rel.object,
                "evidence": evidence,
            }
        finally:
            db.close()

    def get_neighbors(self, entity_id: int, hops: int = 1) -> dict:
        """Get 1-hop neighborhood for lazy expansion.

        Returns nodes and edges for Cytoscape.js, excluding already-loaded nodes.
        """
        db = self._session_factory()
        try:
            entity = db.get(Entity, entity_id)
            if not entity:
                return {"nodes": [], "edges": []}

            relationships = (
                db.query(Relationship)
                .filter(
                    (Relationship.subject == entity.name)
                    | (Relationship.object == entity.name)
                )
                .all()
            )

            neighbor_names = set()
            for rel in relationships:
                neighbor_names.add(rel.subject)
                neighbor_names.add(rel.object)
            neighbor_names.discard(entity.name)

            neighbors = (
                db.query(Entity)
                .filter(Entity.name.in_(neighbor_names))
                .all()
            )
            neighbor_id_map = {n.name: n.id for n in neighbors}

            all_names = neighbor_names | {entity.name}
            all_id_map = {**neighbor_id_map, entity.name: entity.id}

            cross_relationships = (
                db.query(Relationship)
                .filter(
                    Relationship.subject.in_(all_names),
                    Relationship.object.in_(all_names),
                )
                .all()
            )

            nodes = []
            seen = set()
            for n in neighbors:
                if n.id not in seen:
                    seen.add(n.id)
                    nodes.append({
                        "id": n.id,
                        "name": n.name,
                        "type": n.type,
                        "confidence": n.confidence,
                        "degree": 0,
                    })

            edges = []
            for rel in cross_relationships:
                src = all_id_map.get(rel.subject)
                tgt = all_id_map.get(rel.object)
                if src and tgt:
                    edges.append({
                        "id": rel.id,
                        "source": src,
                        "target": tgt,
                        "predicate": rel.predicate,
                        "confidence": rel.confidence,
                    })

            return {"nodes": nodes, "edges": edges}
        finally:
            db.close()

    def get_types(self) -> list[dict]:
        """Get all entity types with counts."""
        db = self._session_factory()
        try:
            from sqlalchemy import func

            rows = (
                db.query(Entity.type, func.count(Entity.id))
                .group_by(Entity.type)
                .order_by(func.count(Entity.id).desc())
                .all()
            )
            return [{"type": r[0], "count": r[1]} for r in rows]
        finally:
            db.close()

    def get_statistics(self) -> dict:
        """Get graph statistics."""
        db = self._session_factory()
        try:
            from sqlalchemy import func

            node_count = db.query(func.count(Entity.id)).scalar() or 0
            edge_count = db.query(func.count(Relationship.id)).scalar() or 0
            avg_conf = (
                db.query(func.avg(Entity.confidence)).scalar() or 0
            )
            types = self.get_types()

            return {
                "nodes": node_count,
                "edges": edge_count,
                "types": types,
                "avg_confidence": round(float(avg_conf), 3),
            }
        finally:
            db.close()
