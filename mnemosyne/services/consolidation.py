"""Memory consolidation service for improving graph quality."""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from sqlalchemy.orm import Session

from mnemosyne.database import get_session_factory
from mnemosyne.models import Entity, Fact, Relationship

logger = logging.getLogger(__name__)

_last_report: list[dict] = []


def _tokenize(text: str) -> set[str]:
    """Lowercase, split on non-alphanumeric, return token set."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two token sets."""
    if not a and not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union) if union else 0.0


def _containment(short: set, long: set) -> float:
    """What fraction of short tokens appear in long tokens."""
    if not short:
        return 0.0
    return len(short & long) / len(short)


class ConsolidationService:
    """Analyzes the knowledge graph and generates improvement recommendations.

    All operations are advisory — the user must approve changes.
    """

    def __init__(self) -> None:
        self._session_factory = get_session_factory()

    def analyze(self) -> dict:
        """Run all consolidation analyses and return a report."""
        global _last_report
        db = self._session_factory()
        try:
            recommendations = []
            recommendations.extend(self._find_duplicate_entities(db))
            recommendations.extend(self._find_relationship_normalization(db))
            recommendations.extend(self._find_duplicate_relationships(db))
            recommendations.extend(self._find_orphans(db))
            recommendations.extend(self._find_unsupported_relationships(db))
            recommendations.extend(self._calculate_confidence_changes(db))

            for i, rec in enumerate(recommendations):
                rec["index"] = i

            _last_report = recommendations

            summary = {
                "duplicate_entities": sum(1 for r in recommendations if r["type"] == "duplicate_entity"),
                "normalize_relationships": sum(1 for r in recommendations if r["type"] == "normalize_relationship"),
                "duplicate_relationships": sum(1 for r in recommendations if r["type"] == "duplicate_relationship"),
                "orphans": sum(1 for r in recommendations if r["type"] == "orphan"),
                "unsupported_relationships": sum(1 for r in recommendations if r["type"] == "unsupported_relationship"),
                "confidence_changes": sum(1 for r in recommendations if r["type"] == "confidence_change"),
            }

            return {"recommendations": recommendations, "summary": summary}
        finally:
            db.close()

    def apply_recommendations(self, actions: list[dict]) -> dict:
        """Apply or reject recommendations by index."""
        global _last_report
        db = self._session_factory()
        applied = 0
        rejected = 0
        errors = []

        try:
            for action in actions:
                idx = action.get("index", -1)
                act = action.get("action", "")

                if idx < 0 or idx >= len(_last_report):
                    errors.append(f"Index {idx}: not found in last report")
                    continue

                rec = _last_report[idx]

                try:
                    if act == "approve":
                        self._apply_single(db, rec)
                        applied += 1
                        logger.info("Applied recommendation %d: %s", idx, rec["type"])
                    elif act == "reject":
                        rejected += 1
                except Exception as e:
                    error_msg = f"{idx} ({rec['type']}): {str(e)[:200]}"
                    errors.append(error_msg)
                    logger.error("Failed to apply recommendation: %s", error_msg)

            db.commit()
            return {"applied": applied, "rejected": rejected, "errors": errors}
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _apply_single(self, db: Session, rec: dict) -> None:
        """Apply a single recommendation based on its type and proposed_action."""
        rec_type = rec["type"]
        action = rec["proposed_action"]

        if rec_type == "duplicate_entity":
            self._apply_merge_entities(db, action["keep"], action["remove"])

        elif rec_type == "normalize_relationship":
            self._apply_normalize_predicate(db, action["synonym"], action["canonical"])

        elif rec_type == "duplicate_relationship":
            self._apply_merge_relationships(db, rec["affected_nodes"], rec)

        elif rec_type == "orphan":
            self._apply_delete_entity(db, action["entity"])

        elif rec_type == "unsupported_relationship":
            self._apply_delete_relationship(db, action["subject"], action["predicate"], action["object"])

        elif rec_type == "confidence_change":
            self._apply_confidence_change(db, action["entity"] if "entity" in action else rec["affected_nodes"][0], action["new"])

    def _apply_merge_entities(self, db: Session, keep_name: str, remove_name: str) -> None:
        """Merge two entities: update relationships, delete the removed entity."""
        rels_to_update = (
            db.query(Relationship)
            .filter((Relationship.subject == remove_name) | (Relationship.object == remove_name))
            .all()
        )
        for rel in rels_to_update:
            if rel.subject == remove_name:
                rel.subject = keep_name
            if rel.object == remove_name:
                rel.object = keep_name

        facts_to_update = (
            db.query(Fact)
            .filter((Fact.subject == remove_name) | (Fact.object == remove_name))
            .all()
        )
        for fact in facts_to_update:
            if fact.subject == remove_name:
                fact.subject = keep_name
            if fact.object == remove_name:
                fact.object = keep_name

        entity = db.query(Entity).filter_by(name=remove_name).first()
        if entity:
            db.delete(entity)
            logger.info("Deleted entity '%s' (merged into '%s')", remove_name, keep_name)

    def _apply_normalize_predicate(self, db: Session, old_pred: str, new_pred: str) -> None:
        """Normalize all instances of a predicate to the canonical form."""
        rels = db.query(Relationship).filter_by(predicate=old_pred).all()
        for rel in rels:
            rel.predicate = new_pred

        facts = db.query(Fact).filter_by(predicate=old_pred).all()
        for fact in facts:
            fact.predicate = new_pred

        logger.info("Normalized predicate '%s' -> '%s' (%d rels, %d facts)", old_pred, new_pred, len(rels), len(facts))

    def _apply_merge_relationships(self, db: Session, affected_nodes: list[str], rec: dict) -> None:
        """Merge duplicate relationships, keeping highest confidence."""
        action = rec["proposed_action"]
        if len(affected_nodes) < 2:
            return
        subject, obj = affected_nodes[0], affected_nodes[1]
        predicate = rec.get("reason", "").split('"')[1] if '"' in rec.get("reason", "") else ""

        rels = (
            db.query(Relationship)
            .filter_by(subject=subject, object=obj)
            .order_by(Relationship.confidence.desc())
            .all()
        )

        seen_predicates = set()
        for rel in rels:
            if rel.predicate in seen_predicates:
                db.delete(rel)
            else:
                seen_predicates.add(rel.predicate)

        logger.info("Merged duplicate relationships for %s -> %s", subject, obj)

    def _apply_delete_entity(self, db: Session, entity_name: str) -> None:
        """Delete an orphan entity."""
        entity = db.query(Entity).filter_by(name=entity_name).first()
        if entity:
            db.delete(entity)
            logger.info("Deleted orphan entity '%s'", entity_name)

    def _apply_confidence_change(self, db: Session, entity_name: str, new_confidence: float) -> None:
        """Update entity confidence."""
        entity = db.query(Entity).filter_by(name=entity_name).first()
        if entity:
            old = entity.confidence
            entity.confidence = new_confidence
            logger.info("Updated confidence for '%s': %.2f -> %.2f", entity_name, old, new_confidence)

    def _find_duplicate_entities(self, db: Session) -> list[dict]:
        """Find entity pairs with similar names."""
        entities = db.query(Entity).all()
        recommendations = []
        seen = set()

        for i, e1 in enumerate(entities):
            tokens1 = _tokenize(e1.name)
            for e2 in entities[i + 1 :]:
                pair_key = tuple(sorted([e1.name, e2.name]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                tokens2 = _tokenize(e2.name)
                sim = _jaccard(tokens1, tokens2)
                contain = _containment(tokens1, tokens2) if len(tokens1) <= len(tokens2) else _containment(tokens2, tokens1)

                if sim > 0.6 or contain > 0.8:
                    keep = e1 if e1.confidence >= e2.confidence else e2
                    remove = e2 if keep == e1 else e1

                    facts1 = db.query(Fact).filter(
                        (Fact.subject == e1.name) | (Fact.object == e1.name)
                    ).count()
                    facts2 = db.query(Fact).filter(
                        (Fact.subject == e2.name) | (Fact.object == e2.name)
                    ).count()

                    recommendations.append({
                        "type": "duplicate_entity",
                        "reason": f'"{e1.name}" and "{e2.name}" appear to be the same entity (similarity: {sim:.2f})',
                        "confidence": round(max(sim, contain), 3),
                        "evidence": [f"Entity 1: {e1.name} (type={e1.type}, facts={facts1})", f"Entity 2: {e2.name} (type={e2.type}, facts={facts2})"],
                        "affected_nodes": [e1.name, e2.name],
                        "proposed_action": {"keep": keep.name, "remove": remove.name},
                        "status": "pending",
                    })

        return recommendations

    def _find_relationship_normalization(self, db: Session) -> list[dict]:
        """Find synonymous predicates that should be normalized."""
        predicates = db.query(Relationship.predicate).distinct().all()
        pred_list = [p[0] for p in predicates]
        recommendations = []
        seen = set()

        for i, p1 in enumerate(pred_list):
            tokens1 = _tokenize(p1)
            for p2 in pred_list[i + 1 :]:
                pair_key = tuple(sorted([p1, p2]))
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                tokens2 = _tokenize(p2)
                sim = _jaccard(tokens1, tokens2)

                if sim > 0.5:
                    canonical = p1 if len(p1) <= len(p2) else p2
                    synonym = p2 if canonical == p1 else p1

                    count1 = db.query(Relationship).filter_by(predicate=p1).count()
                    count2 = db.query(Relationship).filter_by(predicate=p2).count()

                    recommendations.append({
                        "type": "normalize_relationship",
                        "reason": f'"{synonym}" and "{canonical}" are similar predicates (similarity: {sim:.2f})',
                        "confidence": round(sim, 3),
                        "evidence": [f'"{p1}" used {count1} times', f'"{p2}" used {count2} times'],
                        "affected_nodes": [p1, p2],
                        "proposed_action": {"canonical": canonical, "synonym": synonym},
                        "status": "pending",
                    })

        return recommendations

    def _find_duplicate_relationships(self, db: Session) -> list[dict]:
        """Find exact duplicate edges (same subject+predicate+object)."""
        from sqlalchemy import func

        dups = (
            db.query(
                Relationship.subject,
                Relationship.predicate,
                Relationship.object,
                func.count(Relationship.id).label("cnt"),
            )
            .group_by(Relationship.subject, Relationship.predicate, Relationship.object)
            .having(func.count(Relationship.id) > 1)
            .all()
        )

        recommendations = []
        for d in dups:
            rels = (
                db.query(Relationship)
                .filter_by(subject=d[0], predicate=d[1], object=d[2])
                .order_by(Relationship.confidence.desc())
                .all()
            )
            max_conf = rels[0].confidence if rels else 0
            recommendations.append({
                "type": "duplicate_relationship",
                "reason": f'"{d[0]} {d[1]} {d[2]}" appears {d[3]} times',
                "confidence": round(max_conf, 3),
                "evidence": [f"Found {d[3]} duplicate edges, max confidence: {max_conf:.2f}"],
                "affected_nodes": [d[0], d[2]],
                "proposed_action": {"keep_confidence": max_conf, "remove_count": d[3] - 1},
                "status": "pending",
            })

        return recommendations

    def _find_orphans(self, db: Session) -> list[dict]:
        """Find entities with no relationships or low confidence."""
        entities = db.query(Entity).all()
        recommendations = []

        for ent in entities:
            rel_count = (
                db.query(Relationship)
                .filter(
                    (Relationship.subject == ent.name) | (Relationship.object == ent.name)
                )
                .count()
            )

            if rel_count == 0:
                recommendations.append({
                    "type": "orphan",
                    "reason": f'"{ent.name}" has no relationships',
                    "confidence": round(ent.confidence, 3),
                    "evidence": [f"Type: {ent.type}, confidence: {ent.confidence:.2f}, relationships: 0"],
                    "affected_nodes": [ent.name],
                    "proposed_action": {"action": "delete", "entity": ent.name},
                    "status": "pending",
                })

        return recommendations

    def _calculate_confidence_changes(self, db: Session) -> list[dict]:
        """Suggest confidence updates based on evidence count."""
        entities = db.query(Entity).all()
        recommendations = []

        for ent in entities:
            fact_count = (
                db.query(Fact)
                .filter((Fact.subject == ent.name) | (Fact.object == ent.name))
                .count()
            )
            rel_count = (
                db.query(Relationship)
                .filter(
                    (Relationship.subject == ent.name) | (Relationship.object == ent.name)
                )
                .count()
            )

            evidence_score = fact_count + rel_count
            suggested = min(1.0, 0.5 + (evidence_score * 0.05))

            if abs(suggested - ent.confidence) > 0.05 and evidence_score > 0:
                recommendations.append({
                    "type": "confidence_change",
                    "reason": f'"{ent.name}" confidence should be {suggested:.2f} (currently {ent.confidence:.2f}, {evidence_score} evidence items)',
                    "confidence": round(abs(suggested - ent.confidence), 3),
                    "evidence": [f"Facts: {fact_count}, Relationships: {rel_count}"],
                    "affected_nodes": [ent.name],
                    "proposed_action": {"old": ent.confidence, "new": round(suggested, 3), "entity": ent.name},
                    "status": "pending",
                })

        return recommendations

    def _find_unsupported_relationships(self, db: Session) -> list[dict]:
        """Find relationships with no supporting facts."""
        relationships = db.query(Relationship).all()
        recommendations = []

        for rel in relationships:
            fact_count = (
                db.query(Fact)
                .filter(
                    Fact.subject == rel.subject,
                    Fact.predicate == rel.predicate,
                    Fact.object == rel.object,
                )
                .count()
            )

            if fact_count == 0:
                recommendations.append({
                    "type": "unsupported_relationship",
                    "reason": f'"{rel.subject} {rel.predicate} {rel.object}" has no supporting facts',
                    "confidence": round(rel.confidence, 3),
                    "evidence": [
                        f"Relationship confidence: {rel.confidence:.2f}",
                        f"Supporting facts: 0",
                    ],
                    "affected_nodes": [rel.subject, rel.object],
                    "proposed_action": {
                        "subject": rel.subject,
                        "predicate": rel.predicate,
                        "object": rel.object,
                    },
                    "status": "pending",
                })

        return recommendations

    def _apply_delete_relationship(self, db: Session, subject: str, predicate: str, obj: str) -> None:
        """Delete a relationship with no supporting evidence."""
        rel = (
            db.query(Relationship)
            .filter_by(subject=subject, predicate=predicate, object=obj)
            .first()
        )
        if rel:
            db.delete(rel)
            logger.info("Deleted unsupported relationship: %s %s %s", subject, predicate, obj)
