"""Me Graph compiler: identifies Me, runs onboarding, and connects concepts."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from mnemosyne.database import get_session_factory
from mnemosyne.models import Entity, Fact, Relationship

logger = logging.getLogger(__name__)

_ME_NAMES = {"me", "i", "myself", "user", "owner"}
_POSSESSIVE_PATTERNS = [
    re.compile(r"\bmy\b", re.IGNORECASE),
    re.compile(r"\bi\b", re.IGNORECASE),
    re.compile(r"\bme\b", re.IGNORECASE),
]
_ME_PREDICATES = {
    "owns", "has", "likes", "dislikes", "works_for", "works_at",
    "lives_in", "located_in", "knows", "uses", "created", "built",
    "visited", "belongs_to", "has_goal", "has_project", "has_skill",
    "has_habit", "has_resource", "speaks", "learning", "interested_in",
    "believes", "has_pet", "has_friend", "attended", "is_a",
    "has_name", "has_role", "discussed", "is_interested_in",
}

ONBOARDING_QUESTIONS = [
    {"key": "has_name", "predicate": "has_name", "question": "What should I call you?", "entity_type": "person"},
    {"key": "has_role", "predicate": "has_role", "question": "What do you do?", "entity_type": "role"},
    {"key": "has_goal", "predicate": "has_goal", "question": "What's your biggest goal right now?", "entity_type": "goal"},
    {"key": "works_on", "predicate": "works_on", "question": "What project are you currently working on?", "entity_type": "project"},
    {"key": "interested_in", "predicate": "interested_in", "question": "What topics are you most interested in?", "entity_type": "topic"},
]


@dataclass
class MeConnection:
    """A connection from Me to another entity."""

    subject: str
    predicate: str
    object: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    source_rel_id: int | None = None


class OwnerCompiler:
    """Compiles the Me-centric subgraph from the global ontology.

    Identifies the Me entity, runs onboarding, and creates relationships
    connecting Me to relevant concepts.
    """

    def __init__(self) -> None:
        self._session_factory = get_session_factory()

    def get_onboarding_status(self, db: Session) -> dict:
        """Check if onboarding is needed and return status."""
        me = db.query(Entity).filter_by(name="Me").first()
        if not me:
            return {"needs_onboarding": True, "completed": 0, "total": len(ONBOARDING_QUESTIONS)}

        completed = set()
        for q in ONBOARDING_QUESTIONS:
            has = (
                db.query(Relationship)
                .filter(
                    Relationship.subject == "Me",
                    Relationship.predicate == q["predicate"],
                )
                .first()
            )
            if has:
                completed.add(q["key"])

        return {
            "needs_onboarding": len(completed) < len(ONBOARDING_QUESTIONS),
            "completed": len(completed),
            "total": len(ONBOARDING_QUESTIONS),
            "questions": [
                {
                    "key": q["key"],
                    "question": q["question"],
                    "answered": q["key"] in completed,
                }
                for q in ONBOARDING_QUESTIONS
            ],
        }

    def answer_onboarding(self, db: Session, key: str, answer: str) -> dict:
        """Process an onboarding answer."""
        question = next((q for q in ONBOARDING_QUESTIONS if q["key"] == key), None)
        if not question:
            return {"error": f"Unknown question: {key}"}

        self._ensure_me(db)

        entity = Entity(name=answer.strip(), type=question["entity_type"], confidence=0.95)
        existing = db.query(Entity).filter_by(name=answer.strip()).first()
        if not existing:
            db.add(entity)
        else:
            entity = existing

        existing_rel = (
            db.query(Relationship)
            .filter_by(subject="Me", predicate=question["predicate"], object=answer.strip())
            .first()
        )
        if not existing_rel:
            db.add(Relationship(
                subject="Me",
                predicate=question["predicate"],
                object=answer.strip(),
                confidence=0.95,
                is_owner=1,
            ))

        if key == "has_name":
            self._connect_me_to_name(db, answer.strip())

        db.commit()

        status = self.get_onboarding_status(db)
        return {
            "stored": True,
            "key": key,
            "answer": answer.strip(),
            "onboarding_complete": not status["needs_onboarding"],
            "completed": status["completed"],
            "total": status["total"],
        }

    def get_me_profile(self, db: Session) -> dict:
        """Get Me's full profile."""
        me = db.query(Entity).filter_by(name="Me").first()
        if not me:
            return {"found": False, "name": None, "profile": {}}

        profile = {}
        for q in ONBOARDING_QUESTIONS:
            rel = (
                db.query(Relationship)
                .filter_by(subject="Me", predicate=q["predicate"])
                .first()
            )
            if rel:
                profile[q["key"]] = rel.object

        return {
            "found": True,
            "name": profile.get("has_name", "Me"),
            "profile": profile,
        }

    def get_me_graph(self, db: Session) -> dict:
        """Get the Me subgraph for visualization."""
        me = db.query(Entity).filter(Entity.name == "Me").first()
        if not me:
            return {"owner": None, "nodes": [], "edges": []}

        me_rels = (
            db.query(Relationship)
            .filter(Relationship.is_owner == 1)
            .all()
        )

        node_names = {me.name}
        nodes = [{"name": me.name, "type": "person", "confidence": 1.0, "is_me": True}]
        edges = []

        for rel in me_rels:
            node_names.add(rel.object)
            edges.append({
                "subject": rel.subject,
                "predicate": rel.predicate,
                "object": rel.object,
                "confidence": rel.confidence,
            })

        for name in node_names:
            if name == me.name:
                continue
            ent = db.query(Entity).filter_by(name=name).first()
            if ent:
                nodes.append({
                    "name": ent.name,
                    "type": ent.type,
                    "confidence": ent.confidence,
                    "is_me": False,
                })

        return {"owner": me.name, "nodes": nodes, "edges": edges}

    def compile(self, db: Session) -> dict:
        """Run the full Me compilation."""
        me_name = self._ensure_me(db)
        self._attach_discussion_concepts(db, me_name)
        connections = self._find_me_connections(db, me_name)
        stored = self._store_me_connections(db, me_name, connections)
        self._cleanup_old_me_edges(db, me_name)

        return {
            "owner": me_name,
            "connections_found": len(connections),
            "connections_stored": stored,
        }

    def _ensure_me(self, db: Session) -> str:
        """Find or create the Me entity, renaming User/Owner to Me."""
        me = db.query(Entity).filter_by(name="Me").first()
        if me:
            return "Me"

        for old_name in ["User", "Owner"]:
            old = db.query(Entity).filter_by(name=old_name).first()
            if old:
                old.name = "Me"
                rels = (
                    db.query(Relationship)
                    .filter((Relationship.subject == old_name) | (Relationship.object == old_name))
                    .all()
                )
                for rel in rels:
                    if rel.subject == old_name:
                        rel.subject = "Me"
                    if rel.object == old_name:
                        rel.object = "Me"
                facts = (
                    db.query(Fact)
                    .filter((Fact.subject == old_name) | (Fact.object == old_name))
                    .all()
                )
                for fact in facts:
                    if fact.subject == old_name:
                        fact.subject = "Me"
                    if fact.object == old_name:
                        fact.object = "Me"
                db.commit()
                logger.info("Renamed %s to Me", old_name)
                return "Me"

        new_me = Entity(name="Me", type="person", confidence=1.0)
        db.add(new_me)
        db.commit()
        logger.info("Created Me entity")
        return "Me"

    def _connect_me_to_name(self, db: Session, name: str) -> None:
        """Connect Me entity to the user's name via is_same_as relationship."""
        existing = (
            db.query(Relationship)
            .filter_by(subject="Me", predicate="is_same_as", object=name)
            .first()
        )
        if not existing:
            db.add(Relationship(
                subject="Me",
                predicate="is_same_as",
                object=name,
                confidence=0.95,
                is_owner=1,
            ))
            logger.info("Connected Me to name: %s", name)

    def _attach_discussion_concepts(self, db: Session, me_name: str) -> None:
        """Attach concepts discussed in conversations to Me, capped to avoid unbounded growth."""
        existing_count = (
            db.query(Relationship)
            .filter(Relationship.subject == me_name, Relationship.predicate == "discussed", Relationship.is_owner == 1)
            .count()
        )
        if existing_count >= 50:
            return

        facts = db.query(Fact).filter(Fact.source_message.isnot(None)).all()
        attached = 0
        seen_objects = set()

        for fact in facts:
            if fact.subject.lower() in _ME_NAMES or fact.object.lower() in _ME_NAMES:
                continue
            if fact.subject in seen_objects:
                continue

            has_link = (
                db.query(Relationship)
                .filter(
                    Relationship.subject == me_name,
                    Relationship.predicate.in_(["discussed", "is_interested_in"]),
                    Relationship.object == fact.subject,
                )
                .first()
            )
            if not has_link:
                db.add(Relationship(
                    subject=me_name,
                    predicate="discussed",
                    object=fact.subject,
                    confidence=0.6,
                    is_owner=1,
                ))
                attached += 1
                seen_objects.add(fact.subject)
                if existing_count + attached >= 50:
                    break

        if attached > 0:
            db.commit()
            logger.info("Attached %d discussion concepts to Me", attached)

    def _find_me_connections(self, db: Session, me_name: str) -> list[MeConnection]:
        """Scan relationships for Me-relevant connections."""
        connections = []

        rels = db.query(Relationship).filter(Relationship.is_owner == 0).all()

        for rel in rels:
            if self._is_me_relationship(rel, me_name):
                evidence = self._get_evidence(db, rel)
                connections.append(MeConnection(
                    subject=me_name,
                    predicate=rel.predicate,
                    object=rel.object if rel.subject.lower() in _ME_NAMES else rel.subject,
                    confidence=rel.confidence,
                    evidence=evidence,
                    source_rel_id=rel.id,
                ))

        facts = db.query(Fact).filter(Fact.source_message.isnot(None)).all()
        for fact in facts:
            if self._fact_implies_me(fact):
                already = any(
                    c.predicate == fact.predicate and c.object.lower() == fact.object.lower()
                    for c in connections
                )
                if not already:
                    connections.append(MeConnection(
                        subject=me_name,
                        predicate=fact.predicate,
                        object=fact.object,
                        confidence=0.7,
                        evidence=[fact.source_message] if fact.source_message else [],
                    ))

        seen = set()
        unique = []
        for c in connections:
            key = (c.predicate.lower(), c.object.lower())
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique

    def _is_me_relationship(self, rel: Relationship, me_name: str) -> bool:
        """Check if a relationship should be connected to Me."""
        subject_lower = rel.subject.lower()

        if subject_lower in _ME_NAMES:
            return True

        if rel.predicate.lower() in _ME_PREDICATES:
            if subject_lower == me_name.lower():
                return True

        if rel.subject.lower() == me_name.lower():
            return True

        return False

    def _fact_implies_me(self, fact: Fact) -> bool:
        """Check if a fact implies a Me connection via source message."""
        if not fact.source_message:
            return False

        msg = fact.source_message.lower()

        for pattern in _POSSESSIVE_PATTERNS:
            if pattern.search(msg):
                return True

        return False

    def _get_evidence(self, db: Session, rel: Relationship) -> list[str]:
        """Get source messages that support this relationship."""
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
        for f in facts:
            if f.source_message and f.source_message not in evidence:
                evidence.append(f.source_message)

        return evidence

    def _store_me_connections(self, db: Session, me_name: str, connections: list[MeConnection]) -> int:
        """Store Me connections as marked relationships."""
        stored = 0

        for conn in connections:
            existing = (
                db.query(Relationship)
                .filter_by(
                    subject=conn.subject,
                    predicate=conn.predicate,
                    object=conn.object,
                    is_owner=1,
                )
                .first()
            )

            if existing:
                if conn.confidence > existing.confidence:
                    existing.confidence = conn.confidence
                    stored += 1
            else:
                db.add(Relationship(
                    subject=conn.subject,
                    predicate=conn.predicate,
                    object=conn.object,
                    confidence=conn.confidence,
                    is_owner=1,
                ))
                stored += 1

        db.commit()
        return stored

    def _cleanup_old_me_edges(self, db: Session, me_name: str) -> None:
        """Remove Me edges that no longer have supporting evidence."""
        me_edges = (
            db.query(Relationship)
            .filter(Relationship.is_owner == 1)
            .all()
        )

        for edge in me_edges:
            if edge.predicate == "discussed":
                continue

            has_fact = (
                db.query(Fact)
                .filter(
                    Fact.subject == edge.subject,
                    Fact.predicate == edge.predicate,
                    Fact.object == edge.object,
                )
                .first()
            )

            has_source_rel = (
                db.query(Relationship)
                .filter(
                    Relationship.subject == edge.subject,
                    Relationship.predicate == edge.predicate,
                    Relationship.object == edge.object,
                    Relationship.is_owner == 0,
                )
                .first()
            )

            if not has_fact and not has_source_rel:
                db.delete(edge)

        db.commit()
