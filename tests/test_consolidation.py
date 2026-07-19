"""Tests for the consolidation service."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mnemosyne.models import Base, Entity, Fact, Relationship
from mnemosyne.services.consolidation import ConsolidationService, _tokenize, _jaccard, _containment


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def consolidation_service(db_session, monkeypatch):
    """Create a ConsolidationService with a mocked session factory."""
    svc = ConsolidationService()
    svc._session_factory = lambda: db_session
    return svc


class TestTokenizeAndSimilarity:
    """Test text tokenization and similarity functions."""

    def test_tokenize(self):
        assert _tokenize("Hello World") == {"hello", "world"}

    def test_tokenize_special_chars(self):
        assert _tokenize("works_for") == {"works", "for"}

    def test_jaccard_identical(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_jaccard_partial(self):
        sim = _jaccard({"a", "b"}, {"b", "c"})
        assert sim == pytest.approx(1 / 3)

    def test_containment(self):
        assert _containment({"a"}, {"a", "b"}) == 1.0

    def test_containment_partial(self):
        assert _containment({"a", "b"}, {"a", "c", "d"}) == pytest.approx(0.5)


class TestDuplicateEntityDetection:
    """Test duplicate entity detection."""

    def test_finds_duplicates(self, consolidation_service, db_session):
        db_session.add(Entity(name="GPT-4", type="concept", confidence=0.9))
        db_session.add(Entity(name="GPT 4", type="concept", confidence=0.95))
        db_session.commit()

        report = consolidation_service.analyze()
        dup_recs = [r for r in report["recommendations"] if r["type"] == "duplicate_entity"]
        assert len(dup_recs) >= 1
        assert any("GPT" in r["reason"] for r in dup_recs)

    def test_no_duplicates(self, consolidation_service, db_session):
        db_session.add(Entity(name="Max", type="animal", confidence=0.9))
        db_session.add(Entity(name="Boeing", type="organization", confidence=0.9))
        db_session.commit()

        report = consolidation_service.analyze()
        dup_recs = [r for r in report["recommendations"] if r["type"] == "duplicate_entity"]
        assert len(dup_recs) == 0


class TestOrphanDetection:
    """Test orphan entity detection."""

    def test_finds_orphans(self, consolidation_service, db_session):
        db_session.add(Entity(name="Orphan", type="entity", confidence=0.5))
        db_session.commit()

        report = consolidation_service.analyze()
        orphan_recs = [r for r in report["recommendations"] if r["type"] == "orphan"]
        assert len(orphan_recs) == 1
        assert orphan_recs[0]["affected_nodes"] == ["Orphan"]

    def test_no_orphans_with_relationships(self, consolidation_service, db_session):
        db_session.add(Entity(name="A", type="person", confidence=0.9))
        db_session.add(Entity(name="B", type="person", confidence=0.9))
        db_session.add(Relationship(subject="A", predicate="knows", object="B", confidence=0.9))
        db_session.commit()

        report = consolidation_service.analyze()
        orphan_recs = [r for r in report["recommendations"] if r["type"] == "orphan"]
        assert len(orphan_recs) == 0


class TestDuplicateRelationshipDetection:
    """Test duplicate relationship detection."""

    def test_finds_duplicates(self, consolidation_service, db_session):
        db_session.add(Relationship(subject="A", predicate="knows", object="B", confidence=0.9))
        db_session.add(Relationship(subject="A", predicate="knows", object="B", confidence=0.8))
        db_session.commit()

        report = consolidation_service.analyze()
        dup_recs = [r for r in report["recommendations"] if r["type"] == "duplicate_relationship"]
        assert len(dup_recs) == 1
        assert dup_recs[0]["proposed_action"]["remove_count"] == 1


class TestConfidenceRecalculation:
    """Test confidence recalculation suggestions."""

    def test_suggests_update(self, consolidation_service, db_session):
        db_session.add(Entity(name="A", type="person", confidence=0.5))
        db_session.add(Fact(subject="A", predicate="is_a", object="person"))
        db_session.add(Fact(subject="A", predicate="works_at", object="Corp"))
        db_session.commit()

        report = consolidation_service.analyze()
        conf_recs = [r for r in report["recommendations"] if r["type"] == "confidence_change"]
        assert len(conf_recs) >= 1
        assert conf_recs[0]["proposed_action"]["new"] > 0.5


class TestApplyRecommendations:
    """Test applying consolidation actions."""

    def test_apply_merge_entities(self, consolidation_service, db_session):
        db_session.add(Entity(name="GPT-4", type="concept", confidence=0.9))
        db_session.add(Entity(name="GPT 4", type="concept", confidence=0.95))
        db_session.add(Relationship(subject="GPT 4", predicate="is_a", object="concept", confidence=0.9))
        db_session.commit()

        report = consolidation_service.analyze()
        idx = next((i for i, r in enumerate(report["recommendations"]) if r["type"] == "duplicate_entity"), None)
        assert idx is not None, f"No duplicate_entity found in: {[r['type'] for r in report['recommendations']]}"

        actions = [{"index": idx, "action": "approve"}]
        result = consolidation_service.apply_recommendations(actions)
        assert result["applied"] == 1

        remaining = db_session.query(Entity).all()
        names = {e.name for e in remaining}
        assert "GPT 4" in names
        assert "GPT-4" not in names

    def test_apply_normalize_predicate(self, consolidation_service, db_session):
        db_session.add(Relationship(subject="A", predicate="works for", object="B", confidence=0.9))
        db_session.add(Relationship(subject="C", predicate="works_for", object="D", confidence=0.9))
        db_session.commit()

        report = consolidation_service.analyze()
        idx = next((i for i, r in enumerate(report["recommendations"]) if r["type"] == "normalize_relationship"), None)
        assert idx is not None, f"No normalize_relationship found in: {[r['type'] for r in report['recommendations']]}"

        actions = [{"index": idx, "action": "approve"}]
        consolidation_service.apply_recommendations(actions)

        rels = db_session.query(Relationship).all()
        predicates = {r.predicate for r in rels}
        assert len(predicates) == 1
