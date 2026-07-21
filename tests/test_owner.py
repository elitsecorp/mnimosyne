"""Tests for the Owner Compiler."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mnemosyne.models import Base, Entity, Relationship, Fact
from mnemosyne.services.owner_compiler import OwnerCompiler, OwnerConnection


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def compiler():
    return OwnerCompiler()


class TestOwnerIdentification:
    def test_creates_owner_if_missing(self, db_session, compiler):
        result = compiler.compile(db_session)
        assert result["owner"] == "Owner"
        owner = db_session.query(Entity).filter_by(name="Owner").first()
        assert owner is not None
        assert owner.type == "person"

    def test_finds_existing_owner(self, db_session, compiler):
        db_session.add(Entity(name="Owner", type="person", confidence=1.0))
        db_session.commit()
        result = compiler.compile(db_session)
        assert result["owner"] == "Owner"

    def test_finds_user_entity(self, db_session, compiler):
        db_session.add(Entity(name="User", type="person", confidence=0.9))
        db_session.commit()
        result = compiler.compile(db_session)
        assert result["owner"] == "Owner"
        owner = db_session.query(Entity).filter_by(name="Owner").first()
        assert owner is not None


class TestOwnerConnections:
    def test_connects_user_owned_entities(self, db_session, compiler):
        db_session.add(Entity(name="User", type="person", confidence=0.9))
        db_session.add(Entity(name="Max", type="animal", confidence=0.95))
        db_session.add(Relationship(subject="User", predicate="owns", object="Max", confidence=0.9))
        db_session.commit()

        result = compiler.compile(db_session)
        assert result["connections_stored"] >= 1

        owner_edges = db_session.query(Relationship).filter(Relationship.is_owner == 1).all()
        assert len(owner_edges) >= 1
        assert any(e.predicate == "owns" and e.object == "Max" for e in owner_edges)

    def test_connects_i_statement_entities(self, db_session, compiler):
        db_session.add(Entity(name="Owner", type="person", confidence=1.0))
        db_session.add(Entity(name="Boeing", type="organization", confidence=0.99))
        db_session.add(Relationship(subject="Owner", predicate="works_for", object="Boeing", confidence=0.85))
        db_session.commit()

        result = compiler.compile(db_session)
        owner_edges = db_session.query(Relationship).filter(Relationship.is_owner == 1).all()
        assert len(owner_edges) >= 1

    def test_fact_based_owner_connection(self, db_session, compiler):
        db_session.add(Entity(name="Owner", type="person", confidence=1.0))
        db_session.add(Entity(name="Coffee", type="concept", confidence=0.8))
        db_session.add(Fact(subject="Owner", predicate="likes", object="Coffee", source_message="I love coffee every morning"))
        db_session.commit()

        result = compiler.compile(db_session)
        owner_edges = db_session.query(Relationship).filter(Relationship.is_owner == 1).all()
        assert len(owner_edges) >= 1


class TestOwnerGraph:
    def test_get_owner_graph(self, db_session, compiler):
        db_session.add(Entity(name="Owner", type="person", confidence=1.0))
        db_session.add(Entity(name="Max", type="animal", confidence=0.95))
        db_session.add(Relationship(subject="Owner", predicate="owns", object="Max", confidence=0.9, is_owner=1))
        db_session.commit()

        graph = compiler.get_owner_graph(db_session)
        assert graph["owner"] == "Owner"
        assert len(graph["nodes"]) == 2
        assert len(graph["edges"]) == 1
        assert graph["edges"][0]["predicate"] == "owns"

    def test_get_owner_graph_no_owner(self, db_session, compiler):
        graph = compiler.get_owner_graph(db_session)
        assert graph["owner"] is None
        assert len(graph["nodes"]) == 0


class TestCleanup:
    def test_removes_unsupported_owner_edges(self, db_session, compiler):
        db_session.add(Entity(name="Owner", type="person", confidence=1.0))
        db_session.add(Entity(name="Phantom", type="concept", confidence=0.5))
        db_session.add(Relationship(subject="Owner", predicate="likes", object="Phantom", confidence=0.7, is_owner=1))
        db_session.commit()

        compiler.compile(db_session)

        edge = db_session.query(Relationship).filter_by(subject="Owner", object="Phantom").first()
        assert edge is None
