"""Tests for the Me Compiler."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mnemosyne.models import Base, Entity, Relationship, Fact
from mnemosyne.services.owner_compiler import OwnerCompiler, MeConnection


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


class TestMeIdentification:
    def test_creates_me_if_missing(self, db_session, compiler):
        result = compiler.compile(db_session)
        assert result["owner"] == "Me"
        me = db_session.query(Entity).filter_by(name="Me").first()
        assert me is not None
        assert me.type == "person"

    def test_finds_existing_me(self, db_session, compiler):
        db_session.add(Entity(name="Me", type="person", confidence=1.0))
        db_session.commit()
        result = compiler.compile(db_session)
        assert result["owner"] == "Me"

    def test_finds_user_entity(self, db_session, compiler):
        db_session.add(Entity(name="User", type="person", confidence=0.9))
        db_session.commit()
        result = compiler.compile(db_session)
        assert result["owner"] == "Me"
        me = db_session.query(Entity).filter_by(name="Me").first()
        assert me is not None


class TestMeConnections:
    def test_connects_user_owned_entities(self, db_session, compiler):
        db_session.add(Entity(name="User", type="person", confidence=0.9))
        db_session.add(Entity(name="Max", type="animal", confidence=0.95))
        db_session.add(Relationship(subject="User", predicate="owns", object="Max", confidence=0.9))
        db_session.commit()

        result = compiler.compile(db_session)
        assert result["connections_stored"] >= 1

        me_edges = db_session.query(Relationship).filter(Relationship.is_owner == 1).all()
        assert len(me_edges) >= 1
        assert any(e.predicate == "owns" and e.object == "Max" for e in me_edges)

    def test_connects_i_statement_entities(self, db_session, compiler):
        db_session.add(Entity(name="Me", type="person", confidence=1.0))
        db_session.add(Entity(name="Boeing", type="organization", confidence=0.99))
        db_session.add(Relationship(subject="Me", predicate="works_for", object="Boeing", confidence=0.85))
        db_session.commit()

        result = compiler.compile(db_session)
        me_edges = db_session.query(Relationship).filter(Relationship.is_owner == 1).all()
        assert len(me_edges) >= 1

    def test_fact_based_me_connection(self, db_session, compiler):
        db_session.add(Entity(name="Me", type="person", confidence=1.0))
        db_session.add(Entity(name="Coffee", type="concept", confidence=0.8))
        db_session.add(Fact(subject="Me", predicate="likes", object="Coffee", source_message="I love coffee every morning"))
        db_session.commit()

        result = compiler.compile(db_session)
        me_edges = db_session.query(Relationship).filter(Relationship.is_owner == 1).all()
        assert len(me_edges) >= 1


class TestMeGraph:
    def test_get_me_graph(self, db_session, compiler):
        db_session.add(Entity(name="Me", type="person", confidence=1.0))
        db_session.add(Entity(name="Max", type="animal", confidence=0.95))
        db_session.add(Relationship(subject="Me", predicate="owns", object="Max", confidence=0.9, is_owner=1))
        db_session.commit()

        graph = compiler.get_me_graph(db_session)
        assert graph["owner"] == "Me"
        assert len(graph["nodes"]) == 2
        assert len(graph["edges"]) == 1
        assert graph["edges"][0]["predicate"] == "owns"

    def test_get_me_graph_no_me(self, db_session, compiler):
        graph = compiler.get_me_graph(db_session)
        assert graph["owner"] is None
        assert len(graph["nodes"]) == 0


class TestCleanup:
    def test_removes_unsupported_me_edges(self, db_session, compiler):
        db_session.add(Entity(name="Me", type="person", confidence=1.0))
        db_session.add(Entity(name="Phantom", type="concept", confidence=0.5))
        db_session.add(Relationship(subject="Me", predicate="likes", object="Phantom", confidence=0.7, is_owner=1))
        db_session.commit()

        compiler.compile(db_session)

        edge = db_session.query(Relationship).filter_by(subject="Me", object="Phantom").first()
        assert edge is None
