"""Tests for the memory engine pipeline."""

from __future__ import annotations

import json
import struct
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mnemosyne.config import Settings
from mnemosyne.database import Base, get_engine
from mnemosyne.embeddings import EmbeddingService, _serialize_vector, _deserialize_vector
from mnemosyne.extraction import _parse_extraction, _normalize_predicate
from mnemosyne.graph import GraphService
from mnemosyne.models import Embedding, Entity, Fact, Message, Relationship
from mnemosyne.prompts import build_chat_messages, build_extraction_messages
from mnemosyne.schemas import ExtractionResult, EntitySchema, RelationshipSchema, FactSchema


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def graph_service():
    """Create a fresh GraphService for testing."""
    return GraphService()


class TestVectorSerialization:
    """Test vector pack/unpack utilities."""

    def test_roundtrip(self):
        vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        packed = _serialize_vector(vec)
        unpacked = _deserialize_vector(packed, 5)
        assert unpacked == pytest.approx(vec)

    def test_empty_vector(self):
        packed = _serialize_vector([])
        unpacked = _deserialize_vector(packed, 0)
        assert unpacked == []


class TestExtractionParsing:
    """Test memory extraction JSON parsing."""

    def test_parse_valid_extraction(self):
        data = {
            "entities": [
                {"name": "Max", "type": "animal", "confidence": 0.95},
                {"name": "Alice", "type": "person", "confidence": 0.9},
            ],
            "relationships": [
                {"subject": "Alice", "predicate": "owns", "object": "Max", "confidence": 0.94},
            ],
            "facts": [
                {"subject": "Max", "predicate": "is_a", "object": "dog"},
            ],
        }
        result = _parse_extraction(data)
        assert len(result.entities) == 2
        assert result.entities[0].name == "Max"
        assert result.entities[0].type == "animal"
        assert len(result.relationships) == 1
        assert result.relationships[0].predicate == "owns"
        assert len(result.facts) == 1

    def test_parse_empty_extraction(self):
        result = _parse_extraction({})
        assert result.entities == []
        assert result.relationships == []
        assert result.facts == []

    def test_parse_skips_invalid(self):
        data = {
            "entities": [
                {"name": "Valid", "type": "person", "confidence": 0.9},
                {"missing": "fields"},
            ],
            "relationships": [],
            "facts": [],
        }
        result = _parse_extraction(data)
        assert len(result.entities) == 1

    def test_normalize_predicate(self):
        assert _normalize_predicate("works_for") == "works_for"
        assert _normalize_predicate("Works For") == "works_for"
        assert _normalize_predicate("works-for") == "works_for"
        assert _normalize_predicate("  OWNs  ") == "owns"


class TestGraphService:
    """Test graph operations."""

    def test_add_entity(self, graph_service):
        graph_service.add_entity("Max", "animal", 0.95)
        assert graph_service.graph.has_node("Max")
        assert graph_service.graph.nodes["Max"]["type"] == "animal"
        assert graph_service.graph.nodes["Max"]["confidence"] == 0.95

    def test_add_relationship(self, graph_service):
        graph_service.add_relationship("Alice", "owns", "Max", 0.9)
        assert graph_service.graph.has_edge("Alice", "Max")
        assert graph_service.graph.edges["Alice", "Max"]["predicate"] == "owns"

    def test_search_entity(self, graph_service):
        graph_service.add_entity("Max", "animal", 0.95)
        graph_service.add_entity("Boeing", "organization", 0.99)
        results = graph_service.search_entity("max")
        assert len(results) == 1
        assert results[0]["name"] == "Max"

    def test_get_neighbors(self, graph_service):
        graph_service.add_entity("Alice", "person", 0.9)
        graph_service.add_entity("Max", "animal", 0.95)
        graph_service.add_entity("Dog Park", "place", 0.8)
        graph_service.add_relationship("Alice", "owns", "Max", 0.9)
        graph_service.add_relationship("Alice", "visits", "Dog Park", 0.85)

        neighbors = graph_service.get_neighbors("Alice", hops=1)
        assert neighbors["entity"] == "Alice"
        assert len(neighbors["relationships"]) == 2

    def test_to_context(self, graph_service):
        entities = [{"name": "Max", "type": "animal"}]
        relationships = [{"subject": "Alice", "predicate": "owns", "object": "Max"}]
        facts = []
        context = graph_service.to_context(entities, relationships, facts)
        assert "Max" in context
        assert "owns" in context


class TestPrompts:
    """Test prompt building."""

    def test_build_chat_messages_basic(self):
        messages = build_chat_messages(
            conversation=[],
            vector_memories="",
            ontology_facts="",
            user_message="Hello",
        )
        assert len(messages) == 2  # system + user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Hello" in messages[1]["content"]

    def test_build_chat_messages_with_context(self):
        messages = build_chat_messages(
            conversation=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
            vector_memories="- Something about dogs",
            ontology_facts="Max is a dog",
            user_message="What's my dog's name?",
        )
        assert len(messages) == 4  # system + 2 conversation + user
        content = messages[-1]["content"]
        assert "dogs" in content
        assert "Max" in content

    def test_build_extraction_messages(self):
        messages = build_extraction_messages("My dog is Max", "Got it, Max is your dog!")
        assert len(messages) == 2
        assert "My dog is Max" in messages[1]["content"]


class TestDatabaseModels:
    """Test database model creation."""

    def test_create_message(self, db_session):
        msg = Message(role="user", content="Hello world")
        db_session.add(msg)
        db_session.commit()
        assert msg.id is not None

    def test_create_entity(self, db_session):
        ent = Entity(name="Max", type="animal", confidence=0.95)
        db_session.add(ent)
        db_session.commit()
        assert ent.id is not None

    def test_create_relationship(self, db_session):
        rel = Relationship(subject="Alice", predicate="owns", object="Max", confidence=0.9)
        db_session.add(rel)
        db_session.commit()
        assert rel.id is not None

    def test_create_fact(self, db_session):
        fact = Fact(subject="Max", predicate="is_a", object="dog")
        db_session.add(fact)
        db_session.commit()
        assert fact.id is not None
