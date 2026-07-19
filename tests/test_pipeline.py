"""Tests for the deterministic retrieval pipeline."""

from __future__ import annotations

import pytest
import networkx as nx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mnemosyne.models import Base, Entity, Relationship, Fact, Message, Embedding
from mnemosyne.retrieval.planner import QueryPlanner, QueryPlan
from mnemosyne.retrieval.resolver import EntityResolver, ResolvedEntity, _levenshtein, _jaccard, _tokenize
from mnemosyne.retrieval.graph_retriever import GraphRetriever, GraphResult
from mnemosyne.retrieval.ranker import Ranker, ScoredItem
from mnemosyne.retrieval.deduplicator import Deduplicator
from mnemosyne.retrieval.compressor import Compressor
from mnemosyne.retrieval.builder import ContextBuilder


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def sample_graph():
    g = nx.DiGraph()
    g.add_node("Max", type="animal", confidence=0.95)
    g.add_node("Dog", type="entity", confidence=0.9)
    g.add_node("Alice", type="person", confidence=0.9)
    g.add_node("Ethiopian Airlines", type="organization", confidence=0.99)
    g.add_node("Boeing", type="organization", confidence=0.99)
    g.add_edge("Alice", "Max", predicate="owns", confidence=0.9)
    g.add_edge("Max", "Dog", predicate="is_a", confidence=0.95)
    g.add_edge("Alice", "Ethiopian Airlines", predicate="works_for", confidence=0.85)
    g.add_edge("Ethiopian Airlines", "Boeing", predicate="operates", confidence=0.9)
    return g


class TestQueryPlanner:
    def test_entity_lookup(self):
        planner = QueryPlanner(known_entities=["max", "alice"])
        plan = planner.analyze("Tell me about Max")
        assert plan.query_type == "entity_lookup"
        assert "max" in plan.detected_entities

    def test_relationship_query(self):
        planner = QueryPlanner(known_entities=["alice"])
        plan = planner.analyze("Where does Alice work?")
        assert plan.query_type == "relationship_query"

    def test_conversation_query(self):
        planner = QueryPlanner()
        plan = planner.analyze("What did we discuss last time?")
        assert plan.query_type == "conversation"
        assert plan.vector_enabled is True
        assert plan.graph_enabled is False

    def test_semantic_query(self):
        planner = QueryPlanner()
        plan = planner.analyze("Tell me something about dogs")
        assert plan.query_type == "semantic_query"

    def test_stopwords_removed(self):
        planner = QueryPlanner()
        plan = planner.analyze("What is the name of my dog?")
        assert "the" not in plan.keywords
        assert "is" not in plan.keywords

    def test_direct_answer(self):
        planner = QueryPlanner(known_entities=["max"])
        plan = planner.analyze("Who is Max?")
        assert plan.direct_answer_possible is True


class TestEntityResolver:
    def test_exact_match(self, db_session):
        db_session.add(Entity(name="Max", type="animal", confidence=0.95))
        db_session.commit()
        resolver = EntityResolver(db_session)
        results = resolver.resolve("Max")
        assert len(results) >= 1
        assert results[0].name == "Max"
        assert results[0].match_method == "exact"

    def test_substring_match(self, db_session):
        db_session.add(Entity(name="Ethiopian Airlines", type="organization", confidence=0.9))
        db_session.commit()
        resolver = EntityResolver(db_session)
        results = resolver.resolve("Ethiopian")
        assert len(results) >= 1
        assert "Ethiopian" in results[0].name

    def test_no_match(self, db_session):
        db_session.add(Entity(name="Max", type="animal", confidence=0.95))
        db_session.commit()
        resolver = EntityResolver(db_session)
        results = resolver.resolve("xyz123")
        assert len(results) == 0


class TestLevenshtein:
    def test_identical(self):
        assert _levenshtein("cat", "cat") == 0

    def test_one_edit(self):
        assert _levenshtein("cat", "bat") == 1

    def test_empty(self):
        assert _levenshtein("", "abc") == 3


class TestJaccard:
    def test_identical(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint(self):
        assert _jaccard({"a"}, {"b"}) == 0.0


class TestGraphRetriever:
    def test_direct_facts(self, sample_graph):
        retriever = GraphRetriever(sample_graph)
        facts = retriever.get_direct_facts("Alice")
        assert len(facts) == 2
        subjects = {f["subject"] for f in facts}
        assert "Alice" in subjects

    def test_bfs_traversal(self, sample_graph):
        retriever = GraphRetriever(sample_graph)
        resolved = [ResolvedEntity(id=1, name="Alice", type="person", confidence=0.9, match_method="exact", match_score=1.0)]
        result = retriever.retrieve(resolved, max_hops=2)
        assert len(result.relationships) >= 2
        entity_names = {e["name"] for e in result.entities}
        assert "Alice" in entity_names

    def test_max_hops(self, sample_graph):
        retriever = GraphRetriever(sample_graph)
        resolved = [ResolvedEntity(id=1, name="Alice", type="person", confidence=0.9, match_method="exact", match_score=1.0)]
        result_1 = retriever.retrieve(resolved, max_hops=1)
        result_2 = retriever.retrieve(resolved, max_hops=2)
        assert len(result_2.relationships) >= len(result_1.relationships)


class TestRanker:
    def test_rank_entities(self):
        ranker = Ranker()
        entities = [
            {"name": "Max", "type": "animal", "confidence": 0.95},
            {"name": "Dog", "type": "entity", "confidence": 0.9},
        ]
        scored = ranker.rank_entities(entities, ["max"])
        assert len(scored) == 2
        assert scored[0].item_type == "entity"
        assert scored[0].score >= scored[1].score

    def test_rank_relationships(self):
        ranker = Ranker()
        rels = [
            {"subject": "Alice", "predicate": "owns", "object": "Max", "confidence": 0.9, "distance": 1, "score": 0.9},
        ]
        scored = ranker.rank_relationships(rels, ["alice"])
        assert len(scored) == 1
        assert scored[0].signals["entity_overlap"] > 0


class TestDeduplicator:
    def test_removes_duplicates(self):
        dedup = Deduplicator()
        items = [
            ScoredItem(item={"name": "Max", "type": "animal"}, item_type="entity", score=0.9),
            ScoredItem(item={"name": "max", "type": "animal"}, item_type="entity", score=0.8),
            ScoredItem(item={"name": "Alice", "type": "person"}, item_type="entity", score=0.7),
        ]
        result = dedup.dedup(items)
        assert len(result) == 2

    def test_keeps_highest_score(self):
        dedup = Deduplicator()
        items = [
            ScoredItem(item={"subject": "A", "predicate": "owns", "object": "B"}, item_type="relationship", score=0.5),
            ScoredItem(item={"subject": "a", "predicate": "owns", "object": "b"}, item_type="relationship", score=0.9),
        ]
        result = dedup.dedup(items)
        assert len(result) == 1
        assert result[0].score == 0.9


class TestCompressor:
    def test_within_budget(self):
        comp = Compressor(token_budget=100)
        items = [
            ScoredItem(item={"name": "Max", "type": "animal"}, item_type="entity", score=0.9),
            ScoredItem(item={"subject": "Alice", "predicate": "owns", "object": "Max"}, item_type="relationship", score=0.8),
        ]
        result = comp.compress(items)
        assert len(result) == 2

    def test_truncates_over_budget(self):
        comp = Compressor(token_budget=5)
        items = [
            ScoredItem(item={"name": "Max", "type": "animal"}, item_type="entity", score=0.9),
            ScoredItem(item={"name": "Very Long Entity Name That Takes Many Tokens", "type": "concept"}, item_type="entity", score=0.8),
            ScoredItem(item={"name": "Another", "type": "person"}, item_type="entity", score=0.7),
        ]
        result = comp.compress(items)
        assert len(result) <= len(items)


class TestContextBuilder:
    def test_builds_sections(self):
        builder = ContextBuilder()
        plan = QueryPlan(query_type="entity_lookup")
        items = [
            ScoredItem(item={"name": "Max", "type": "animal"}, item_type="entity", score=0.9),
            ScoredItem(item={"subject": "Alice", "predicate": "owns", "object": "Max"}, item_type="relationship", score=0.8),
            ScoredItem(item={"subject": "Max", "predicate": "is_a", "object": "dog"}, item_type="fact", score=0.7),
        ]
        context = builder.build(items, plan)
        assert "Entities:" in context
        assert "Relationships:" in context
        assert "Facts:" in context
        assert "Max" in context

    def test_empty_items(self):
        builder = ContextBuilder()
        plan = QueryPlan(query_type="general")
        context = builder.build([], plan)
        assert "No relevant information" in context
