"""Recall testing: measure retrieval quality for memory queries."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from mnemosyne.embeddings import EmbeddingService
from mnemosyne.graph import GraphService
from mnemosyne.retrieval.pipeline import ContextPipeline

logger = logging.getLogger(__name__)


@dataclass
class RecallTest:
    """A single recall test case."""
    query: str
    expected_entities: list[str] = field(default_factory=list)
    expected_relationships: list[list[str]] = field(default_factory=list)
    description: str = ""


@dataclass
class RecallResult:
    """Result of a single recall test."""
    query: str
    description: str
    # Entity recall
    entities_expected: int
    entities_found: int
    entities_missed: list[str]
    # Relationship recall
    relationships_expected: int
    relationships_found: int
    relationships_missed: list[list[str]]
    # Metrics
    recall_at_k: float  # 0.0 - 1.0
    context_chars: int
    latency_ms: float
    # Pipeline stats
    resolved_entities: int
    graph_items: int
    memory_items: int


@dataclass
class RecallReport:
    """Summary of all recall tests."""
    total_tests: int
    # Aggregate metrics
    avg_recall: float
    avg_latency_ms: float
    total_context_chars: int
    # Per-test results
    results: list[RecallResult] = field(default_factory=list)
    # Parameter snapshot
    params: dict = field(default_factory=dict)


# Standard recall test suite
RECALL_TESTS = [
    # Direct entity lookup
    RecallTest(
        query="Who is Alice?",
        expected_entities=["Alice"],
        description="Direct entity lookup",
    ),
    RecallTest(
        query="Tell me about Max",
        expected_entities=["Max"],
        description="Pet entity lookup",
    ),

    # Relationship traversal
    RecallTest(
        query="Where does Alice work?",
        expected_entities=["Alice", "Google"],
        description="Work relationship",
    ),
    RecallTest(
        query="Who does Alice know from college?",
        expected_entities=["Alice", "Carol"],
        description="Personal relationship",
    ),

    # Multi-hop traversal
    RecallTest(
        query="What does Alice's colleague work on?",
        expected_entities=["Bob", "search"],
        description="Multi-hop: colleague's work",
    ),
    RecallTest(
        query="Who manages Alice's team?",
        expected_entities=["David"],
        description="Multi-hop: team lead",
    ),

    # Technical knowledge
    RecallTest(
        query="What technologies does Mnemosyne use?",
        expected_entities=["Python", "SQLAlchemy", "NetworkX"],
        description="Technology stack",
    ),
    RecallTest(
        query="Tell me about the memory project",
        expected_entities=["Mnemosyne"],
        description="Project lookup",
    ),

    # Semantic/vector search
    RecallTest(
        query="What do I know about philosophy?",
        expected_entities=["existentialism", "Sartre"],
        description="Semantic: philosophy",
    ),
    RecallTest(
        query="What are my daily habits?",
        expected_entities=["coffee", "gym", "meditation"],
        description="Semantic: habits",
    ),

    # Cross-source fusion
    RecallTest(
        query="What programming languages do I use?",
        expected_entities=["Python", "Rust"],
        description="Cross-source: languages",
    ),
    RecallTest(
        query="Tell me about my family",
        expected_entities=["parents", "sister", "Emma"],
        description="Cross-source: family",
    ),

    # Factual recall
    RecallTest(
        query="When did I start at Google?",
        expected_entities=["Alice", "Google"],
        description="Temporal: start date",
    ),
    RecallTest(
        query="Where did I go to college?",
        expected_entities=["Stanford"],
        description="Fact: education",
    ),

    # Complex queries
    RecallTest(
        query="What are my goals for this year?",
        expected_entities=["Mnemosyne", "book", "Rust"],
        description="Complex: multiple goals",
    ),
    RecallTest(
        query="Describe my living situation",
        expected_entities=["San Francisco", "Max", "Luna"],
        description="Complex: living situation",
    ),
]


class RecallTester:
    """Tests recall quality for memory queries."""

    def __init__(
        self,
        embeddings: EmbeddingService,
        graph: GraphService,
    ) -> None:
        self._embeddings = embeddings
        self._graph = graph

    def run_all(
        self,
        db: Session,
        tests: list[RecallTest] | None = None,
        params: dict | None = None,
    ) -> RecallReport:
        """Run all recall tests and return a report.

        Args:
            db: Database session.
            tests: List of test cases. Uses RECALL_TESTS if None.
            params: Pipeline parameters to use.

        Returns:
            RecallReport with metrics.
        """
        tests = tests or RECALL_TESTS
        params = params or {}

        # Build pipeline with test params
        pipeline = ContextPipeline(
            self._embeddings,
            self._graph,
            max_hops=params.get("max_graph_depth", 3),
            min_confidence=params.get("min_confidence", 0.0),
            char_budget=params.get("char_budget", 8000),
        )

        report = RecallReport(
            total_tests=len(tests),
            avg_recall=0.0,
            avg_latency_ms=0.0,
            total_context_chars=0,
            params=params,
        )

        total_recall = 0.0
        total_latency = 0.0

        for test in tests:
            result = self._run_one(db, pipeline, test)
            report.results.append(result)
            total_recall += result.recall_at_k
            total_latency += result.latency_ms
            report.total_context_chars += result.context_chars

        report.avg_recall = total_recall / len(tests) if tests else 0.0
        report.avg_latency_ms = total_latency / len(tests) if tests else 0.0

        logger.info(
            "Recall test complete: avg_recall=%.2f, avg_latency=%.0fms, context=%d chars",
            report.avg_recall, report.avg_latency_ms, report.total_context_chars,
        )

        return report

    def _run_one(self, db: Session, pipeline: ContextPipeline, test: RecallTest) -> RecallResult:
        """Run a single recall test."""
        start = time.time()

        result = pipeline.run(db, test.query)

        latency_ms = (time.time() - start) * 1000

        # Extract found entities from graph result
        graph_result = result.graph_result or {}
        found_entities = {e.get("name", "").lower() for e in graph_result.get("entities", [])}
        found_rels = {
            (r.get("subject", "").lower(), r.get("predicate", "").lower(), r.get("object", "").lower())
            for r in graph_result.get("relationships", [])
        }

        # Calculate entity recall
        expected_entity_lower = [e.lower() for e in test.expected_entities]
        entities_found = sum(1 for e in expected_entity_lower if e in found_entities)
        entities_missed = [e for e in test.expected_entities if e.lower() not in found_entities]

        # Calculate relationship recall
        expected_rels_lower = [
            (r[0].lower(), r[1].lower(), r[2].lower())
            for r in test.expected_relationships
            if len(r) == 3
        ]
        relationships_found = sum(1 for r in expected_rels_lower if r in found_rels)
        relationships_missed = [
            r for r in test.expected_relationships
            if len(r) == 3 and (r[0].lower(), r[1].lower(), r[2].lower()) not in found_rels
        ]

        # Calculate recall@K
        total_expected = len(test.expected_entities) + len(test.expected_relationships)
        total_found = entities_found + relationships_found
        recall_at_k = total_found / total_expected if total_expected > 0 else 1.0

        return RecallResult(
            query=test.query,
            description=test.description,
            entities_expected=len(test.expected_entities),
            entities_found=entities_found,
            entities_missed=entities_missed,
            relationships_expected=len(test.expected_relationships),
            relationships_found=relationships_found,
            relationships_missed=relationships_missed,
            recall_at_k=recall_at_k,
            context_chars=len(result.context),
            latency_ms=latency_ms,
            resolved_entities=result.stats.get("resolved_entities", 0),
            graph_items=result.stats.get("graph_items", 0),
            memory_items=result.stats.get("memory_items", 0),
        )
