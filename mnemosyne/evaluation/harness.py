"""Pipeline evaluation harness."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from mnemosyne.retrieval.pipeline import ContextPipeline, PipelineResult

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single evaluation test case."""

    query: str
    expected_entities: list[str] = field(default_factory=list)
    expected_relationships: list[list[str]] = field(default_factory=list)
    expected_facts: list[list[str]] = field(default_factory=list)
    expected_query_type: str = ""
    min_entities: int = 0
    min_relationships: int = 0
    min_facts: int = 0
    description: str = ""


@dataclass
class TestResult:
    """Result of evaluating a single test case."""

    query: str
    description: str
    passed: bool
    query_type_correct: bool
    entities_found: int
    entities_expected: int
    relationships_found: int
    relationships_expected: int
    facts_found: int
    facts_expected: int
    context_length: int
    elapsed_ms: float
    issues: list[str] = field(default_factory=list)


class EvalHarness:
    """Runs test cases through the pipeline and evaluates quality."""

    def __init__(self, pipeline: ContextPipeline, db: Session) -> None:
        self._pipeline = pipeline
        self._db = db

    def run_test(self, test: TestCase) -> TestResult:
        """Run a single test case and evaluate results."""
        result = self._pipeline.run(self._db, test.query)
        plan = result.plan

        issues = []
        query_type_correct = True
        if test.expected_query_type and plan.query_type != test.expected_query_type:
            query_type_correct = False
            issues.append(f"Query type: got '{plan.query_type}', expected '{test.expected_query_type}'")

        found_entity_names = {e.get("name", "").lower() for e in (result.graph_result or {}).get("entities", [])}
        found_rel_keys = {
            (r.get("subject", "").lower(), r.get("predicate", "").lower(), r.get("object", "").lower())
            for r in (result.graph_result or {}).get("relationships", [])
        }
        found_fact_keys = {
            (f.get("subject", "").lower(), f.get("predicate", "").lower(), f.get("object", "").lower())
            for f in (result.graph_result or {}).get("facts", [])
        }

        entities_found = sum(1 for e in test.expected_entities if e.lower() in found_entity_names)
        relationships_found = sum(
            1 for r in test.expected_relationships
            if len(r) == 3 and (r[0].lower(), r[1].lower(), r[2].lower()) in found_rel_keys
        )
        facts_found = sum(
            1 for f in test.expected_facts
            if len(f) == 3 and (f[0].lower(), f[1].lower(), f[2].lower()) in found_fact_keys
        )

        if entities_found < test.min_entities:
            issues.append(f"Too few entities: {entities_found} < {test.min_entities}")
        if relationships_found < test.min_relationships:
            issues.append(f"Too few relationships: {relationships_found} < {test.min_relationships}")
        if facts_found < test.min_facts:
            issues.append(f"Too few facts: {facts_found} < {test.min_facts}")

        passed = len(issues) == 0 and query_type_correct

        return TestResult(
            query=test.query,
            description=test.description,
            passed=passed,
            query_type_correct=query_type_correct,
            entities_found=entities_found,
            entities_expected=len(test.expected_entities),
            relationships_found=relationships_found,
            relationships_expected=len(test.expected_relationships),
            facts_found=facts_found,
            facts_expected=len(test.expected_facts),
            context_length=len(result.context),
            elapsed_ms=result.stats.get("elapsed_ms", 0),
            issues=issues,
        )

    def run_all(self, tests: list[TestCase]) -> list[TestResult]:
        """Run all test cases and return results."""
        results = []
        for test in tests:
            logger.info("Running test: %s", test.description or test.query)
            result = self.run_test(test)
            status = "PASS" if result.passed else "FAIL"
            logger.info("  %s (%.1fms)", status, result.elapsed_ms)
            if result.issues:
                for issue in result.issues:
                    logger.warning("    - %s", issue)
            results.append(result)
        return results

    def print_report(self, results: list[TestResult]) -> None:
        """Print a summary report."""
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT: {passed}/{total} passed ({failed} failed)")
        print(f"{'='*60}\n")

        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"[{status}] {r.description or r.query}")
            print(f"  Query type: {'correct' if r.query_type_correct else 'WRONG'}")
            print(f"  Entities: {r.entities_found}/{r.entities_expected}")
            print(f"  Relationships: {r.relationships_found}/{r.relationships_expected}")
            print(f"  Facts: {r.facts_found}/{r.facts_expected}")
            print(f"  Context: {r.context_length} chars, {r.elapsed_ms:.1f}ms")
            if r.issues:
                for issue in r.issues:
                    print(f"  ISSUE: {issue}")
            print()

    @staticmethod
    def load_tests(path: str | Path) -> list[TestCase]:
        """Load test cases from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return [TestCase(**t) for t in data]
