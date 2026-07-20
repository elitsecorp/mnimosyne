"""Run the evaluation harness against the pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mnemosyne.database import init_db, get_session_factory
from mnemosyne.embeddings import EmbeddingService
from mnemosyne.graph import GraphService
from mnemosyne.retrieval.pipeline import ContextPipeline
from mnemosyne.evaluation.harness import EvalHarness


def main():
    """Run the evaluation."""
    init_db()

    embeddings = EmbeddingService()
    graph = GraphService()

    db = get_session_factory()()
    try:
        graph.load_from_db(db)
    finally:
        db.close()

    pipeline = ContextPipeline(embeddings, graph)

    db = get_session_factory()()
    try:
        harness = EvalHarness(pipeline, db)

        test_cases_path = Path(__file__).parent / "test_cases.json"
        tests = EvalHarness.load_tests(test_cases_path)

        print(f"Running {len(tests)} test cases...")
        results = harness.run_all(tests)
        harness.print_report(results)
    finally:
        db.close()


if __name__ == "__main__":
    main()
