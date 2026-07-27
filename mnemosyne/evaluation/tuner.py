"""Parameter tuner: grid search over pipeline parameters for optimal recall."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from mnemosyne.embeddings import EmbeddingService
from mnemosyne.evaluation.recall_test import RecallReport, RecallTester
from mnemosyne.graph import GraphService

logger = logging.getLogger(__name__)


@dataclass
class TuneResult:
    """Result of testing one parameter combination."""
    params: dict
    avg_recall: float
    avg_latency_ms: float
    total_context_chars: int
    recall_report: RecallReport


@dataclass
class TuneReport:
    """Summary of parameter tuning."""
    total_combinations: int
    best_params: dict
    best_recall: float
    results: list[TuneResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# Small parameter grid for fast iteration
PARAM_GRID = {
    "max_graph_depth": [2, 3, 4],
    "min_similarity": [0.4, 0.5, 0.6],
    "char_budget": [6000, 8000, 12000],
}


class ParameterTuner:
    """Grid search over pipeline parameters for optimal recall."""

    def __init__(
        self,
        embeddings: EmbeddingService,
        graph: GraphService,
    ) -> None:
        self._embeddings = embeddings
        self._graph = graph
        self._tester = RecallTester(embeddings, graph)

    def tune(
        self,
        db: Session,
        param_grid: dict | None = None,
    ) -> TuneReport:
        """Run grid search over parameter combinations.

        Args:
            db: Database session.
            param_grid: Parameter grid. Uses PARAM_GRID if None.

        Returns:
            TuneReport with best parameters and all results.
        """
        param_grid = param_grid or PARAM_GRID
        combinations = self._generate_combinations(param_grid)

        logger.info("Starting parameter tuning: %d combinations", len(combinations))

        start = time.time()
        report = TuneReport(
            total_combinations=len(combinations),
            best_params={},
            best_recall=0.0,
        )

        for i, params in enumerate(combinations):
            logger.info("Testing combination %d/%d: %s", i + 1, len(combinations), params)

            # Run recall tests with this config
            recall_report = self._tester.run_all(db, params=params)

            result = TuneResult(
                params=params,
                avg_recall=recall_report.avg_recall,
                avg_latency_ms=recall_report.avg_latency_ms,
                total_context_chars=recall_report.total_context_chars,
                recall_report=recall_report,
            )
            report.results.append(result)

            # Track best
            if recall_report.avg_recall > report.best_recall:
                report.best_recall = recall_report.avg_recall
                report.best_params = params.copy()

            logger.info(
                "  → recall=%.2f, latency=%.0fms, context=%d",
                recall_report.avg_recall,
                recall_report.avg_latency_ms,
                recall_report.total_context_chars,
            )

        report.elapsed_seconds = time.time() - start

        # Sort by recall (primary), latency (secondary)
        report.results.sort(key=lambda r: (-r.avg_recall, r.avg_latency_ms))

        logger.info(
            "Tuning complete: best_recall=%.2f, best_params=%s, elapsed=%.1fs",
            report.best_recall, report.best_params, report.elapsed_seconds,
        )

        return report

    def _generate_combinations(self, grid: dict) -> list[dict]:
        """Generate all combinations from a parameter grid."""
        import itertools

        keys = list(grid.keys())
        values = list(grid.values())

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def quick_tune(self, db: Session) -> TuneReport:
        """Quick tune with a smaller grid for fast feedback."""
        small_grid = {
            "max_graph_depth": [2, 3],
            "min_similarity": [0.5, 0.6],
            "char_budget": [8000],
        }
        return self.tune(db, small_grid)
