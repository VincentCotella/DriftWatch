"""Tests for ConceptMonitor."""

from __future__ import annotations

import numpy as np
import pytest

from driftwatch.core.concept_monitor import ConceptMonitor
from driftwatch.core.report import DriftType


class TestConceptMonitorClassification:
    """Tests for concept drift in classification."""

    def test_no_drift_same_performance(self) -> None:
        """Should report no drift when performance is stable."""
        np.random.seed(42)
        n = 500
        y_true = np.random.randint(0, 2, n)
        # Add some noise but similar accuracy for both
        y_pred_ref = y_true.copy()
        y_pred_ref[:25] = 1 - y_pred_ref[:25]  # 5% error
        y_pred_prod = y_true.copy()
        y_pred_prod[25:50] = 1 - y_pred_prod[25:50]  # 5% error, different samples

        monitor = ConceptMonitor(task="classification", metrics=["accuracy"])
        report = monitor.check(y_true, y_pred_ref, y_true, y_pred_prod)

        assert not report.has_drift()

    def test_drift_degraded_performance(self) -> None:
        """Should detect drift when performance degrades significantly."""
        np.random.seed(42)
        n = 500
        y_true = np.random.randint(0, 2, n)

        # Good reference performance (95% accuracy)
        y_pred_ref = y_true.copy()
        y_pred_ref[:25] = 1 - y_pred_ref[:25]

        # Bad production performance (70% accuracy)
        y_pred_prod = y_true.copy()
        y_pred_prod[:150] = 1 - y_pred_prod[:150]

        monitor = ConceptMonitor(task="classification", metrics=["accuracy"])
        report = monitor.check(y_true, y_pred_ref, y_true, y_pred_prod)

        assert report.has_drift()

    def test_results_have_concept_drift_type(self) -> None:
        """All results should have drift_type=CONCEPT."""
        np.random.seed(42)
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1])

        monitor = ConceptMonitor(task="classification")
        report = monitor.check(y_true, y_pred, y_true, y_pred)

        for result in report.feature_results:
            assert result.drift_type == DriftType.CONCEPT

    def test_f1_metric(self) -> None:
        """Should compute F1 score correctly."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])
        y_pred_ref = np.array([1, 1, 1, 0, 0, 0, 1, 0])  # Perfect
        y_pred_prod = np.array([0, 0, 0, 1, 1, 1, 0, 1])  # All wrong

        monitor = ConceptMonitor(task="classification", metrics=["f1"])
        report = monitor.check(y_true, y_pred_ref, y_true, y_pred_prod)

        assert report.has_drift()

    def test_multiple_metrics(self) -> None:
        """Should track multiple metrics simultaneously."""
        np.random.seed(42)
        n = 200
        y_true = np.random.randint(0, 2, n)
        y_pred = y_true.copy()

        monitor = ConceptMonitor(
            task="classification",
            metrics=["accuracy", "precision", "recall", "f1"],
        )
        report = monitor.check(y_true, y_pred, y_true, y_pred)

        assert len(report.feature_results) == 4
        metric_names = [r.feature_name for r in report.feature_results]
        assert "accuracy" in metric_names
        assert "precision" in metric_names
        assert "recall" in metric_names
        assert "f1" in metric_names


class TestConceptMonitorRegression:
    """Tests for concept drift in regression."""

    def test_no_drift_same_performance(self) -> None:
        """Should report no drift for stable regression performance."""
        np.random.seed(42)
        n = 500
        y_true = np.random.normal(100, 10, n)
        noise_ref = np.random.normal(0, 1, n)
        noise_prod = np.random.normal(0, 1, n)

        monitor = ConceptMonitor(task="regression", metrics=["rmse"])
        report = monitor.check(
            y_true,
            y_true + noise_ref,
            y_true,
            y_true + noise_prod,
        )

        assert not report.has_drift()

    def test_drift_degraded_regression(self) -> None:
        """Should detect drift when regression error increases significantly."""
        np.random.seed(42)
        n = 500
        y_true = np.random.normal(100, 10, n)

        # Good reference (small error)
        y_pred_ref = y_true + np.random.normal(0, 0.5, n)

        # Bad production (large error)
        y_pred_prod = y_true + np.random.normal(0, 20, n)

        monitor = ConceptMonitor(task="regression", metrics=["rmse"])
        report = monitor.check(y_true, y_pred_ref, y_true, y_pred_prod)

        assert report.has_drift()

    def test_r2_metric(self) -> None:
        """Should compute R² correctly."""
        np.random.seed(42)
        n = 200
        y_true = np.random.normal(0, 1, n)

        monitor = ConceptMonitor(task="regression", metrics=["r2"])
        report = monitor.check(
            y_true,
            y_true,  # Perfect R² = 1.0
            y_true,
            np.zeros(n),  # Bad R² ≈ 0
        )

        assert report.has_drift()

    def test_mae_metric(self) -> None:
        """Should compute MAE correctly."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred_ref = y_true  # MAE = 0
        y_pred_prod = y_true + 10  # MAE = 10

        monitor = ConceptMonitor(
            task="regression",
            metrics=["mae"],
            thresholds={"mae": 5.0},
        )
        report = monitor.check(y_true, y_pred_ref, y_true, y_pred_prod)

        assert report.has_drift()


class TestConceptMonitorEdgeCases:
    """Edge case tests."""

    def test_empty_reference_raises(self) -> None:
        """Should raise for empty reference data."""
        monitor = ConceptMonitor(task="classification")
        with pytest.raises(ValueError, match="Reference data cannot be empty"):
            monitor.check(np.array([]), np.array([]), np.array([1]), np.array([1]))

    def test_empty_production_raises(self) -> None:
        """Should raise for empty production data."""
        monitor = ConceptMonitor(task="classification")
        with pytest.raises(ValueError, match="Production data cannot be empty"):
            monitor.check(np.array([1]), np.array([1]), np.array([]), np.array([]))

    def test_mismatched_lengths_raises(self) -> None:
        """Should raise for mismatched y_true/y_pred lengths."""
        monitor = ConceptMonitor(task="classification")
        with pytest.raises(ValueError, match="same length"):
            monitor.check(
                np.array([1, 0]),
                np.array([1]),
                np.array([1]),
                np.array([1]),
            )

    def test_invalid_task_raises(self) -> None:
        """Should raise for invalid task type."""
        with pytest.raises(
            ValueError, match="must be 'classification' or 'regression'"
        ):
            ConceptMonitor(task="invalid")

    def test_unknown_metric_raises(self) -> None:
        """Should raise for unknown metric."""
        with pytest.raises(ValueError, match="Unknown metrics"):
            ConceptMonitor(task="classification", metrics=["unknown_metric"])

    def test_performance_details(self) -> None:
        """Should expose detailed performance comparison."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = y_true.copy()

        monitor = ConceptMonitor(
            task="classification",
            metrics=["accuracy", "f1"],
        )
        monitor.check(y_true, y_pred, y_true, y_pred)

        details = monitor.performance_details

        assert len(details) == 2
        assert details[0].metric_name == "accuracy"
        assert details[0].reference_value == 1.0
        assert details[0].production_value == 1.0
        assert details[0].absolute_change == 0.0

    def test_get_config(self) -> None:
        """Should return configuration."""
        monitor = ConceptMonitor(
            task="classification",
            metrics=["accuracy", "f1"],
            thresholds={"accuracy": 0.1},
        )

        config = monitor.get_config()

        assert config["task"] == "classification"
        assert "accuracy" in config["metrics"]
        assert config["thresholds"]["accuracy"] == 0.1
