"""Tests for PredictionMonitor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftwatch.core.prediction_monitor import PredictionMonitor
from driftwatch.core.report import DriftType


class TestPredictionMonitorRegression:
    """Tests for prediction drift in regression tasks."""

    def test_no_drift_identical_predictions(self) -> None:
        """Should report no drift for identical predictions."""
        np.random.seed(42)
        preds = pd.Series(np.random.normal(100, 10, 1000))

        monitor = PredictionMonitor(reference_predictions=preds)
        report = monitor.check(preds)

        assert not report.has_drift()

    def test_drift_shifted_predictions(self) -> None:
        """Should detect drift when predictions shift."""
        np.random.seed(42)
        ref_preds = pd.Series(np.random.normal(100, 10, 1000))
        prod_preds = pd.Series(np.random.normal(150, 10, 1000))

        monitor = PredictionMonitor(reference_predictions=ref_preds)
        report = monitor.check(prod_preds)

        assert report.has_drift()

    def test_results_have_prediction_drift_type(self) -> None:
        """All results should have drift_type=PREDICTION."""
        np.random.seed(42)
        preds = pd.Series(np.random.normal(0, 1, 500))

        monitor = PredictionMonitor(reference_predictions=preds)
        report = monitor.check(preds)

        for result in report.feature_results:
            assert result.drift_type == DriftType.PREDICTION

    def test_auto_detect_regression(self) -> None:
        """Should auto-detect regression from 1D input."""
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        monitor = PredictionMonitor(reference_predictions=preds)

        assert monitor.task == "regression"
        assert monitor.monitored_outputs == ["prediction"]

    def test_numpy_input(self) -> None:
        """Should accept numpy array input."""
        np.random.seed(42)
        ref_preds = np.random.normal(0, 1, 1000)
        prod_preds = np.random.normal(0, 1, 1000)

        monitor = PredictionMonitor(reference_predictions=ref_preds)
        report = monitor.check(prod_preds)

        assert not report.has_drift()

    def test_empty_reference_raises(self) -> None:
        """Should raise for empty reference predictions."""
        with pytest.raises(ValueError, match="cannot be empty"):
            PredictionMonitor(reference_predictions=pd.Series(dtype=float))

    def test_empty_production_raises(self) -> None:
        """Should raise for empty production predictions."""
        monitor = PredictionMonitor(reference_predictions=np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="cannot be empty"):
            monitor.check(np.array([]))


class TestPredictionMonitorClassification:
    """Tests for prediction drift in classification tasks."""

    def test_no_drift_classification_probabilities(self) -> None:
        """Should report no drift for identical class probabilities."""
        np.random.seed(42)
        ref_proba = np.column_stack(
            [np.random.beta(2, 5, 1000), np.random.beta(5, 2, 1000)]
        )

        monitor = PredictionMonitor(
            reference_predictions=ref_proba,
            task="classification",
            class_names=["negative", "positive"],
        )
        report = monitor.check(ref_proba)

        assert not report.has_drift()

    def test_drift_classification_probabilities(self) -> None:
        """Should detect drift when class probability distributions shift."""
        np.random.seed(42)
        ref_proba = np.column_stack(
            [np.random.beta(2, 5, 1000), np.random.beta(5, 2, 1000)]
        )
        prod_proba = np.column_stack(
            [np.random.beta(5, 2, 1000), np.random.beta(2, 5, 1000)]
        )

        monitor = PredictionMonitor(
            reference_predictions=ref_proba,
            task="classification",
            class_names=["negative", "positive"],
        )
        report = monitor.check(prod_proba)

        assert report.has_drift()

    def test_auto_detect_classification(self) -> None:
        """Should auto-detect classification from 2D input."""
        proba = np.array([[0.7, 0.3], [0.4, 0.6]])
        monitor = PredictionMonitor(reference_predictions=proba)

        assert monitor.task == "classification"
        assert len(monitor.monitored_outputs) == 2


class TestPredictionMonitorConfig:
    """Test configuration options."""

    def test_custom_detector(self) -> None:
        """Should use custom detector."""
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        monitor = PredictionMonitor(
            reference_predictions=preds,
            detector="ks",
        )

        assert monitor.detector_name == "ks"

    def test_get_config(self) -> None:
        """Should return configuration."""
        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        monitor = PredictionMonitor(reference_predictions=preds)

        config = monitor.get_config()

        assert config["task"] == "regression"
        assert config["reference_size"] == 5
