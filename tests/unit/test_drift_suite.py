"""Tests for DriftSuite — unified drift monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftwatch.core.drift_suite import DriftSuite
from driftwatch.core.report import DriftStatus, DriftType


@pytest.fixture
def reference_data() -> pd.DataFrame:
    """Create reference DataFrame."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": np.random.normal(35, 10, 1000),
            "income": np.random.normal(50000, 15000, 1000),
        }
    )


@pytest.fixture
def production_data_no_drift() -> pd.DataFrame:
    """Create production DataFrame (same distribution)."""
    np.random.seed(123)
    return pd.DataFrame(
        {
            "age": np.random.normal(35, 10, 500),
            "income": np.random.normal(50000, 15000, 500),
        }
    )


@pytest.fixture
def production_data_drift() -> pd.DataFrame:
    """Create production DataFrame (shifted distribution)."""
    np.random.seed(123)
    return pd.DataFrame(
        {
            "age": np.random.normal(55, 10, 500),  # Shifted
            "income": np.random.normal(80000, 15000, 500),  # Shifted
        }
    )


class TestDriftSuiteFeatureOnly:
    """Test DriftSuite with feature drift only."""

    def test_no_drift(
        self,
        reference_data: pd.DataFrame,
        production_data_no_drift: pd.DataFrame,
    ) -> None:
        """Should report no drift for similar distributions."""
        suite = DriftSuite(reference_data=reference_data)
        report = suite.check(production_data=production_data_no_drift)

        assert not report.has_drift()
        assert report.status == DriftStatus.OK
        assert report.feature_report is not None
        assert report.prediction_report is None
        assert report.concept_report is None

    def test_feature_drift_detected(
        self,
        reference_data: pd.DataFrame,
        production_data_drift: pd.DataFrame,
    ) -> None:
        """Should detect feature drift."""
        suite = DriftSuite(reference_data=reference_data)
        report = suite.check(production_data=production_data_drift)

        assert report.has_drift()
        assert DriftType.FEATURE in report.drift_types_detected()


class TestDriftSuiteWithPredictions:
    """Test DriftSuite with prediction drift."""

    def test_prediction_drift_detected(
        self,
        reference_data: pd.DataFrame,
    ) -> None:
        """Should detect prediction drift when prediction distributions shift."""
        np.random.seed(42)
        ref_preds = np.random.normal(0.5, 0.1, 1000)
        prod_preds = np.random.normal(0.8, 0.1, 500)

        suite = DriftSuite(
            reference_data=reference_data,
            reference_predictions=ref_preds,
        )
        report = suite.check(
            production_data=reference_data.iloc[:500],
            production_predictions=prod_preds,
        )

        assert report.prediction_report is not None
        assert DriftType.PREDICTION in report.drift_types_detected()


class TestDriftSuiteWithConceptDrift:
    """Test DriftSuite with concept drift."""

    def test_concept_drift_detected(
        self,
        reference_data: pd.DataFrame,
    ) -> None:
        """Should detect concept drift when performance degrades."""
        np.random.seed(42)
        n = 500

        y_true = np.random.randint(0, 2, n)
        y_pred_ref = y_true.copy()  # Perfect accuracy
        y_pred_prod = np.random.randint(0, 2, n)  # Random (50% accuracy)

        suite = DriftSuite(
            reference_data=reference_data,
            task="classification",
        )
        report = suite.check(
            production_data=reference_data.iloc[:n],
            y_true_ref=y_true,
            y_pred_ref=y_pred_ref,
            y_true_prod=y_true,
            y_pred_prod=y_pred_prod,
        )

        assert report.concept_report is not None
        assert DriftType.CONCEPT in report.drift_types_detected()


class TestDriftSuiteFullCheck:
    """Test DriftSuite with all three drift types."""

    def test_all_drifts_detected(
        self,
        reference_data: pd.DataFrame,
        production_data_drift: pd.DataFrame,
    ) -> None:
        """Should detect all drift types simultaneously."""
        np.random.seed(42)
        n = 500

        ref_preds = np.random.normal(0.5, 0.1, 1000)
        prod_preds = np.random.normal(0.8, 0.1, n)

        y_true = np.random.randint(0, 2, n)
        y_pred_ref = y_true.copy()
        y_pred_prod = np.random.randint(0, 2, n)

        suite = DriftSuite(
            reference_data=reference_data,
            reference_predictions=ref_preds,
            task="classification",
            model_version="v1.2.0",
        )

        report = suite.check(
            production_data=production_data_drift,
            production_predictions=prod_preds,
            y_true_ref=y_true,
            y_pred_ref=y_pred_ref,
            y_true_prod=y_true,
            y_pred_prod=y_pred_prod,
        )

        assert report.has_drift()
        assert report.model_version == "v1.2.0"
        assert report.feature_report is not None
        assert report.prediction_report is not None
        assert report.concept_report is not None
        assert report.status in [DriftStatus.WARNING, DriftStatus.CRITICAL]


class TestComprehensiveDriftReport:
    """Test comprehensive report features."""

    def test_summary_output(
        self,
        reference_data: pd.DataFrame,
        production_data_drift: pd.DataFrame,
    ) -> None:
        """Summary should contain all three drift type sections."""
        suite = DriftSuite(reference_data=reference_data)
        report = suite.check(production_data=production_data_drift)

        summary = report.summary()

        assert "FEATURE DRIFT" in summary
        assert "PREDICTION DRIFT" in summary
        assert "CONCEPT DRIFT" in summary
        assert "COMPREHENSIVE DRIFT REPORT" in summary

    def test_to_dict(
        self,
        reference_data: pd.DataFrame,
    ) -> None:
        """Should convert to dict with all sections."""
        suite = DriftSuite(reference_data=reference_data)
        report = suite.check(production_data=reference_data)

        d = report.to_dict()

        assert "status" in d
        assert "drift_types_detected" in d
        assert "feature_drift" in d
        assert "prediction_drift" in d
        assert "concept_drift" in d

    def test_to_json(
        self,
        reference_data: pd.DataFrame,
    ) -> None:
        """Should convert to valid JSON."""
        import json

        suite = DriftSuite(reference_data=reference_data)
        report = suite.check(production_data=reference_data)

        json_str = report.to_json()
        parsed = json.loads(json_str)

        assert parsed["status"] == "OK"

    def test_repr(
        self,
        reference_data: pd.DataFrame,
    ) -> None:
        """Should have meaningful repr."""
        suite = DriftSuite(reference_data=reference_data)
        report = suite.check(production_data=reference_data)

        repr_str = repr(report)

        assert "ComprehensiveDriftReport" in repr_str


class TestDriftSuiteConfig:
    """Test DriftSuite configuration."""

    def test_get_config(self, reference_data: pd.DataFrame) -> None:
        """Should return full configuration."""
        suite = DriftSuite(
            reference_data=reference_data,
            model_version="v1.0",
        )

        config = suite.get_config()

        assert config["model_version"] == "v1.0"
        assert "feature_monitor" in config
        assert "concept_monitor" in config

    def test_access_sub_monitors(self, reference_data: pd.DataFrame) -> None:
        """Should expose sub-monitors via properties."""
        np.random.seed(42)
        ref_preds = np.random.normal(0, 1, 1000)

        suite = DriftSuite(
            reference_data=reference_data,
            reference_predictions=ref_preds,
        )

        assert suite.feature_monitor is not None
        assert suite.prediction_monitor is not None
        assert suite.concept_monitor is not None
