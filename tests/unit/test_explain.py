"""Tests for the drift explanation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftwatch import Monitor
from driftwatch.explain import DriftExplainer, DriftVisualizer


def _matplotlib_available() -> bool:
    """Check if matplotlib is available."""
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def reference_data() -> pd.DataFrame:
    """Create reference data with known distributions."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": np.random.normal(30, 5, 1000),
            "income": np.random.normal(50000, 10000, 1000),
            "score": np.random.uniform(0, 100, 1000),
        }
    )


@pytest.fixture
def production_data_with_drift() -> pd.DataFrame:
    """Create production data with drift in some features."""
    np.random.seed(43)
    return pd.DataFrame(
        {
            "age": np.random.normal(40, 5, 1000),  # Mean shifted from 30 to 40
            "income": np.random.normal(50000, 10000, 1000),  # No drift
            "score": np.random.uniform(0, 100, 1000),  # No drift
        }
    )


@pytest.fixture
def production_data_no_drift() -> pd.DataFrame:
    """Create production data without drift."""
    np.random.seed(44)
    return pd.DataFrame(
        {
            "age": np.random.normal(30, 5, 1000),
            "income": np.random.normal(50000, 10000, 1000),
            "score": np.random.uniform(0, 100, 1000),
        }
    )


class TestDriftExplainer:
    """Tests for the DriftExplainer class."""

    def test_explain_returns_explanation(
        self,
        reference_data: pd.DataFrame,
        production_data_with_drift: pd.DataFrame,
    ) -> None:
        """Test that explain() returns a DriftExplanation object."""
        monitor = Monitor(reference_data=reference_data)
        report = monitor.check(production_data_with_drift)

        explainer = DriftExplainer(reference_data, production_data_with_drift, report)
        explanation = explainer.explain()

        assert explanation is not None
        assert len(explanation.feature_explanations) == 3
        assert explanation.reference_size == 1000
        assert explanation.production_size == 1000

    def test_explain_detects_mean_shift(
        self,
        reference_data: pd.DataFrame,
        production_data_with_drift: pd.DataFrame,
    ) -> None:
        """Test that the explainer correctly detects mean shift."""
        monitor = Monitor(reference_data=reference_data)
        report = monitor.check(production_data_with_drift)

        explainer = DriftExplainer(reference_data, production_data_with_drift, report)
        explanation = explainer.explain()

        age_explanation = explanation["age"]
        assert age_explanation is not None
        # Mean should shift from ~30 to ~40
        assert age_explanation.mean_shift > 5  # Significant positive shift
        assert age_explanation.mean_shift_percent > 20  # >20% increase

    def test_explain_no_drift_features(
        self,
        reference_data: pd.DataFrame,
        production_data_with_drift: pd.DataFrame,
    ) -> None:
        """Test that stable features show minimal shift."""
        monitor = Monitor(reference_data=reference_data)
        report = monitor.check(production_data_with_drift)

        explainer = DriftExplainer(reference_data, production_data_with_drift, report)
        explanation = explainer.explain()

        income_explanation = explanation["income"]
        assert income_explanation is not None
        # Income mean should be stable
        assert abs(income_explanation.mean_shift_percent) < 5  # <5% change

    def test_explain_feature_returns_single_explanation(
        self,
        reference_data: pd.DataFrame,
        production_data_with_drift: pd.DataFrame,
    ) -> None:
        """Test explain_feature() returns a single FeatureExplanation."""
        monitor = Monitor(reference_data=reference_data)
        report = monitor.check(production_data_with_drift)

        explainer = DriftExplainer(reference_data, production_data_with_drift, report)
        age_explanation = explainer.explain_feature("age")

        assert age_explanation is not None
        assert age_explanation.feature_name == "age"

    def test_explain_feature_not_found(
        self,
        reference_data: pd.DataFrame,
        production_data_with_drift: pd.DataFrame,
    ) -> None:
        """Test explain_feature() returns None for non-existent feature."""
        monitor = Monitor(reference_data=reference_data)
        report = monitor.check(production_data_with_drift)

        explainer = DriftExplainer(reference_data, production_data_with_drift, report)
        result = explainer.explain_feature("nonexistent")

        assert result is None

    def test_quantile_stats_computed(
        self,
        reference_data: pd.DataFrame,
        production_data_with_drift: pd.DataFrame,
    ) -> None:
        """Test that quantile statistics are computed."""
        monitor = Monitor(reference_data=reference_data)
        report = monitor.check(production_data_with_drift)

        explainer = DriftExplainer(reference_data, production_data_with_drift, report)
        explanation = explainer.explain()

        age_explanation = explanation["age"]
        assert age_explanation is not None
        assert len(age_explanation.quantile_stats.quantiles) == 3
        assert 0.5 in age_explanation.quantile_stats.reference_values
        assert 0.5 in age_explanation.quantile_stats.production_values

    def test_custom_quantiles(
        self,
        reference_data: pd.DataFrame,
        production_data_with_drift: pd.DataFrame,
    ) -> None:
        """Test that custom quantiles can be specified."""
        monitor = Monitor(reference_data=reference_data)
        report = monitor.check(production_data_with_drift)

        custom_quantiles = [0.1, 0.5, 0.9]
        explainer = DriftExplainer(
            reference_data,
            production_data_with_drift,
            report,
            quantiles=custom_quantiles,
        )
        explanation = explainer.explain()

        age_explanation = explanation["age"]
        assert age_explanation is not None
        assert age_explanation.quantile_stats.quantiles == custom_quantiles

    def test_explanation_summary(
        self,
        reference_data: pd.DataFrame,
        production_data_with_drift: pd.DataFrame,
    ) -> None:
        """Test that summary() produces readable output."""
        monitor = Monitor(reference_data=reference_data)
        report = monitor.check(production_data_with_drift)

        explainer = DriftExplainer(reference_data, production_data_with_drift, report)
        explanation = explainer.explain()

        summary = explanation.summary()
        assert "DRIFT EXPLANATION REPORT" in summary
        assert "age" in summary
        assert "income" in summary

    def test_explanation_to_dict(
        self,
        reference_data: pd.DataFrame,
        production_data_with_drift: pd.DataFrame,
    ) -> None:
        """Test that to_dict() produces valid dictionary."""
        monitor = Monitor(reference_data=reference_data)
        report = monitor.check(production_data_with_drift)

        explainer = DriftExplainer(reference_data, production_data_with_drift, report)
        explanation = explainer.explain()

        data = explanation.to_dict()
        assert "feature_explanations" in data
        assert "reference_size" in data
        assert "production_size" in data
        assert len(data["feature_explanations"]) == 3

    def test_drifted_features_list(
        self,
        reference_data: pd.DataFrame,
        production_data_with_drift: pd.DataFrame,
    ) -> None:
        """Test that drifted_features() returns features with drift."""
        monitor = Monitor(reference_data=reference_data)
        report = monitor.check(production_data_with_drift)

        explainer = DriftExplainer(reference_data, production_data_with_drift, report)
        explanation = explainer.explain()

        drifted = explanation.drifted_features()
        # At least age should have drift
        feature_names = [exp.feature_name for exp in drifted]
        assert "age" in feature_names


class TestDriftVisualizer:
    """Tests for the DriftVisualizer class."""

    def test_init(
        self,
        reference_data: pd.DataFrame,
        production_data_with_drift: pd.DataFrame,
    ) -> None:
        """Test DriftVisualizer initialization."""
        monitor = Monitor(reference_data=reference_data)
        report = monitor.check(production_data_with_drift)

        viz = DriftVisualizer(reference_data, production_data_with_drift, report)
        assert viz.reference_data is not None
        assert viz.production_data is not None
        assert viz.report is not None

    @pytest.mark.skipif(
        not _matplotlib_available(),
        reason="matplotlib not installed",
    )
    def test_plot_feature(
        self,
        reference_data: pd.DataFrame,
        production_data_with_drift: pd.DataFrame,
    ) -> None:
        """Test plotting a single feature."""
        monitor = Monitor(reference_data=reference_data)
        report = monitor.check(production_data_with_drift)

        viz = DriftVisualizer(reference_data, production_data_with_drift, report)
        fig = viz.plot_feature("age")

        import matplotlib.pyplot as plt

        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(
        not _matplotlib_available(),
        reason="matplotlib not installed",
    )
    def test_plot_all(
        self,
        reference_data: pd.DataFrame,
        production_data_with_drift: pd.DataFrame,
    ) -> None:
        """Test plotting all features."""
        monitor = Monitor(reference_data=reference_data)
        report = monitor.check(production_data_with_drift)

        viz = DriftVisualizer(reference_data, production_data_with_drift, report)
        fig = viz.plot_all()

        import matplotlib.pyplot as plt

        assert fig is not None
        plt.close(fig)

    @pytest.mark.skipif(
        not _matplotlib_available(),
        reason="matplotlib not installed",
    )
    def test_plot_feature_not_found(
        self,
        reference_data: pd.DataFrame,
        production_data_with_drift: pd.DataFrame,
    ) -> None:
        """Test that plotting non-existent feature raises error."""
        monitor = Monitor(reference_data=reference_data)
        report = monitor.check(production_data_with_drift)

        viz = DriftVisualizer(reference_data, production_data_with_drift, report)

        with pytest.raises(ValueError, match="not found"):
            viz.plot_feature("nonexistent")
