"""Integration tests for mixed data types.

Tests scenarios with both numerical and categorical features.
"""

import numpy as np
import pandas as pd
import pytest

from driftwatch import Monitor
from driftwatch.core.report import DriftStatus


@pytest.mark.integration
class TestMixedDtypesWorkflow:
    """Tests for DataFrames with mixed numerical and categorical columns."""

    def test_mixed_dtypes_no_drift(
        self,
        large_synthetic_reference_df: pd.DataFrame,
    ) -> None:
        """Should handle mixed dtypes with no drift."""
        # Select both numerical and categorical features
        features = ["age", "income", "category", "status"]

        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=features,
        )

        # Same data = no drift
        report = monitor.check(large_synthetic_reference_df)

        assert not report.has_drift()
        assert len(report.feature_results) == 4

    def test_mixed_dtypes_with_drift(
        self,
        large_synthetic_reference_df: pd.DataFrame,
        large_synthetic_drifted_df: pd.DataFrame,
    ) -> None:
        """Should detect drift in mixed dtype dataset."""
        features = ["age", "income", "category", "status"]

        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=features,
        )

        report = monitor.check(large_synthetic_drifted_df)

        assert report.has_drift()
        # Both numerical and categorical features should show drift
        drifted = report.drifted_features()
        assert len(drifted) > 0

    def test_partial_drift_numerical_only(
        self,
        large_synthetic_reference_df: pd.DataFrame,
    ) -> None:
        """Should detect drift in numerical features while categorical stable."""
        # Create production data with only numerical drift
        production = large_synthetic_reference_df.copy()
        # Shift numerical features
        production["age"] = production["age"] + 20  # Significant shift
        production["income"] = production["income"] * 3  # Large multiplier
        # Keep categorical features identical

        features = ["age", "income", "category", "status"]
        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=features,
        )

        report = monitor.check(production)

        assert report.has_drift()
        drifted = report.drifted_features()
        # Numerical features should drift
        assert "age" in drifted or "income" in drifted

    def test_partial_drift_categorical_only(
        self,
        large_synthetic_reference_df: pd.DataFrame,
    ) -> None:
        """Should detect drift in categorical features while numerical stable."""
        np.random.seed(999)
        n = len(large_synthetic_reference_df)

        # Create production with same numerical but different categorical distribution
        production = large_synthetic_reference_df.copy()
        production["category"] = pd.Categorical(
            np.random.choice(["A", "B", "C", "D"], n, p=[0.1, 0.1, 0.1, 0.7])
        )
        production["status"] = pd.Categorical(
            np.random.choice(["active", "inactive", "pending"], n, p=[0.1, 0.1, 0.8])
        )

        features = ["age", "category", "status"]
        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=features,
        )

        report = monitor.check(production)

        # Should detect categorical drift
        drifted = report.drifted_features()
        assert "category" in drifted or "status" in drifted


@pytest.mark.integration
class TestDetectorSelection:
    """Tests that correct detectors are selected based on dtype."""

    def test_numerical_uses_appropriate_method(
        self,
        large_synthetic_reference_df: pd.DataFrame,
        large_synthetic_drifted_df: pd.DataFrame,
    ) -> None:
        """Numerical features should use KS or similar test."""
        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=["age", "income"],
        )

        report = monitor.check(large_synthetic_drifted_df)

        for result in report.feature_results:
            # Method should be appropriate for numerical data
            assert result.method in ["ks", "psi", "wasserstein"]

    def test_categorical_uses_appropriate_method(
        self,
        large_synthetic_reference_df: pd.DataFrame,
        large_synthetic_drifted_df: pd.DataFrame,
    ) -> None:
        """Categorical features should use chi2 or similar test."""
        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=["category", "status"],
        )

        report = monitor.check(large_synthetic_drifted_df)

        for result in report.feature_results:
            # Method should be appropriate for categorical data
            assert result.method in ["chi_squared", "chi2", "psi", "frequency_psi"]


@pytest.mark.integration
class TestDriftRatioCalculation:
    """Tests for drift ratio calculation with mixed features."""

    def test_drift_ratio_zero_when_no_drift(
        self,
        large_synthetic_reference_df: pd.DataFrame,
    ) -> None:
        """Drift ratio should be 0 when no features drift."""
        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=["age", "income", "category"],
        )

        report = monitor.check(large_synthetic_reference_df)

        assert report.drift_ratio() == 0.0
        assert report.status == DriftStatus.OK

    def test_drift_ratio_partial(
        self,
        large_synthetic_reference_df: pd.DataFrame,
    ) -> None:
        """Drift ratio should reflect proportion of drifted features."""
        production = large_synthetic_reference_df.copy()
        # Drift only age significantly
        production["age"] = production["age"] + 100

        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=["age", "income", "score", "transactions"],
        )

        report = monitor.check(production)

        # At least age should drift, ratio should be positive
        if report.has_drift():
            assert 0 < report.drift_ratio() <= 1.0

    def test_drift_ratio_full(
        self,
        large_synthetic_reference_df: pd.DataFrame,
        large_synthetic_drifted_df: pd.DataFrame,
    ) -> None:
        """Drift ratio should be high when all features drift."""
        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=["age", "income", "score", "transactions"],
        )

        report = monitor.check(large_synthetic_drifted_df)

        # All features have significant drift in test data
        assert report.drift_ratio() >= 0.5
        assert report.status == DriftStatus.CRITICAL


@pytest.mark.integration
class TestCustomThresholds:
    """Tests for custom threshold configuration with mixed data."""

    def test_stricter_thresholds_more_sensitivity(
        self,
        large_synthetic_reference_df: pd.DataFrame,
    ) -> None:
        """Stricter thresholds should detect more drift."""
        # Create slight drift
        production = large_synthetic_reference_df.copy()
        production["age"] = production["age"] + 5  # Small shift

        # Normal thresholds
        monitor_normal = Monitor(
            reference_data=large_synthetic_reference_df,
            features=["age"],
        )

        # Stricter thresholds
        monitor_strict = Monitor(
            reference_data=large_synthetic_reference_df,
            features=["age"],
            thresholds={"ks_pvalue": 0.5, "psi": 0.05},  # Much stricter
        )

        report_normal = monitor_normal.check(production)
        report_strict = monitor_strict.check(production)

        # Stricter thresholds may detect drift that normal doesn't
        # (or both detect, strict should not detect less)
        if report_normal.has_drift():
            assert report_strict.has_drift()

    def test_relaxed_thresholds_less_sensitivity(
        self,
        large_synthetic_reference_df: pd.DataFrame,
        large_synthetic_drifted_df: pd.DataFrame,
    ) -> None:
        """Relaxed thresholds should be more tolerant."""
        monitor_normal = Monitor(
            reference_data=large_synthetic_reference_df,
            features=["age"],
        )

        monitor_relaxed = Monitor(
            reference_data=large_synthetic_reference_df,
            features=["age"],
            thresholds={"ks_pvalue": 0.001, "psi": 1.0},  # Very relaxed
        )

        report_normal = monitor_normal.check(large_synthetic_drifted_df)
        report_relaxed = monitor_relaxed.check(large_synthetic_drifted_df)

        # Normal should definitely detect drift given significant shift
        assert report_normal.has_drift()
        # Relaxed might not (depending on how much drift)
        # At minimum, relaxed should not detect MORE drift
        assert len(report_relaxed.drifted_features()) <= len(
            report_normal.drifted_features()
        )
