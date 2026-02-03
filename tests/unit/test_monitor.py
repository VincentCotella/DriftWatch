"""Tests for the Monitor class."""

import pandas as pd
import pytest

from driftwatch import Monitor
from driftwatch.core.report import DriftStatus


class TestMonitor:
    """Tests for the main Monitor class."""

    def test_init_with_valid_data(self, sample_numerical_df: pd.DataFrame) -> None:
        """Should initialize successfully with valid data."""
        monitor = Monitor(
            reference_data=sample_numerical_df,
            features=["age", "income"],
        )

        assert monitor.monitored_features == ["age", "income"]
        assert len(monitor._detectors) == 2

    def test_init_all_columns_if_no_features(
        self, sample_numerical_df: pd.DataFrame
    ) -> None:
        """Should monitor all columns if features not specified."""
        monitor = Monitor(reference_data=sample_numerical_df)

        assert set(monitor.monitored_features) == set(sample_numerical_df.columns)

    def test_init_empty_data_raises(self) -> None:
        """Should raise ValueError for empty reference data."""
        with pytest.raises(ValueError, match="Reference data cannot be empty"):
            Monitor(reference_data=pd.DataFrame())

    def test_init_invalid_feature_raises(
        self, sample_numerical_df: pd.DataFrame
    ) -> None:
        """Should raise ValueError for non-existent feature."""
        with pytest.raises(ValueError, match="Feature 'invalid' not found"):
            Monitor(
                reference_data=sample_numerical_df,
                features=["age", "invalid"],
            )

    def test_check_no_drift(self, sample_numerical_df: pd.DataFrame) -> None:
        """Should report no drift for identical data."""
        monitor = Monitor(reference_data=sample_numerical_df)

        report = monitor.check(sample_numerical_df)

        assert not report.has_drift()
        assert report.status == DriftStatus.OK

    def test_check_with_drift(
        self,
        sample_numerical_df: pd.DataFrame,
        drifted_numerical_df: pd.DataFrame,
    ) -> None:
        """Should detect drift between different distributions."""
        monitor = Monitor(reference_data=sample_numerical_df)

        report = monitor.check(drifted_numerical_df)

        assert report.has_drift()
        assert report.status in [DriftStatus.WARNING, DriftStatus.CRITICAL]

    def test_check_empty_production_raises(
        self, sample_numerical_df: pd.DataFrame
    ) -> None:
        """Should raise ValueError for empty production data."""
        monitor = Monitor(reference_data=sample_numerical_df)

        with pytest.raises(ValueError, match="Production data cannot be empty"):
            monitor.check(pd.DataFrame())

    def test_check_missing_features_raises(
        self, sample_numerical_df: pd.DataFrame
    ) -> None:
        """Should raise ValueError for missing features in production data."""
        monitor = Monitor(
            reference_data=sample_numerical_df,
            features=["age", "income"],
        )

        production = sample_numerical_df[["age"]].copy()

        with pytest.raises(ValueError, match="Missing features"):
            monitor.check(production)

    def test_add_feature(self, sample_numerical_df: pd.DataFrame) -> None:
        """Should add a new feature to monitoring."""
        monitor = Monitor(
            reference_data=sample_numerical_df,
            features=["age"],
        )

        monitor.add_feature("income")

        assert "income" in monitor.monitored_features

    def test_remove_feature(self, sample_numerical_df: pd.DataFrame) -> None:
        """Should remove a feature from monitoring."""
        monitor = Monitor(
            reference_data=sample_numerical_df,
            features=["age", "income"],
        )

        monitor.remove_feature("income")

        assert "income" not in monitor.monitored_features

    def test_custom_thresholds(self, sample_numerical_df: pd.DataFrame) -> None:
        """Should use custom thresholds."""
        monitor = Monitor(
            reference_data=sample_numerical_df,
            thresholds={"psi": 0.1},  # Stricter threshold
        )

        assert monitor.thresholds["psi"] == 0.1
        assert monitor.thresholds["ks_pvalue"] == 0.05  # Default preserved
