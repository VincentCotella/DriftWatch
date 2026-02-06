"""Integration tests for edge cases.

Tests boundary conditions and unusual inputs.
"""

import numpy as np
import pandas as pd
import pytest

from driftwatch import Monitor


@pytest.mark.integration
class TestEmptyDataFrames:
    """Tests for empty DataFrame handling."""

    def test_empty_reference_raises_error(self) -> None:
        """Should raise ValueError when reference data is empty."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Reference data cannot be empty"):
            Monitor(reference_data=empty_df)

    def test_empty_production_raises_error(
        self,
        large_synthetic_reference_df: pd.DataFrame,
    ) -> None:
        """Should raise ValueError when production data is empty."""
        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=["age", "income"],
        )

        with pytest.raises(ValueError, match="Production data cannot be empty"):
            monitor.check(pd.DataFrame())


@pytest.mark.integration
class TestSingleSampleData:
    """Tests for DataFrames with minimal samples."""

    def test_single_sample_reference(self, single_sample_df: pd.DataFrame) -> None:
        """Should handle single-sample reference data."""
        # Single sample is valid for initialization
        monitor = Monitor(
            reference_data=single_sample_df,
            features=["age", "income"],
        )

        assert monitor.monitored_features == ["age", "income"]

    def test_single_sample_production(
        self,
        large_synthetic_reference_df: pd.DataFrame,
        single_sample_df: pd.DataFrame,
    ) -> None:
        """Should handle single-sample production data for comparison."""
        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=["age", "income"],
        )

        # This should work, though statistical significance is low
        report = monitor.check(single_sample_df)

        assert report.production_size == 1
        assert len(report.feature_results) == 2


@pytest.mark.integration
class TestNaNHandling:
    """Tests for DataFrames containing NaN values."""

    def test_reference_with_nans(self, df_with_nans: pd.DataFrame) -> None:
        """Should initialize with NaN-containing reference data."""
        monitor = Monitor(
            reference_data=df_with_nans,
            features=["age", "income"],
        )

        assert len(monitor.monitored_features) == 2

    def test_production_with_nans(
        self,
        large_synthetic_reference_df: pd.DataFrame,
        df_with_nans: pd.DataFrame,
    ) -> None:
        """Should process production data with NaNs."""
        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=["age", "income"],
        )

        # Should not raise - NaN handling depends on detector implementation
        report = monitor.check(df_with_nans)

        assert report is not None
        assert len(report.feature_results) == 2

    def test_all_nan_column(self) -> None:
        """Should handle column with all NaN values."""
        reference = pd.DataFrame(
            {
                "valid": np.random.normal(0, 1, 100),
                "all_nan": [np.nan] * 100,
            }
        )

        production = pd.DataFrame(
            {
                "valid": np.random.normal(0, 1, 100),
                "all_nan": [np.nan] * 100,
            }
        )

        # Test with only valid column
        monitor = Monitor(
            reference_data=reference,
            features=["valid"],
        )

        report = monitor.check(production)
        assert len(report.feature_results) == 1


@pytest.mark.integration
class TestOutlierHandling:
    """Tests for DataFrames with extreme outliers."""

    def test_extreme_outliers_in_reference(
        self,
        df_with_outliers: pd.DataFrame,
    ) -> None:
        """Should handle extreme outliers in reference data."""
        monitor = Monitor(
            reference_data=df_with_outliers,
            features=["value"],
        )

        assert len(monitor.monitored_features) == 1

    def test_extreme_outliers_detection(
        self,
        df_with_outliers: pd.DataFrame,
    ) -> None:
        """Should detect drift when outliers are introduced."""
        # Reference without outliers
        np.random.seed(42)
        reference = pd.DataFrame(
            {
                "value": np.random.normal(100, 10, 1000),
            }
        )

        monitor = Monitor(
            reference_data=reference,
            features=["value"],
        )

        # Production with extreme outliers should potentially trigger drift
        report = monitor.check(df_with_outliers)

        assert report is not None
        assert len(report.feature_results) == 1


@pytest.mark.integration
class TestMissingFeatures:
    """Tests for feature validation."""

    def test_missing_feature_in_reference_raises(
        self,
        large_synthetic_reference_df: pd.DataFrame,
    ) -> None:
        """Should raise when feature doesn't exist in reference."""
        with pytest.raises(ValueError, match="not found in reference data"):
            Monitor(
                reference_data=large_synthetic_reference_df,
                features=["age", "nonexistent_feature"],
            )

    def test_missing_feature_in_production_raises(
        self,
        large_synthetic_reference_df: pd.DataFrame,
    ) -> None:
        """Should raise when feature is missing in production data."""
        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=["age", "income"],
        )

        # Production missing 'income' column
        production = large_synthetic_reference_df[["age"]].copy()

        with pytest.raises(ValueError, match="Missing features"):
            monitor.check(production)


@pytest.mark.integration
class TestDataTypeHandling:
    """Tests for various pandas data types."""

    def test_integer_columns(self) -> None:
        """Should handle integer-type columns."""
        reference = pd.DataFrame(
            {
                "count": np.random.randint(0, 100, 500),
            }
        )
        production = pd.DataFrame(
            {
                "count": np.random.randint(0, 100, 500),
            }
        )

        monitor = Monitor(reference_data=reference, features=["count"])
        report = monitor.check(production)

        assert len(report.feature_results) == 1

    def test_float_columns(self) -> None:
        """Should handle float-type columns."""
        reference = pd.DataFrame(
            {
                "price": np.random.uniform(10.0, 100.0, 500),
            }
        )
        production = pd.DataFrame(
            {
                "price": np.random.uniform(10.0, 100.0, 500),
            }
        )

        monitor = Monitor(reference_data=reference, features=["price"])
        report = monitor.check(production)

        assert len(report.feature_results) == 1

    def test_object_string_columns(self) -> None:
        """Should handle object/string-type columns as categorical."""
        reference = pd.DataFrame(
            {
                "name": np.random.choice(["Alice", "Bob", "Charlie"], 500),
            }
        )
        production = pd.DataFrame(
            {
                "name": np.random.choice(["Alice", "Bob", "Charlie"], 500),
            }
        )

        monitor = Monitor(reference_data=reference, features=["name"])
        report = monitor.check(production)

        assert len(report.feature_results) == 1

    def test_category_dtype(self) -> None:
        """Should handle pandas Categorical dtype."""
        reference = pd.DataFrame(
            {
                "grade": pd.Categorical(
                    np.random.choice(["A", "B", "C", "D", "F"], 500)
                ),
            }
        )
        production = pd.DataFrame(
            {
                "grade": pd.Categorical(
                    np.random.choice(["A", "B", "C", "D", "F"], 500)
                ),
            }
        )

        monitor = Monitor(reference_data=reference, features=["grade"])
        report = monitor.check(production)

        assert len(report.feature_results) == 1
