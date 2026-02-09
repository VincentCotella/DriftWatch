"""Integration tests for full end-to-end workflow.

Tests the complete cycle: Monitor initialization → check() → DriftReport
"""

import pandas as pd
import pytest

from driftwatch import Monitor
from driftwatch.core.report import DriftReport, DriftStatus, FeatureDriftResult


@pytest.mark.integration
class TestFullWorkflowWithIris:
    """End-to-end tests using sklearn Iris dataset."""

    def test_workflow_no_drift_detected(
        self,
        iris_reference_df: pd.DataFrame,
        iris_production_no_drift: pd.DataFrame,
    ) -> None:
        """Complete workflow should report no drift for similar distributions."""
        # Only monitor numerical features
        numerical_features = iris_reference_df.select_dtypes(
            include=["number"]
        ).columns.tolist()

        monitor = Monitor(
            reference_data=iris_reference_df,
            features=numerical_features,
        )

        report = monitor.check(iris_production_no_drift)

        assert isinstance(report, DriftReport)
        assert not report.has_drift()
        assert report.status == DriftStatus.OK
        assert report.drift_ratio() == 0.0
        assert len(report.drifted_features()) == 0

    def test_workflow_drift_detected(
        self,
        iris_reference_df: pd.DataFrame,
        iris_production_with_drift: pd.DataFrame,
    ) -> None:
        """Complete workflow should detect drift for shifted distributions."""
        numerical_features = iris_reference_df.select_dtypes(
            include=["number"]
        ).columns.tolist()

        monitor = Monitor(
            reference_data=iris_reference_df,
            features=numerical_features,
        )

        report = monitor.check(iris_production_with_drift)

        assert isinstance(report, DriftReport)
        assert report.has_drift()
        assert report.status in [DriftStatus.WARNING, DriftStatus.CRITICAL]
        assert len(report.drifted_features()) > 0

    def test_report_serialization_json(
        self,
        iris_reference_df: pd.DataFrame,
        iris_production_with_drift: pd.DataFrame,
    ) -> None:
        """DriftReport should serialize to valid JSON."""
        import json

        numerical_features = iris_reference_df.select_dtypes(
            include=["number"]
        ).columns.tolist()
        monitor = Monitor(
            reference_data=iris_reference_df,
            features=numerical_features,
        )

        report = monitor.check(iris_production_with_drift)
        json_str = report.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert "status" in parsed
        assert "feature_results" in parsed
        assert "drift_ratio" in parsed
        assert isinstance(parsed["feature_results"], list)

    def test_report_serialization_dict(
        self,
        iris_reference_df: pd.DataFrame,
        iris_production_with_drift: pd.DataFrame,
    ) -> None:
        """DriftReport should convert to dictionary correctly."""
        numerical_features = iris_reference_df.select_dtypes(
            include=["number"]
        ).columns.tolist()
        monitor = Monitor(
            reference_data=iris_reference_df,
            features=numerical_features,
        )

        report = monitor.check(iris_production_with_drift)
        report_dict = report.to_dict()

        assert isinstance(report_dict, dict)
        assert report_dict["status"] in ["OK", "WARNING", "CRITICAL"]
        assert report_dict["reference_size"] == len(iris_reference_df)
        assert report_dict["production_size"] == len(iris_production_with_drift)
        assert isinstance(report_dict["drifted_features"], list)


@pytest.mark.integration
class TestFullWorkflowWithLargeSyntheticData:
    """End-to-end tests using large synthetic datasets (1000+ samples)."""

    def test_workflow_large_dataset_no_drift(
        self,
        large_synthetic_reference_df: pd.DataFrame,
    ) -> None:
        """Should process large dataset without drift correctly."""
        # Using same data for reference and production = no drift
        numerical_features = ["age", "income", "score", "transactions"]

        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=numerical_features,
        )

        report = monitor.check(large_synthetic_reference_df)

        assert not report.has_drift()
        assert report.reference_size == 2000
        assert report.production_size == 2000

    def test_workflow_large_dataset_with_drift(
        self,
        large_synthetic_reference_df: pd.DataFrame,
        large_synthetic_drifted_df: pd.DataFrame,
    ) -> None:
        """Should detect drift in large dataset with shifted distributions."""
        numerical_features = ["age", "income", "score", "transactions"]

        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=numerical_features,
        )

        report = monitor.check(large_synthetic_drifted_df)

        assert report.has_drift()
        # With significant drift on all features, expect CRITICAL status
        assert report.status in [DriftStatus.WARNING, DriftStatus.CRITICAL]
        assert report.drift_ratio() > 0.5  # Most features should drift

    def test_feature_results_contain_expected_data(
        self,
        large_synthetic_reference_df: pd.DataFrame,
        large_synthetic_drifted_df: pd.DataFrame,
    ) -> None:
        """Each feature result should contain all expected fields."""
        numerical_features = ["age", "income"]

        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=numerical_features,
        )

        report = monitor.check(large_synthetic_drifted_df)

        for result in report.feature_results:
            assert isinstance(result, FeatureDriftResult)
            assert result.feature_name in numerical_features
            assert isinstance(result.has_drift, bool)
            assert isinstance(result.score, float)
            assert isinstance(result.method, str)
            assert isinstance(result.threshold, float)

    def test_individual_feature_drift_lookup(
        self,
        large_synthetic_reference_df: pd.DataFrame,
        large_synthetic_drifted_df: pd.DataFrame,
    ) -> None:
        """Should be able to lookup drift result for specific feature."""
        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=["age", "income"],
        )

        report = monitor.check(large_synthetic_drifted_df)

        age_result = report.feature_drift("age")
        assert age_result is not None
        assert age_result.feature_name == "age"
        assert age_result.has_drift  # We know age was drifted significantly

        nonexistent = report.feature_drift("nonexistent_feature")
        assert nonexistent is None

    def test_summary_output_format(
        self,
        large_synthetic_reference_df: pd.DataFrame,
        large_synthetic_drifted_df: pd.DataFrame,
    ) -> None:
        """Summary should produce readable formatted output."""
        monitor = Monitor(
            reference_data=large_synthetic_reference_df,
            features=["age", "income"],
        )

        report = monitor.check(large_synthetic_drifted_df)
        summary = report.summary()

        assert isinstance(summary, str)
        assert "DRIFT REPORT" in summary
        assert "Status:" in summary
        assert "Features analyzed:" in summary
