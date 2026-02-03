"""Tests for DriftReport class."""

import json
from datetime import datetime

import pytest

from driftwatch.core.report import DriftReport, DriftStatus, FeatureDriftResult


class TestFeatureDriftResult:
    """Tests for FeatureDriftResult dataclass."""

    def test_to_dict(self) -> None:
        """Should convert to dictionary correctly."""
        result = FeatureDriftResult(
            feature_name="age",
            has_drift=True,
            score=0.35,
            method="psi",
            threshold=0.2,
            p_value=None,
        )

        d = result.to_dict()

        assert d["feature_name"] == "age"
        assert d["has_drift"] is True
        assert d["score"] == 0.35
        assert d["method"] == "psi"
        assert d["threshold"] == 0.2
        assert d["p_value"] is None

    def test_to_dict_with_p_value(self) -> None:
        """Should include p_value when present."""
        result = FeatureDriftResult(
            feature_name="income",
            has_drift=False,
            score=0.15,
            method="ks_test",
            threshold=0.05,
            p_value=0.23,
        )

        d = result.to_dict()

        assert d["p_value"] == 0.23


class TestDriftReport:
    """Tests for DriftReport class."""

    @pytest.fixture
    def no_drift_report(self) -> DriftReport:
        """Report with no drift detected."""
        return DriftReport(
            feature_results=[
                FeatureDriftResult("age", False, 0.05, "psi", 0.2),
                FeatureDriftResult("income", False, 0.08, "psi", 0.2),
            ],
            reference_size=1000,
            production_size=500,
        )

    @pytest.fixture
    def partial_drift_report(self) -> DriftReport:
        """Report with partial drift (warning)."""
        return DriftReport(
            feature_results=[
                FeatureDriftResult("age", True, 0.35, "psi", 0.2),
                FeatureDriftResult("income", False, 0.08, "psi", 0.2),
                FeatureDriftResult("score", False, 0.12, "psi", 0.2),
            ],
            reference_size=1000,
            production_size=500,
        )

    @pytest.fixture
    def critical_drift_report(self) -> DriftReport:
        """Report with critical drift (>=50% features)."""
        return DriftReport(
            feature_results=[
                FeatureDriftResult("age", True, 0.35, "psi", 0.2),
                FeatureDriftResult("income", True, 0.45, "psi", 0.2),
            ],
            reference_size=1000,
            production_size=500,
        )

    def test_has_drift_false(self, no_drift_report: DriftReport) -> None:
        """Should return False when no features have drift."""
        assert not no_drift_report.has_drift()

    def test_has_drift_true(self, partial_drift_report: DriftReport) -> None:
        """Should return True when any feature has drift."""
        assert partial_drift_report.has_drift()

    def test_drifted_features(self, partial_drift_report: DriftReport) -> None:
        """Should return list of drifted feature names."""
        assert partial_drift_report.drifted_features() == ["age"]

    def test_drift_ratio(self, partial_drift_report: DriftReport) -> None:
        """Should calculate correct drift ratio."""
        assert partial_drift_report.drift_ratio() == pytest.approx(1 / 3)

    def test_drift_ratio_empty(self) -> None:
        """Should return 0 for empty feature list."""
        report = DriftReport(
            feature_results=[],
            reference_size=100,
            production_size=100,
        )
        assert report.drift_ratio() == 0.0

    def test_status_ok(self, no_drift_report: DriftReport) -> None:
        """Should return OK status when no drift."""
        assert no_drift_report.status == DriftStatus.OK

    def test_status_warning(self, partial_drift_report: DriftReport) -> None:
        """Should return WARNING status when <50% features have drift."""
        assert partial_drift_report.status == DriftStatus.WARNING

    def test_status_critical(self, critical_drift_report: DriftReport) -> None:
        """Should return CRITICAL status when >=50% features have drift."""
        assert critical_drift_report.status == DriftStatus.CRITICAL

    def test_feature_drift_found(self, partial_drift_report: DriftReport) -> None:
        """Should return feature drift result when found."""
        result = partial_drift_report.feature_drift("age")
        assert result is not None
        assert result.feature_name == "age"
        assert result.has_drift is True

    def test_feature_drift_not_found(self, partial_drift_report: DriftReport) -> None:
        """Should return None when feature not found."""
        result = partial_drift_report.feature_drift("nonexistent")
        assert result is None

    def test_summary_contains_key_info(self, partial_drift_report: DriftReport) -> None:
        """Summary should contain key information."""
        summary = partial_drift_report.summary()

        assert "DRIFT REPORT" in summary
        assert "WARNING" in summary
        assert "1,000" in summary  # reference size (formatted with comma)
        assert "500" in summary  # production size
        assert "age" in summary  # drifted feature
        assert "0.35" in summary  # drift score

    def test_to_dict(self, partial_drift_report: DriftReport) -> None:
        """Should convert to dictionary correctly."""
        d = partial_drift_report.to_dict()

        assert d["status"] == "WARNING"
        assert d["reference_size"] == 1000
        assert d["production_size"] == 500
        assert d["has_drift"] is True
        assert d["drift_ratio"] == pytest.approx(1 / 3)
        assert d["drifted_features"] == ["age"]
        assert len(d["feature_results"]) == 3

    def test_to_json(self, partial_drift_report: DriftReport) -> None:
        """Should produce valid JSON."""
        json_str = partial_drift_report.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["status"] == "WARNING"

    def test_repr(self, partial_drift_report: DriftReport) -> None:
        """Should have informative repr."""
        repr_str = repr(partial_drift_report)

        assert "DriftReport" in repr_str
        assert "WARNING" in repr_str
        assert "features=3" in repr_str
        assert "drifted=1" in repr_str

    def test_timestamp_default(self) -> None:
        """Should have timestamp when created."""
        report = DriftReport(
            feature_results=[],
            reference_size=100,
            production_size=100,
        )

        assert report.timestamp is not None
        assert isinstance(report.timestamp, datetime)

    def test_model_version(self) -> None:
        """Should support model version."""
        report = DriftReport(
            feature_results=[],
            reference_size=100,
            production_size=100,
            model_version="v1.2.3",
        )

        assert report.model_version == "v1.2.3"
        assert report.to_dict()["model_version"] == "v1.2.3"
