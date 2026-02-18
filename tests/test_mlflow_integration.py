"""Tests for MLflow drift tracking integration."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from driftwatch.core.report import DriftReport, FeatureDriftResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_report() -> DriftReport:
    """Create a sample drift report for testing."""
    return DriftReport(
        timestamp=datetime(2026, 2, 11, 10, 0, 0, tzinfo=timezone.utc),
        reference_size=1000,
        production_size=500,
        model_version="v1.2.3",
        feature_results=[
            FeatureDriftResult(
                feature_name="age",
                has_drift=True,
                score=0.35,
                method="psi",
                threshold=0.2,
                p_value=0.001,
            ),
            FeatureDriftResult(
                feature_name="income",
                has_drift=True,
                score=0.28,
                method="psi",
                threshold=0.2,
                p_value=0.01,
            ),
            FeatureDriftResult(
                feature_name="credit_score",
                has_drift=False,
                score=0.05,
                method="psi",
                threshold=0.2,
                p_value=0.85,
            ),
        ],
    )


@pytest.fixture
def no_drift_report() -> DriftReport:
    """Create a report with no drift for testing."""
    return DriftReport(
        timestamp=datetime(2026, 2, 11, 10, 0, 0, tzinfo=timezone.utc),
        reference_size=1000,
        production_size=500,
        feature_results=[
            FeatureDriftResult(
                feature_name="age",
                has_drift=False,
                score=0.05,
                method="psi",
                threshold=0.2,
            ),
        ],
    )


@pytest.fixture
def mock_mlflow() -> MagicMock:
    """Create a comprehensive mock of the mlflow module."""
    mock = MagicMock()

    # Experiment lookup
    mock.get_experiment_by_name.return_value = None
    mock.create_experiment.return_value = "exp-123"

    # active_run returns None by default (no active run)
    mock.active_run.return_value = None

    # start_run context manager
    mock_run = MagicMock()
    mock_run.info.run_id = "run-abc-123"
    mock.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock.start_run.return_value.__exit__ = MagicMock(return_value=False)

    # Client mock
    mock_client = MagicMock()
    mock.MlflowClient.return_value = mock_client

    return mock


# ---------------------------------------------------------------------------
# Tests — Initialization
# ---------------------------------------------------------------------------


class TestMLflowDriftTrackerInit:
    """Tests for MLflowDriftTracker initialization."""

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_creates_experiment_when_not_found(
        self, mock_import: MagicMock, mock_mlflow: MagicMock
    ) -> None:
        """Should create experiment if it doesn't exist."""
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        tracker = MLflowDriftTracker(experiment_name="test-experiment")

        mock_mlflow.get_experiment_by_name.assert_called_once_with("test-experiment")
        mock_mlflow.create_experiment.assert_called_once_with("test-experiment")
        assert tracker.get_experiment_id() == "exp-123"

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_reuses_existing_experiment(
        self, mock_import: MagicMock, mock_mlflow: MagicMock
    ) -> None:
        """Should reuse experiment if it already exists."""
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "existing-456"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        tracker = MLflowDriftTracker(experiment_name="existing-exp")

        mock_mlflow.create_experiment.assert_not_called()
        assert tracker.get_experiment_id() == "existing-456"

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_sets_tracking_uri(
        self, mock_import: MagicMock, mock_mlflow: MagicMock
    ) -> None:
        """Should set tracking URI when provided."""
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        MLflowDriftTracker(
            experiment_name="test",
            tracking_uri="http://mlflow.example.com:5000",
        )

        mock_mlflow.set_tracking_uri.assert_called_once_with(
            "http://mlflow.example.com:5000"
        )

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_does_not_set_tracking_uri_when_none(
        self, mock_import: MagicMock, mock_mlflow: MagicMock
    ) -> None:
        """Should NOT set tracking URI when not provided."""
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        MLflowDriftTracker(experiment_name="test")

        mock_mlflow.set_tracking_uri.assert_not_called()

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_default_values(
        self, mock_import: MagicMock, mock_mlflow: MagicMock
    ) -> None:
        """Should have sensible defaults."""
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        tracker = MLflowDriftTracker()

        assert tracker.experiment_name == "driftwatch"
        assert tracker.prefix == "drift"
        assert tracker.log_report_artifact is True
        assert tracker.tags == {}


# ---------------------------------------------------------------------------
# Tests — log_report (new run)
# ---------------------------------------------------------------------------


class TestMLflowLogReport:
    """Tests for logging drift reports to MLflow."""

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_log_report_creates_new_run(
        self,
        mock_import: MagicMock,
        mock_mlflow: MagicMock,
        sample_report: DriftReport,
    ) -> None:
        """Should create a new MLflow run and return its ID."""
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        tracker = MLflowDriftTracker(experiment_name="test")
        run_id = tracker.log_report(sample_report)

        assert run_id == "run-abc-123"
        mock_mlflow.start_run.assert_called()

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_log_report_logs_aggregate_metrics(
        self,
        mock_import: MagicMock,
        mock_mlflow: MagicMock,
        sample_report: DriftReport,
    ) -> None:
        """Should log aggregate drift metrics."""
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        tracker = MLflowDriftTracker(experiment_name="test")
        tracker.log_report(sample_report)

        # Collect all log_metrics calls
        calls = mock_mlflow.log_metrics.call_args_list
        all_metrics: dict[str, float] = {}
        for c in calls:
            all_metrics.update(c[0][0])

        assert all_metrics["drift.has_drift"] == 1.0
        assert all_metrics["drift.num_features"] == 3.0
        assert all_metrics["drift.num_drifted"] == 2.0
        assert abs(all_metrics["drift.drift_ratio"] - 2 / 3) < 1e-9

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_log_report_logs_per_feature_metrics(
        self,
        mock_import: MagicMock,
        mock_mlflow: MagicMock,
        sample_report: DriftReport,
    ) -> None:
        """Should log per-feature drift metrics."""
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        tracker = MLflowDriftTracker(experiment_name="test")
        tracker.log_report(sample_report)

        calls = mock_mlflow.log_metrics.call_args_list
        all_metrics: dict[str, float] = {}
        for c in calls:
            all_metrics.update(c[0][0])

        # Feature "age" — drifted
        assert all_metrics["drift.age.score"] == 0.35
        assert all_metrics["drift.age.has_drift"] == 1.0
        assert all_metrics["drift.age.threshold"] == 0.2
        assert all_metrics["drift.age.p_value"] == 0.001

        # Feature "credit_score" — not drifted
        assert all_metrics["drift.credit_score.score"] == 0.05
        assert all_metrics["drift.credit_score.has_drift"] == 0.0

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_log_report_logs_params(
        self,
        mock_import: MagicMock,
        mock_mlflow: MagicMock,
        sample_report: DriftReport,
    ) -> None:
        """Should log drift parameters."""
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        tracker = MLflowDriftTracker(experiment_name="test")
        tracker.log_report(sample_report)

        calls = mock_mlflow.log_params.call_args_list
        all_params: dict[str, str] = {}
        for c in calls:
            all_params.update(c[0][0])

        assert all_params["drift.reference_size"] == 1000
        assert all_params["drift.production_size"] == 500
        assert all_params["drift.status"] == "CRITICAL"
        assert all_params["drift.model_version"] == "v1.2.3"

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_log_report_with_extra_params(
        self,
        mock_import: MagicMock,
        mock_mlflow: MagicMock,
        sample_report: DriftReport,
    ) -> None:
        """Should log extra params alongside drift data."""
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        tracker = MLflowDriftTracker(experiment_name="test")
        tracker.log_report(sample_report, extra_params={"pipeline": "production"})

        calls = mock_mlflow.log_params.call_args_list
        all_params: dict[str, str] = {}
        for c in calls:
            all_params.update(c[0][0])

        assert all_params["pipeline"] == "production"

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_log_report_with_custom_prefix(
        self,
        mock_import: MagicMock,
        mock_mlflow: MagicMock,
        sample_report: DriftReport,
    ) -> None:
        """Should use custom prefix for metric names."""
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        tracker = MLflowDriftTracker(experiment_name="test", prefix="model_v2")
        tracker.log_report(sample_report)

        calls = mock_mlflow.log_metrics.call_args_list
        all_metrics: dict[str, float] = {}
        for c in calls:
            all_metrics.update(c[0][0])

        assert "model_v2.has_drift" in all_metrics
        assert "model_v2.age.score" in all_metrics

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_log_report_no_drift(
        self,
        mock_import: MagicMock,
        mock_mlflow: MagicMock,
        no_drift_report: DriftReport,
    ) -> None:
        """Should correctly log a report with no drift."""
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        tracker = MLflowDriftTracker(experiment_name="test")
        tracker.log_report(no_drift_report)

        calls = mock_mlflow.log_metrics.call_args_list
        all_metrics: dict[str, float] = {}
        for c in calls:
            all_metrics.update(c[0][0])

        assert all_metrics["drift.has_drift"] == 0.0
        assert all_metrics["drift.num_drifted"] == 0.0


# ---------------------------------------------------------------------------
# Tests — log_report (existing run)
# ---------------------------------------------------------------------------


class TestMLflowLogIntoExistingRun:
    """Tests for logging into existing MLflow runs."""

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_log_into_existing_run(
        self,
        mock_import: MagicMock,
        mock_mlflow: MagicMock,
        sample_report: DriftReport,
    ) -> None:
        """Should log into an existing run when run_id is provided."""
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        tracker = MLflowDriftTracker(experiment_name="test")
        run_id = tracker.log_report(sample_report, run_id="existing-run-789")

        assert run_id == "existing-run-789"

        # Should set tags via MlflowClient
        client = mock_mlflow.MlflowClient.return_value
        tag_calls = client.set_tag.call_args_list
        tag_keys = [c[0][1] for c in tag_calls]
        assert "driftwatch.status" in tag_keys

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_uses_active_run_if_available(
        self,
        mock_import: MagicMock,
        mock_mlflow: MagicMock,
        sample_report: DriftReport,
    ) -> None:
        """Should use the active run when one exists and no run_id given."""
        active_run = MagicMock()
        active_run.info.run_id = "active-run-456"
        mock_mlflow.active_run.return_value = active_run
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        tracker = MLflowDriftTracker(experiment_name="test")
        run_id = tracker.log_report(sample_report)

        assert run_id == "active-run-456"


# ---------------------------------------------------------------------------
# Tests — Tags
# ---------------------------------------------------------------------------


class TestMLflowTags:
    """Tests for tag management."""

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_default_tags_include_status(
        self,
        mock_import: MagicMock,
        mock_mlflow: MagicMock,
        sample_report: DriftReport,
    ) -> None:
        """Should always include driftwatch status tag."""
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        tracker = MLflowDriftTracker(experiment_name="test")
        tracker.log_report(sample_report)

        # start_run should receive tags
        start_run_call = mock_mlflow.start_run.call_args
        tags_arg = start_run_call.kwargs.get("tags", {})
        assert tags_arg["driftwatch.status"] == "CRITICAL"

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_custom_tags_are_merged(
        self,
        mock_import: MagicMock,
        mock_mlflow: MagicMock,
        sample_report: DriftReport,
    ) -> None:
        """Should merge constructor tags with extra_tags."""
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        tracker = MLflowDriftTracker(
            experiment_name="test",
            tags={"env": "production"},
        )
        tracker.log_report(sample_report, extra_tags={"pipeline": "nightly"})

        start_run_call = mock_mlflow.start_run.call_args
        tags_arg = start_run_call.kwargs.get("tags", {})
        assert tags_arg["env"] == "production"
        assert tags_arg["pipeline"] == "nightly"


# ---------------------------------------------------------------------------
# Tests — Artifact logging
# ---------------------------------------------------------------------------


class TestMLflowArtifacts:
    """Tests for report artifact logging."""

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_artifact_is_logged(
        self,
        mock_import: MagicMock,
        mock_mlflow: MagicMock,
        sample_report: DriftReport,
    ) -> None:
        """Should log drift_report.json as an artifact."""
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        tracker = MLflowDriftTracker(experiment_name="test", log_report_artifact=True)
        tracker.log_report(sample_report)

        mock_mlflow.log_artifact.assert_called_once()
        call_args = mock_mlflow.log_artifact.call_args
        artifact_file = call_args[0][0]
        assert artifact_file.endswith("drift_report.json")
        assert call_args.kwargs["artifact_path"] == "driftwatch"

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_artifact_not_logged_when_disabled(
        self,
        mock_import: MagicMock,
        mock_mlflow: MagicMock,
        sample_report: DriftReport,
    ) -> None:
        """Should NOT log artifact when log_report_artifact is False."""
        mock_import.return_value = mock_mlflow

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        tracker = MLflowDriftTracker(experiment_name="test", log_report_artifact=False)
        tracker.log_report(sample_report)

        mock_mlflow.log_artifact.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — Utilities
# ---------------------------------------------------------------------------


class TestSanitizeMetricName:
    """Tests for metric name sanitization."""

    def test_simple_name(self) -> None:
        from driftwatch.integrations.mlflow import MLflowDriftTracker

        assert MLflowDriftTracker._sanitize_metric_name("age") == "age"

    def test_name_with_spaces(self) -> None:
        from driftwatch.integrations.mlflow import MLflowDriftTracker

        assert (
            MLflowDriftTracker._sanitize_metric_name("credit score") == "credit_score"
        )

    def test_name_with_special_chars(self) -> None:
        from driftwatch.integrations.mlflow import MLflowDriftTracker

        assert (
            MLflowDriftTracker._sanitize_metric_name("feature@#$name") == "feature_name"
        )

    def test_name_with_allowed_chars(self) -> None:
        from driftwatch.integrations.mlflow import MLflowDriftTracker

        assert (
            MLflowDriftTracker._sanitize_metric_name("feature-name_v2.0")
            == "feature-name_v2.0"
        )


class TestImportError:
    """Tests for missing mlflow dependency."""

    def test_import_error_message(self) -> None:
        """Should raise ImportError with helpful message."""
        with patch.dict("sys.modules", {"mlflow": None}):
            from driftwatch.integrations.mlflow import _import_mlflow

            with pytest.raises(ImportError, match="pip install driftwatch"):
                _import_mlflow()


class TestGetDriftwatchVersion:
    """Tests for version retrieval."""

    def test_returns_version_string(self) -> None:
        from driftwatch.integrations.mlflow import MLflowDriftTracker

        version = MLflowDriftTracker._get_driftwatch_version()
        assert version == "0.4.0"


# ---------------------------------------------------------------------------
# Tests — Feature: no p_value
# ---------------------------------------------------------------------------


class TestFeatureWithoutPValue:
    """Test handling of features without p_value."""

    @patch("driftwatch.integrations.mlflow._import_mlflow")
    def test_no_pvalue_does_not_log_pvalue_metric(
        self, mock_import: MagicMock, mock_mlflow: MagicMock
    ) -> None:
        """Should skip p_value metric when it's None."""
        mock_import.return_value = mock_mlflow

        report = DriftReport(
            timestamp=datetime(2026, 2, 11, 10, 0, 0, tzinfo=timezone.utc),
            reference_size=100,
            production_size=100,
            feature_results=[
                FeatureDriftResult(
                    feature_name="age",
                    has_drift=False,
                    score=0.05,
                    method="psi",
                    threshold=0.2,
                    p_value=None,
                ),
            ],
        )

        from driftwatch.integrations.mlflow import MLflowDriftTracker

        tracker = MLflowDriftTracker(experiment_name="test")
        tracker.log_report(report)

        calls = mock_mlflow.log_metrics.call_args_list
        all_metrics: dict[str, float] = {}
        for c in calls:
            all_metrics.update(c[0][0])

        assert "drift.age.p_value" not in all_metrics
