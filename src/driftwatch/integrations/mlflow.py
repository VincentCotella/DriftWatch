"""MLflow integration for DriftWatch.

Provides logging of drift detection results to MLflow experiments,
including per-feature metrics, aggregate status, and optional
report artifacts.

Requires: ``pip install driftwatch[mlflow]``
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from driftwatch.core.report import DriftReport

logger = logging.getLogger(__name__)


def _import_mlflow() -> Any:
    """Lazily import mlflow to provide a clear error message."""
    try:
        import mlflow

        return mlflow
    except ImportError:
        raise ImportError(
            "MLflow is required for this integration. "
            "Install it with: pip install driftwatch[mlflow]"
        ) from None


class MLflowDriftTracker:
    """
    Log drift detection results to MLflow.

    Tracks drift metrics, parameters, and optional artifacts within
    an MLflow experiment. Can operate in two modes:

    1. **Managed run** (default): Creates a new MLflow run for each
       ``log_report()`` call, or appends to an active run.
    2. **External run**: Pass ``run_id`` to ``log_report()`` to log
       into an existing run (useful inside training pipelines).

    Args:
        experiment_name: MLflow experiment name. Created if it doesn't exist.
        tracking_uri: MLflow tracking server URI. If ``None``, uses the
            currently configured tracking URI (env var or default).
        prefix: Prefix for all logged metric names (default: ``"drift"``).
            Metrics are logged as ``{prefix}.{feature_name}.{stat}``.
        log_report_artifact: If ``True``, attach the full JSON drift report
            as an artifact to each run.
        tags: Additional tags to attach to every run.

    Example:
        ```python
        from driftwatch import Monitor
        from driftwatch.integrations.mlflow import MLflowDriftTracker

        monitor = Monitor(reference_data=train_df, features=["age", "income"])
        report = monitor.check(production_df)

        tracker = MLflowDriftTracker(experiment_name="my-model-drift")
        tracker.log_report(report)
        ```

    Example — inside an existing training run:
        ```python
        import mlflow

        with mlflow.start_run() as run:
            # ... training code ...
            tracker = MLflowDriftTracker(experiment_name="my-model-drift")
            tracker.log_report(report, run_id=run.info.run_id)
        ```
    """

    def __init__(
        self,
        experiment_name: str = "driftwatch",
        tracking_uri: str | None = None,
        prefix: str = "drift",
        log_report_artifact: bool = True,
        tags: dict[str, str] | None = None,
    ) -> None:
        self._mlflow = _import_mlflow()

        if tracking_uri is not None:
            self._mlflow.set_tracking_uri(tracking_uri)

        self.experiment_name = experiment_name
        self.prefix = prefix
        self.log_report_artifact = log_report_artifact
        self.tags = tags or {}

        # Ensure the experiment exists
        experiment = self._mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self._experiment_id: str = self._mlflow.create_experiment(experiment_name)
            logger.info("Created MLflow experiment: %s", experiment_name)
        else:
            self._experiment_id = experiment.experiment_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_report(
        self,
        report: DriftReport,
        run_id: str | None = None,
        run_name: str | None = None,
        extra_tags: dict[str, str] | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> str:
        """
        Log a drift report to MLflow.

        If ``run_id`` is provided, metrics are logged into that existing run.
        Otherwise, a new run is created (or the currently active run is used).

        Args:
            report: The ``DriftReport`` to log.
            run_id: Optional existing run ID to log into.
            run_name: Optional human-readable run name (ignored when
                ``run_id`` is provided).
            extra_tags: Additional tags for this specific run.
            extra_params: Additional parameters to log alongside drift data.

        Returns:
            The MLflow run ID that was used.
        """
        merged_tags = {**self.tags, **(extra_tags or {})}
        merged_tags["driftwatch.status"] = report.status.value
        merged_tags["driftwatch.version"] = self._get_driftwatch_version()

        if run_id is not None:
            return self._log_into_existing_run(
                report, run_id, merged_tags, extra_params
            )

        return self._log_new_run(report, run_name, merged_tags, extra_params)

    def get_experiment_id(self) -> str:
        """Return the MLflow experiment ID."""
        return self._experiment_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_new_run(
        self,
        report: DriftReport,
        run_name: str | None,
        tags: dict[str, str],
        extra_params: dict[str, Any] | None,
    ) -> str:
        """Create a new run and log the report."""
        # Check for an active run first
        active_run = self._mlflow.active_run()
        if active_run is not None:
            return self._log_into_existing_run(
                report, active_run.info.run_id, tags, extra_params
            )

        with self._mlflow.start_run(
            experiment_id=self._experiment_id,
            run_name=run_name or f"drift-check-{report.status.value.lower()}",
            tags=tags,
        ) as run:
            self._log_metrics_and_params(report, extra_params)
            return str(run.info.run_id)

    def _log_into_existing_run(
        self,
        report: DriftReport,
        run_id: str,
        tags: dict[str, str],
        extra_params: dict[str, Any] | None,
    ) -> str:
        """Log into an already-existing run."""
        client = self._mlflow.MlflowClient()

        for key, value in tags.items():
            client.set_tag(run_id, key, value)

        # Log via the fluent API within the run context
        with self._mlflow.start_run(run_id=run_id, nested=True):
            self._log_metrics_and_params(report, extra_params)

        return run_id

    def _log_metrics_and_params(
        self,
        report: DriftReport,
        extra_params: dict[str, Any] | None,
    ) -> None:
        """Log all metrics, params, and artifacts for a report."""
        # ---- Aggregate metrics ----
        self._mlflow.log_metrics(
            {
                f"{self.prefix}.has_drift": float(report.has_drift()),
                f"{self.prefix}.drift_ratio": report.drift_ratio(),
                f"{self.prefix}.num_features": float(len(report.feature_results)),
                f"{self.prefix}.num_drifted": float(len(report.drifted_features())),
            }
        )

        # ---- Per-feature metrics ----
        for result in report.feature_results:
            safe_name = self._sanitize_metric_name(result.feature_name)
            metrics: dict[str, float] = {
                f"{self.prefix}.{safe_name}.score": result.score,
                f"{self.prefix}.{safe_name}.has_drift": float(result.has_drift),
                f"{self.prefix}.{safe_name}.threshold": result.threshold,
            }
            if result.p_value is not None:
                metrics[f"{self.prefix}.{safe_name}.p_value"] = result.p_value
            self._mlflow.log_metrics(metrics)

        # ---- Parameters ----
        params: dict[str, Any] = {
            f"{self.prefix}.reference_size": report.reference_size,
            f"{self.prefix}.production_size": report.production_size,
            f"{self.prefix}.status": report.status.value,
        }
        if report.model_version is not None:
            params[f"{self.prefix}.model_version"] = report.model_version
        if extra_params:
            params.update(extra_params)
        self._mlflow.log_params(params)

        # ---- Artifact (full JSON report) ----
        if self.log_report_artifact:
            self._log_report_artifact(report)

    def _log_report_artifact(self, report: DriftReport) -> None:
        """Write the full report JSON as an MLflow artifact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "drift_report.json"
            artifact_path.write_text(
                json.dumps(report.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
            self._mlflow.log_artifact(str(artifact_path), artifact_path="driftwatch")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_metric_name(name: str) -> str:
        """
        Sanitize a feature name to be a valid MLflow metric name.

        MLflow metric names may contain alphanumerics, underscores,
        dashes, periods, spaces, and slashes.
        """
        sanitized = "".join(
            c if c.isalnum() or c in ("_", "-", ".", "/") else "_" for c in name
        )
        # Collapse multiple underscores
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        return sanitized.strip("_")

    @staticmethod
    def _get_driftwatch_version() -> str:
        """Get the current DriftWatch version string."""
        try:
            from driftwatch import __version__

            return __version__
        except ImportError:
            return "unknown"
