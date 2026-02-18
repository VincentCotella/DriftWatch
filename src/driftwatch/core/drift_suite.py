"""
DriftSuite — Unified drift monitoring across all drift types.

Combines Feature Drift, Prediction Drift, and Concept Drift
into a single monitoring interface with clear separation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from driftwatch.core.concept_monitor import ConceptMonitor
from driftwatch.core.monitor import Monitor
from driftwatch.core.prediction_monitor import PredictionMonitor
from driftwatch.core.report import ComprehensiveDriftReport

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class DriftSuite:
    """
    Unified drift monitoring suite combining all drift types.

    Manages Feature Drift, Prediction Drift, and Concept Drift monitors,
    producing a comprehensive report that clearly distinguishes each type.

    Args:
        reference_data: Reference DataFrame for feature drift monitoring
        reference_predictions: Reference predictions for prediction drift.
            If None, prediction drift is not monitored.
        features: List of feature columns. If None, all columns are monitored.
        task: Task type for concept drift ("classification" or "regression").
        prediction_detector: Detector to use for prediction drift (default: "psi")
        thresholds: Dict of threshold values shared across monitors.
        performance_metrics: Metrics to monitor for concept drift.
        model_version: Optional model version identifier.

    Example:
        ```python
        from driftwatch.core.drift_suite import DriftSuite

        suite = DriftSuite(
            reference_data=X_train,
            reference_predictions=y_val_pred,
            task="classification",
            model_version="v1.2.0",
        )

        # Full check (all drift types)
        report = suite.check(
            production_data=X_prod,
            production_predictions=y_prod_pred,
            y_true_ref=y_val,
            y_pred_ref=y_val_pred,
            y_true_prod=y_prod,
            y_pred_prod=y_prod_pred,
        )

        print(report.summary())
        # Shows: FEATURE DRIFT, PREDICTION DRIFT, CONCEPT DRIFT sections

        # Check only feature drift
        report = suite.check(production_data=X_prod)

        # Check feature + prediction drift (no labels needed)
        report = suite.check(
            production_data=X_prod,
            production_predictions=y_prod_pred,
        )
        ```
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        reference_predictions: pd.Series | pd.DataFrame | np.ndarray | None = None,
        features: list[str] | None = None,
        task: str = "classification",
        prediction_detector: str = "psi",
        thresholds: dict[str, float] | None = None,
        performance_metrics: list[str] | None = None,
        model_version: str | None = None,
    ) -> None:
        self.model_version = model_version
        self.task = task

        # Feature drift monitor (always available)
        self._feature_monitor = Monitor(
            reference_data=reference_data,
            features=features,
            thresholds=thresholds,
        )

        # Prediction drift monitor (optional)
        self._prediction_monitor: PredictionMonitor | None = None
        if reference_predictions is not None:
            self._prediction_monitor = PredictionMonitor(
                reference_predictions=reference_predictions,
                task=task,
                thresholds=thresholds,
                detector=prediction_detector,
            )

        # Concept drift monitor (always available, but check requires labels)
        self._concept_monitor = ConceptMonitor(
            task=task,
            metrics=performance_metrics,
            thresholds=thresholds,
        )

    def check(
        self,
        production_data: pd.DataFrame | None = None,
        production_predictions: pd.Series | pd.DataFrame | np.ndarray | None = None,
        y_true_ref: np.ndarray | pd.Series | None = None,
        y_pred_ref: np.ndarray | pd.Series | None = None,
        y_true_prod: np.ndarray | pd.Series | None = None,
        y_pred_prod: np.ndarray | pd.Series | None = None,
    ) -> ComprehensiveDriftReport:
        """
        Run drift checks across all available drift types.

        Each drift type is checked only if the required data is provided:
        - **Feature Drift**: Requires `production_data`
        - **Prediction Drift**: Requires `production_predictions`
          (and reference predictions from __init__)
        - **Concept Drift**: Requires all four y_true/y_pred arrays

        Args:
            production_data: Production DataFrame for feature drift
            production_predictions: Production predictions for prediction drift
            y_true_ref: Reference true labels for concept drift
            y_pred_ref: Reference predictions for concept drift
            y_true_prod: Production true labels for concept drift
            y_pred_prod: Production predictions for concept drift

        Returns:
            ComprehensiveDriftReport with clear separation by drift type
        """
        feature_report = None
        prediction_report = None
        concept_report = None

        # Feature Drift
        if production_data is not None:
            feature_report = self._feature_monitor.check(production_data)

        # Prediction Drift
        if production_predictions is not None and self._prediction_monitor is not None:
            prediction_report = self._prediction_monitor.check(production_predictions)

        # Concept Drift
        if all(
            x is not None for x in [y_true_ref, y_pred_ref, y_true_prod, y_pred_prod]
        ):
            concept_report = self._concept_monitor.check(
                y_true_ref=y_true_ref,  # type: ignore[arg-type]
                y_pred_ref=y_pred_ref,  # type: ignore[arg-type]
                y_true_prod=y_true_prod,  # type: ignore[arg-type]
                y_pred_prod=y_pred_prod,  # type: ignore[arg-type]
            )

        return ComprehensiveDriftReport(
            feature_report=feature_report,
            prediction_report=prediction_report,
            concept_report=concept_report,
            model_version=self.model_version,
        )

    @property
    def feature_monitor(self) -> Monitor:
        """Access the underlying feature drift monitor."""
        return self._feature_monitor

    @property
    def prediction_monitor(self) -> PredictionMonitor | None:
        """Access the underlying prediction drift monitor (if configured)."""
        return self._prediction_monitor

    @property
    def concept_monitor(self) -> ConceptMonitor:
        """Access the underlying concept drift monitor."""
        return self._concept_monitor

    def get_config(self) -> dict[str, Any]:
        """Get configuration summary of all monitors."""
        config: dict[str, Any] = {
            "model_version": self.model_version,
            "task": self.task,
            "feature_monitor": {
                "features": self._feature_monitor.monitored_features,
                "thresholds": self._feature_monitor.thresholds,
            },
            "prediction_monitor": (
                self._prediction_monitor.get_config()
                if self._prediction_monitor
                else None
            ),
            "concept_monitor": self._concept_monitor.get_config(),
        }
        return config
