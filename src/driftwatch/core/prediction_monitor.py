"""
PredictionMonitor for detecting prediction drift.

Monitors changes in the distribution of model predictions (P(Ŷ))
between reference (validation) and production data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from driftwatch.core.report import DriftReport, DriftType, FeatureDriftResult

if TYPE_CHECKING:
    import pandas as pd

    from driftwatch.detectors.base import BaseDetector


class PredictionMonitor:
    """
    Monitor for detecting prediction drift (P(Ŷ) changes).

    Compares the distribution of model predictions between a reference
    set (typically validation predictions) and production predictions.

    Supports both:
    - **Regression**: Monitors prediction value distribution
    - **Classification**: Monitors predicted probabilities and class distributions

    Args:
        reference_predictions: Reference predictions (from validation set).
            Can be a 1D array/Series for regression, or 2D for multi-class probabilities.
        task: Task type ("regression" or "classification"). Default: auto-detected.
        thresholds: Dictionary of threshold values for drift detection.
        detector: Detector name to use ("psi", "ks", "jensen_shannon", etc.).
            Default: "psi" for regression, "psi" for classification probabilities.
        class_names: Optional names for classification classes.

    Example:
        ```python
        from driftwatch.core.prediction_monitor import PredictionMonitor

        # Regression
        monitor = PredictionMonitor(reference_predictions=y_val_pred)
        report = monitor.check(y_prod_pred)

        # Classification
        monitor = PredictionMonitor(
            reference_predictions=y_val_proba,  # shape (n, n_classes)
            task="classification",
            class_names=["negative", "positive"],
        )
        report = monitor.check(y_prod_proba)
        ```
    """

    DEFAULT_THRESHOLDS: ClassVar[dict[str, float]] = {
        "psi": 0.2,
        "ks_pvalue": 0.05,
        "jensen_shannon": 0.1,
    }

    def __init__(
        self,
        reference_predictions: pd.Series | pd.DataFrame | np.ndarray,
        task: str | None = None,
        thresholds: dict[str, float] | None = None,
        detector: str = "psi",
        class_names: list[str] | None = None,
    ) -> None:
        import pandas as pd

        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.detector_name = detector

        # Convert to DataFrame for consistent handling
        if isinstance(reference_predictions, np.ndarray):
            if reference_predictions.ndim == 1:
                self._ref_df = pd.DataFrame({"prediction": reference_predictions})
            else:
                cols = class_names or [
                    f"class_{i}" for i in range(reference_predictions.shape[1])
                ]
                self._ref_df = pd.DataFrame(reference_predictions, columns=cols)
        elif isinstance(reference_predictions, pd.Series):
            self._ref_df = pd.DataFrame({"prediction": reference_predictions.values})
        else:
            self._ref_df = reference_predictions.copy()

        # Auto-detect task type
        if task is None:
            self.task = (
                "classification" if len(self._ref_df.columns) > 1 else "regression"
            )
        else:
            self.task = task

        self.class_names = class_names or list(self._ref_df.columns)
        self._detector = self._create_detector()

        # Validate
        if self._ref_df.empty:
            raise ValueError("Reference predictions cannot be empty")

    def _create_detector(self) -> BaseDetector:
        """Create the appropriate detector based on configuration."""
        from driftwatch.detectors.registry import get_detector_by_name

        return get_detector_by_name(self.detector_name, self.thresholds)

    def check(
        self,
        production_predictions: pd.Series | pd.DataFrame | np.ndarray,
    ) -> DriftReport:
        """
        Check for prediction drift between reference and production predictions.

        Args:
            production_predictions: Production predictions to compare.
                Must match the shape of reference predictions.

        Returns:
            DriftReport with drift_type=PREDICTION for all results.

        Raises:
            ValueError: If predictions are empty or shape mismatch.
        """
        import pandas as pd

        # Convert production predictions
        if isinstance(production_predictions, np.ndarray):
            if production_predictions.ndim == 1:
                prod_df = pd.DataFrame({"prediction": production_predictions})
            else:
                prod_df = pd.DataFrame(
                    production_predictions, columns=self._ref_df.columns
                )
        elif isinstance(production_predictions, pd.Series):
            prod_df = pd.DataFrame({"prediction": production_predictions.values})
        else:
            prod_df = production_predictions.copy()

        if prod_df.empty:
            raise ValueError("Production predictions cannot be empty")

        if set(prod_df.columns) != set(self._ref_df.columns):
            raise ValueError(
                f"Production prediction columns {list(prod_df.columns)} "
                f"don't match reference columns {list(self._ref_df.columns)}"
            )

        # Run drift detection on each prediction column
        feature_results: list[FeatureDriftResult] = []

        for col in self._ref_df.columns:
            result = self._detector.detect(self._ref_df[col], prod_df[col])

            feature_results.append(
                FeatureDriftResult(
                    feature_name=col,
                    has_drift=result.has_drift,
                    score=result.score,
                    method=result.method,
                    threshold=result.threshold,
                    p_value=result.p_value,
                    drift_type=DriftType.PREDICTION,
                )
            )

        return DriftReport(
            feature_results=feature_results,
            reference_size=len(self._ref_df),
            production_size=len(prod_df),
        )

    @property
    def monitored_outputs(self) -> list[str]:
        """Return list of monitored prediction outputs."""
        return list(self._ref_df.columns)

    def get_config(self) -> dict[str, Any]:
        """Get monitor configuration."""
        return {
            "task": self.task,
            "detector": self.detector_name,
            "thresholds": self.thresholds,
            "monitored_outputs": self.monitored_outputs,
            "reference_size": len(self._ref_df),
        }
