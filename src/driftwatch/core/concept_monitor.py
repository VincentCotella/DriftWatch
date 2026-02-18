"""
ConceptMonitor for detecting concept drift.

Monitors changes in the relationship between inputs and outputs (P(Y|X))
by tracking model performance metrics over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from driftwatch.core.report import DriftReport, DriftType, FeatureDriftResult

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class PerformanceResult:
    """Result of a single performance metric comparison.

    Attributes:
        metric_name: Name of the metric (e.g., "accuracy", "f1", "rmse")
        reference_value: Metric value on the reference (validation) set
        production_value: Metric value on the production set
        absolute_change: Absolute difference (production - reference)
        relative_change: Relative change as percentage
        has_degradation: Whether performance has degraded beyond threshold
    """

    metric_name: str
    reference_value: float
    production_value: float
    absolute_change: float
    relative_change: float
    has_degradation: bool
    threshold: float


class ConceptMonitor:
    """
    Monitor for detecting concept drift (P(Y|X) changes).

    Tracks model performance metrics between a reference period and
    production period. Significant performance degradation indicates
    that the relationship between features and target has changed.

    Supports:
    - **Classification**: accuracy, precision, recall, F1, AUC-ROC
    - **Regression**: MAE, MSE, RMSE, R², MAPE

    Args:
        task: Task type ("classification" or "regression")
        metrics: List of metrics to track. If None, uses defaults for the task.
        thresholds: Dict of metric_name -> max_allowed_degradation.
            Degradation is measured differently per metric type:
            - Higher-is-better metrics (accuracy, F1, R²): flagged if drop > threshold
            - Lower-is-better metrics (MAE, RMSE): flagged if increase > threshold
        degradation_mode: How to measure degradation:
            - "absolute": Absolute difference (default)
            - "relative": Relative percentage change

    Example:
        ```python
        from driftwatch.core.concept_monitor import ConceptMonitor

        # Classification
        monitor = ConceptMonitor(
            task="classification",
            metrics=["accuracy", "f1"],
        )
        report = monitor.check(
            y_true_ref=y_val, y_pred_ref=y_val_pred,
            y_true_prod=y_prod, y_pred_prod=y_prod_pred,
        )

        # Regression
        monitor = ConceptMonitor(
            task="regression",
            metrics=["rmse", "r2"],
        )
        report = monitor.check(
            y_true_ref=y_val, y_pred_ref=y_val_pred,
            y_true_prod=y_prod, y_pred_prod=y_prod_pred,
        )
        ```
    """

    # Metrics where higher is better
    HIGHER_IS_BETTER: ClassVar[set[str]] = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc_roc",
        "r2",
    }

    # Metrics where lower is better
    LOWER_IS_BETTER: ClassVar[set[str]] = {"mae", "mse", "rmse", "mape"}

    DEFAULT_CLASSIFICATION_METRICS: ClassVar[list[str]] = ["accuracy", "f1"]
    DEFAULT_REGRESSION_METRICS: ClassVar[list[str]] = ["rmse", "r2"]

    DEFAULT_THRESHOLDS: ClassVar[dict[str, float]] = {
        "accuracy": 0.05,
        "f1": 0.05,
        "precision": 0.05,
        "recall": 0.05,
        "auc_roc": 0.05,
        "mae": 0.1,
        "mse": 0.1,
        "rmse": 0.1,
        "r2": 0.1,
        "mape": 0.1,
    }

    def __init__(
        self,
        task: str = "classification",
        metrics: list[str] | None = None,
        thresholds: dict[str, float] | None = None,
        degradation_mode: str = "absolute",
    ) -> None:
        if task not in ("classification", "regression"):
            raise ValueError(
                f"Task must be 'classification' or 'regression', got '{task}'"
            )

        self.task = task
        self.degradation_mode = degradation_mode
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

        if metrics is None:
            self.metrics = (
                self.DEFAULT_CLASSIFICATION_METRICS
                if task == "classification"
                else self.DEFAULT_REGRESSION_METRICS
            )
        else:
            all_known = self.HIGHER_IS_BETTER | self.LOWER_IS_BETTER
            unknown = set(metrics) - all_known
            if unknown:
                raise ValueError(
                    f"Unknown metrics: {unknown}. Available: {sorted(all_known)}"
                )
            self.metrics = metrics

    def check(
        self,
        y_true_ref: np.ndarray | pd.Series,
        y_pred_ref: np.ndarray | pd.Series,
        y_true_prod: np.ndarray | pd.Series,
        y_pred_prod: np.ndarray | pd.Series,
    ) -> DriftReport:
        """
        Check for concept drift by comparing performance between reference and production.

        Args:
            y_true_ref: True labels for reference set
            y_pred_ref: Model predictions for reference set
            y_true_prod: True labels for production set
            y_pred_prod: Model predictions for production set

        Returns:
            DriftReport with drift_type=CONCEPT for all results.
            Each "feature" in the report represents a performance metric.

        Raises:
            ValueError: If inputs are empty or have mismatched lengths.
        """
        # Validate inputs
        y_true_ref_arr = np.asarray(y_true_ref)
        y_pred_ref_arr = np.asarray(y_pred_ref)
        y_true_prod_arr = np.asarray(y_true_prod)
        y_pred_prod_arr = np.asarray(y_pred_prod)

        if len(y_true_ref_arr) == 0 or len(y_pred_ref_arr) == 0:
            raise ValueError("Reference data cannot be empty")
        if len(y_true_prod_arr) == 0 or len(y_pred_prod_arr) == 0:
            raise ValueError("Production data cannot be empty")
        if len(y_true_ref_arr) != len(y_pred_ref_arr):
            raise ValueError("Reference y_true and y_pred must have the same length")
        if len(y_true_prod_arr) != len(y_pred_prod_arr):
            raise ValueError("Production y_true and y_pred must have the same length")

        # Compute metrics for both periods
        ref_metrics = self._compute_metrics(y_true_ref_arr, y_pred_ref_arr)
        prod_metrics = self._compute_metrics(y_true_prod_arr, y_pred_prod_arr)

        # Compare and build results
        feature_results: list[FeatureDriftResult] = []
        self._performance_details: list[PerformanceResult] = []

        for metric_name in self.metrics:
            ref_value = ref_metrics[metric_name]
            prod_value = prod_metrics[metric_name]

            abs_change = prod_value - ref_value

            if ref_value != 0:
                rel_change = abs_change / abs(ref_value)
            else:
                rel_change = float("inf") if abs_change != 0 else 0.0

            threshold = self.thresholds.get(metric_name, 0.05)

            # Determine degradation
            if metric_name in self.HIGHER_IS_BETTER:
                # For higher-is-better metrics, degradation = score dropped
                if self.degradation_mode == "relative":
                    has_degradation = rel_change < -threshold
                else:
                    has_degradation = abs_change < -threshold
                # Score represents how much it degraded (positive = bad)
                drift_score = max(0.0, -abs_change)
            else:
                # For lower-is-better metrics, degradation = score increased
                if self.degradation_mode == "relative":
                    has_degradation = rel_change > threshold
                else:
                    has_degradation = abs_change > threshold
                drift_score = max(0.0, abs_change)

            perf_result = PerformanceResult(
                metric_name=metric_name,
                reference_value=ref_value,
                production_value=prod_value,
                absolute_change=abs_change,
                relative_change=rel_change,
                has_degradation=has_degradation,
                threshold=threshold,
            )
            self._performance_details.append(perf_result)

            feature_results.append(
                FeatureDriftResult(
                    feature_name=metric_name,
                    has_drift=has_degradation,
                    score=drift_score,
                    method=f"performance_{self.degradation_mode}",
                    threshold=threshold,
                    drift_type=DriftType.CONCEPT,
                )
            )

        return DriftReport(
            feature_results=feature_results,
            reference_size=len(y_true_ref_arr),
            production_size=len(y_true_prod_arr),
        )

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Compute all requested metrics."""
        results: dict[str, float] = {}

        for metric in self.metrics:
            results[metric] = self._compute_single_metric(metric, y_true, y_pred)

        return results

    def _compute_single_metric(
        self,
        metric_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """Compute a single metric value."""
        if metric_name == "accuracy":
            return float(np.mean(y_true == y_pred))

        if metric_name == "precision":
            return self._precision(y_true, y_pred)

        if metric_name == "recall":
            return self._recall(y_true, y_pred)

        if metric_name == "f1":
            p = self._precision(y_true, y_pred)
            r = self._recall(y_true, y_pred)
            if p + r == 0:
                return 0.0
            return 2 * p * r / (p + r)

        if metric_name == "auc_roc":
            from scipy import stats

            # Simple AUC approximation using Mann-Whitney U statistic
            positives = y_pred[y_true == 1]
            negatives = y_pred[y_true == 0]
            if len(positives) == 0 or len(negatives) == 0:
                return 0.5
            u_stat = stats.mannwhitneyu(
                positives, negatives, alternative="greater"
            ).statistic
            return float(u_stat / (len(positives) * len(negatives)))

        if metric_name == "mae":
            return float(np.mean(np.abs(y_true - y_pred)))

        if metric_name == "mse":
            return float(np.mean((y_true - y_pred) ** 2))

        if metric_name == "rmse":
            return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        if metric_name == "r2":
            ss_res: float = float(np.sum((y_true - y_pred) ** 2))
            ss_tot: float = float(np.sum((y_true - np.mean(y_true)) ** 2))
            if ss_tot == 0:
                return 1.0 if ss_res == 0 else 0.0
            return float(1 - ss_res / ss_tot)

        if metric_name == "mape":
            mask = y_true != 0
            if not np.any(mask):
                return 0.0
            return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

        raise ValueError(f"Unknown metric: {metric_name}")

    @staticmethod
    def _precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary precision."""
        tp = float(np.sum((y_pred == 1) & (y_true == 1)))
        fp = float(np.sum((y_pred == 1) & (y_true == 0)))
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)

    @staticmethod
    def _recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute binary recall."""
        tp = float(np.sum((y_pred == 1) & (y_true == 1)))
        fn = float(np.sum((y_pred == 0) & (y_true == 1)))
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)

    @property
    def performance_details(self) -> list[PerformanceResult]:
        """Return detailed performance results from last check.

        Returns:
            List of PerformanceResult with ref/prod values and changes.
        """
        return getattr(self, "_performance_details", [])

    def get_config(self) -> dict[str, Any]:
        """Get monitor configuration."""
        return {
            "task": self.task,
            "metrics": self.metrics,
            "thresholds": {m: self.thresholds[m] for m in self.metrics},
            "degradation_mode": self.degradation_mode,
        }
