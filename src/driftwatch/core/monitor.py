"""
Monitor class for detecting drift between reference and production data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from driftwatch.core.report import DriftReport, FeatureDriftResult
from driftwatch.detectors import get_detector

if TYPE_CHECKING:
    from driftwatch.detectors.base import BaseDetector


class Monitor:
    """
    Main class for monitoring data and model drift.

    The Monitor compares production data against a reference dataset
    (typically training data) to detect distribution shifts.

    Args:
        reference_data: Reference DataFrame (training data)
        features: List of feature columns to monitor.
            If None, all columns are monitored.
        model: Optional ML model for prediction drift detection
        thresholds: Dictionary of threshold values for drift detection.
            Supported keys: "psi", "ks_pvalue", "wasserstein", "chi2_pvalue"

    Example:
        >>> monitor = Monitor(
        ...     reference_data=train_df,
        ...     features=["age", "income", "category"],
        ...     thresholds={"psi": 0.2, "ks_pvalue": 0.05}
        ... )
        >>> report = monitor.check(production_df)
        >>> print(report.has_drift())
    """

    DEFAULT_THRESHOLDS: dict[str, float] = {
        "psi": 0.2,
        "ks_pvalue": 0.05,
        "wasserstein": 0.1,
        "chi2_pvalue": 0.05,
    }

    def __init__(
        self,
        reference_data: pd.DataFrame,
        features: list[str] | None = None,
        model: Any | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self._validate_reference_data(reference_data)

        self.reference_data = reference_data
        self.features = features or list(reference_data.columns)
        self.model = model
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

        self._detectors: dict[str, BaseDetector] = {}
        self._setup_detectors()

    def _validate_reference_data(self, data: pd.DataFrame) -> None:
        """Validate reference data is not empty."""
        if data.empty:
            raise ValueError("Reference data cannot be empty")

    def _setup_detectors(self) -> None:
        """Initialize detectors for each feature based on dtype."""
        for feature in self.features:
            if feature not in self.reference_data.columns:
                raise ValueError(f"Feature '{feature}' not found in reference data")

            dtype = self.reference_data[feature].dtype
            detector = get_detector(dtype, self.thresholds)
            self._detectors[feature] = detector

    def check(self, production_data: pd.DataFrame) -> DriftReport:
        """
        Check for drift between reference and production data.

        Args:
            production_data: Production DataFrame to compare

        Returns:
            DriftReport containing per-feature and aggregate drift results

        Raises:
            ValueError: If production data is empty or missing features
        """
        self._validate_production_data(production_data)

        feature_results: list[FeatureDriftResult] = []

        for feature in self.features:
            ref_series = self.reference_data[feature]
            prod_series = production_data[feature]

            detector = self._detectors[feature]
            result = detector.detect(ref_series, prod_series)

            feature_results.append(
                FeatureDriftResult(
                    feature_name=feature,
                    has_drift=result.has_drift,
                    score=result.score,
                    method=result.method,
                    threshold=result.threshold,
                    p_value=result.p_value,
                )
            )

        return DriftReport(
            feature_results=feature_results,
            reference_size=len(self.reference_data),
            production_size=len(production_data),
        )

    def _validate_production_data(self, data: pd.DataFrame) -> None:
        """Validate production data has required features."""
        if data.empty:
            raise ValueError("Production data cannot be empty")

        missing = set(self.features) - set(data.columns)
        if missing:
            raise ValueError(f"Missing features in production data: {missing}")

    def add_feature(self, feature: str) -> None:
        """Add a feature to monitor."""
        if feature in self.features:
            return

        if feature not in self.reference_data.columns:
            raise ValueError(f"Feature '{feature}' not found in reference data")

        self.features.append(feature)
        dtype = self.reference_data[feature].dtype
        self._detectors[feature] = get_detector(dtype, self.thresholds)

    def remove_feature(self, feature: str) -> None:
        """Remove a feature from monitoring."""
        if feature in self.features:
            self.features.remove(feature)
            del self._detectors[feature]

    @property
    def monitored_features(self) -> list[str]:
        """Return list of monitored features."""
        return self.features.copy()
