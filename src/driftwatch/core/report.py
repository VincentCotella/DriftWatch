"""
DriftReport class for structured drift detection results.
Utilities for summarizing and representation of DriftReport
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class DriftType(str, Enum):
    """Types of drift that can be detected.

    - FEATURE: Input data distribution shift (P(X) changes)
    - PREDICTION: Model output distribution shift (P(Y_hat) changes)
    - CONCEPT: Input-output relationship shift (P(Y|X) changes)
    """

    FEATURE = "FEATURE"
    PREDICTION = "PREDICTION"
    CONCEPT = "CONCEPT"


class DriftStatus(str, Enum):
    """Overall drift status levels."""

    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class FeatureDriftResult:
    """
    Result of drift detection for a single feature.

    Attributes:
        feature_name: name of feature.
        has_drift: whether drift was detected.
        score: drift score as produced by detector.
        method: name of detection method used.
        threshold: threshold value used for detection.
        p_value: optional p_value associated with test.

    Example:
        >>> result = FeatureDriftResult(
        ...        feature_name = "age",
        ...        has_drift = True,
        ...        score = 0.32,
        ...        method = "psi",
        ...        threshold = 0.2,
        ... )
        >>> result.to_dict()["has_drift"]
        True
    """

    feature_name: str
    has_drift: bool
    score: float
    method: str
    threshold: float
    p_value: float | None = None
    drift_type: DriftType = DriftType.FEATURE

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "has_drift": self.has_drift,
            "score": self.score,
            "method": self.method,
            "threshold": self.threshold,
            "p_value": self.p_value,
            "drift_type": self.drift_type.value,
        }


@dataclass
class DriftReport:
    """
    Comprehensive report of drift detection results.

    Contains per-feature metrics and aggregate status.

    Attributes:
        feature_results: List of per-feature drift results
        reference_size: Number of samples in reference data
        production_size: Number of samples in production data
        timestamp: When the check was performed
        model_version: Optional model version identifier
    """

    feature_results: list[FeatureDriftResult]
    reference_size: int
    production_size: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_version: str | None = None

    def has_drift(self) -> bool:
        """
        Check if any feature has drift.

        Returns:
            True if at least one feature has drift, otherwise False
        Example:
            >>> report.has_drift()
            False
        """
        return any(r.has_drift for r in self.feature_results)

    def drifted_features(self) -> list[str]:
        """
        Return list of features with detected drift.

        Returns:
            list of feature name with drift.

        Example:
            >>> report.drifted_features()
            ["age", "income"]
        """
        return [r.feature_name for r in self.feature_results if r.has_drift]

    def drift_ratio(self) -> float:
        """
        Return ratio of drifted features to total features.

        Returns:
            ratio (fraction) of detected drift features with total features.

        Example:
            >>> report.drift_ratio()
            0.25
        """
        if not self.feature_results:
            return 0.0
        return len(self.drifted_features()) / len(self.feature_results)

    @property
    def status(self) -> DriftStatus:
        """
        Determine overall drift status.

        Status levels are determined based on the proportion
        of features with detected drift:

        - OK: No drift detected
        - WARNING: <50% of features have drift
        - CRITICAL: >=50% of features have drift

        Returns:
            DriftStatus representing overall drift status

        Example:
            >>> report.status
            <DriftStatus.OK: 'OK'>

        """

        ratio = self.drift_ratio()
        if ratio == 0:
            return DriftStatus.OK
        elif ratio < 0.5:
            return DriftStatus.WARNING
        else:
            return DriftStatus.CRITICAL

    def feature_drift(self, feature_name: str) -> FeatureDriftResult | None:
        """
        Get drift result for a specific feature.

        Args:
          feature_name: name of the feature to get drift result.

        Returns:
            FeatureDriftResult if the feature exists otherwise None.

        Example:
            >>> result=report.feature_drift("age")
            >>> result.has_drift
            True
        """
        for result in self.feature_results:
            if result.feature_name == feature_name:
                return result
        return None

    def summary(self) -> str:
        """
        Generate a human-readable summary of the drift report.

        Returns:
            Formatted string summary

        Example:
            >>> print(report.summary())
            DRIFT REPORT
        """
        lines = [
            "=" * 50,
            "DRIFT REPORT",
            "=" * 50,
            f"Status: {self.status.value}",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Reference samples: {self.reference_size:,}",
            f"Production samples: {self.production_size:,}",
            "",
            f"Features analyzed: {len(self.feature_results)}",
            f"Features with drift: {len(self.drifted_features())}",
            f"Drift ratio: {self.drift_ratio():.1%}",
            "",
        ]

        if self.drifted_features():
            lines.append("Drifted features:")
            for result in self.feature_results:
                if result.has_drift:
                    lines.append(
                        f"  - {result.feature_name}: "
                        f"{result.method}={result.score:.4f} "
                        f"(threshold={result.threshold})"
                    )

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert report to dictionary.

        Returns:
            Dictionary representation of the drift report.

        Example:
            >>> report.to_dict()["status"]
            'OK'
        """
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "reference_size": self.reference_size,
            "production_size": self.production_size,
            "model_version": self.model_version,
            "has_drift": self.has_drift(),
            "drift_ratio": self.drift_ratio(),
            "drifted_features": self.drifted_features(),
            "feature_results": [r.to_dict() for r in self.feature_results],
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Convert report to JSON string.

        Args:
            indent: number of spaces used for JSON indentation.

        Returns:
            JSON-formatted string.

        Example:
            >>> json_str = report.to_json()
            >>> '"status"' in json_str
            True

        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __repr__(self) -> str:
        return (
            f"DriftReport(status={self.status.value}, "
            f"features={len(self.feature_results)}, "
            f"drifted={len(self.drifted_features())})"
        )


@dataclass
class ComprehensiveDriftReport:
    """Comprehensive drift report combining feature, prediction, and concept drift.

    Provides a unified view of all drift types with clear separation
    between each type for easy analysis and alerting.

    Attributes:
        feature_report: DriftReport from feature drift analysis
        prediction_report: Optional DriftReport from prediction drift analysis
        concept_report: Optional DriftReport from concept drift analysis
        model_version: Optional model version identifier
    """

    feature_report: DriftReport | None = None
    prediction_report: DriftReport | None = None
    concept_report: DriftReport | None = None
    model_version: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def has_drift(self) -> bool:
        """Check if any type of drift was detected."""
        reports = [self.feature_report, self.prediction_report, self.concept_report]
        return any(r.has_drift() for r in reports if r is not None)

    def drift_types_detected(self) -> list[DriftType]:
        """Return list of drift types that were detected."""
        detected: list[DriftType] = []
        if self.feature_report and self.feature_report.has_drift():
            detected.append(DriftType.FEATURE)
        if self.prediction_report and self.prediction_report.has_drift():
            detected.append(DriftType.PREDICTION)
        if self.concept_report and self.concept_report.has_drift():
            detected.append(DriftType.CONCEPT)
        return detected

    @property
    def status(self) -> DriftStatus:
        """Determine overall drift status across all drift types.

        Returns the worst status across all reports.
        CONCEPT drift is weighted most heavily.
        """
        statuses: list[DriftStatus] = []
        for report in [
            self.feature_report,
            self.prediction_report,
            self.concept_report,
        ]:
            if report is not None:
                statuses.append(report.status)

        if not statuses:
            return DriftStatus.OK

        if DriftStatus.CRITICAL in statuses:
            return DriftStatus.CRITICAL
        if DriftStatus.WARNING in statuses:
            return DriftStatus.WARNING
        return DriftStatus.OK

    def summary(self) -> str:
        """Generate a comprehensive summary of all drift types."""
        lines = [
            "=" * 60,
            "COMPREHENSIVE DRIFT REPORT",
            "=" * 60,
            f"Overall Status: {self.status.value}",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Drift Types Detected: {', '.join(d.value for d in self.drift_types_detected()) or 'None'}",
        ]

        if self.model_version:
            lines.append(f"Model Version: {self.model_version}")

        lines.append("")

        # Feature Drift Section
        lines.append("-" * 60)
        lines.append("📊 FEATURE DRIFT (Data Distribution)")
        lines.append("-" * 60)
        if self.feature_report:
            lines.append(f"  Status: {self.feature_report.status.value}")
            lines.append(f"  Drift Ratio: {self.feature_report.drift_ratio():.1%}")
            lines.append(
                f"  Affected: {len(self.feature_report.drifted_features())}"
                f"/{len(self.feature_report.feature_results)} features"
            )
            if self.feature_report.drifted_features():
                for r in self.feature_report.feature_results:
                    if r.has_drift:
                        lines.append(
                            f"    ⚠ {r.feature_name}: {r.method}={r.score:.4f}"
                        )
        else:
            lines.append("  Not analyzed")

        lines.append("")

        # Prediction Drift Section
        lines.append("-" * 60)
        lines.append("🎯 PREDICTION DRIFT (Model Output Distribution)")
        lines.append("-" * 60)
        if self.prediction_report:
            lines.append(f"  Status: {self.prediction_report.status.value}")
            lines.append(f"  Drift Ratio: {self.prediction_report.drift_ratio():.1%}")
            if self.prediction_report.drifted_features():
                for r in self.prediction_report.feature_results:
                    if r.has_drift:
                        lines.append(
                            f"    ⚠ {r.feature_name}: {r.method}={r.score:.4f}"
                        )
        else:
            lines.append("  Not analyzed")

        lines.append("")

        # Concept Drift Section
        lines.append("-" * 60)
        lines.append("🧠 CONCEPT DRIFT (Model Performance Degradation)")
        lines.append("-" * 60)
        if self.concept_report:
            lines.append(f"  Status: {self.concept_report.status.value}")
            lines.append(f"  Drift Ratio: {self.concept_report.drift_ratio():.1%}")
            if self.concept_report.drifted_features():
                for r in self.concept_report.feature_results:
                    if r.has_drift:
                        lines.append(
                            f"    ⚠ {r.feature_name}: {r.method}={r.score:.4f}"
                        )
        else:
            lines.append("  Not analyzed (requires ground truth labels)")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert comprehensive report to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "model_version": self.model_version,
            "has_drift": self.has_drift(),
            "drift_types_detected": [d.value for d in self.drift_types_detected()],
            "feature_drift": self.feature_report.to_dict()
            if self.feature_report
            else None,
            "prediction_drift": self.prediction_report.to_dict()
            if self.prediction_report
            else None,
            "concept_drift": self.concept_report.to_dict()
            if self.concept_report
            else None,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert comprehensive report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __repr__(self) -> str:
        types = ", ".join(d.value for d in self.drift_types_detected()) or "none"
        return f"ComprehensiveDriftReport(status={self.status.value}, drift_types=[{types}])"
