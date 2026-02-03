"""
DriftReport class for structured drift detection results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class DriftStatus(str, Enum):
    """Overall drift status levels."""

    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class FeatureDriftResult:
    """Result of drift detection for a single feature."""

    feature_name: str
    has_drift: bool
    score: float
    method: str
    threshold: float
    p_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "has_drift": self.has_drift,
            "score": self.score,
            "method": self.method,
            "threshold": self.threshold,
            "p_value": self.p_value,
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
        """Check if any feature has drift."""
        return any(r.has_drift for r in self.feature_results)

    def drifted_features(self) -> list[str]:
        """Return list of features with detected drift."""
        return [r.feature_name for r in self.feature_results if r.has_drift]

    def drift_ratio(self) -> float:
        """Return ratio of drifted features to total features."""
        if not self.feature_results:
            return 0.0
        return len(self.drifted_features()) / len(self.feature_results)

    @property
    def status(self) -> DriftStatus:
        """
        Determine overall drift status.

        - OK: No drift detected
        - WARNING: <50% of features have drift
        - CRITICAL: >=50% of features have drift
        """
        ratio = self.drift_ratio()
        if ratio == 0:
            return DriftStatus.OK
        elif ratio < 0.5:
            return DriftStatus.WARNING
        else:
            return DriftStatus.CRITICAL

    def feature_drift(self, feature_name: str) -> FeatureDriftResult | None:
        """Get drift result for a specific feature."""
        for result in self.feature_results:
            if result.feature_name == feature_name:
                return result
        return None

    def summary(self) -> str:
        """
        Generate a human-readable summary of the drift report.

        Returns:
            Formatted string summary
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
        """Convert report to dictionary."""
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
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __repr__(self) -> str:
        return (
            f"DriftReport(status={self.status.value}, "
            f"features={len(self.feature_results)}, "
            f"drifted={len(self.drifted_features())})"
        )
