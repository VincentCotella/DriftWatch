"""
Statistical explanation of drift detection results.

Provides detailed metrics to understand distribution shifts:
- Mean shift
- Standard deviation change
- Quantile differences
- Min/Max changes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    from driftwatch.core.report import DriftReport


@dataclass
class QuantileStats:
    """Quantile comparison statistics."""

    quantiles: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    reference_values: dict[float, float] = field(default_factory=dict)
    production_values: dict[float, float] = field(default_factory=dict)
    absolute_diffs: dict[float, float] = field(default_factory=dict)
    relative_diffs: dict[float, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quantiles": self.quantiles,
            "reference": self.reference_values,
            "production": self.production_values,
            "absolute_diff": self.absolute_diffs,
            "relative_diff_percent": self.relative_diffs,
        }


@dataclass
class FeatureExplanation:
    """
    Detailed statistical explanation for a single feature.

    Provides comprehensive statistics to understand how and why
    a feature's distribution has shifted between reference and production.

    Attributes:
        feature_name: Name of the feature
        has_drift: Whether drift was detected
        drift_score: The drift score from the detector

        # Central tendency
        ref_mean: Reference mean
        prod_mean: Production mean
        mean_shift: Absolute change in mean
        mean_shift_percent: Relative change in mean (%)

        # Spread
        ref_std: Reference standard deviation
        prod_std: Production standard deviation
        std_change: Absolute change in std
        std_change_percent: Relative change in std (%)

        # Range
        ref_min: Reference minimum
        prod_min: Production minimum
        ref_max: Reference maximum
        prod_max: Production maximum

        # Quantiles
        quantile_stats: Detailed quantile comparison
    """

    feature_name: str
    has_drift: bool
    drift_score: float
    drift_method: str

    # Central tendency
    ref_mean: float
    prod_mean: float
    mean_shift: float
    mean_shift_percent: float

    # Spread
    ref_std: float
    prod_std: float
    std_change: float
    std_change_percent: float

    # Range
    ref_min: float
    prod_min: float
    ref_max: float
    prod_max: float

    # Quantiles
    quantile_stats: QuantileStats

    # Sample sizes
    ref_count: int
    prod_count: int

    def summary(self) -> str:
        """Generate a human-readable summary of the feature explanation."""
        drift_status = "🔴 DRIFT DETECTED" if self.has_drift else "✅ NO DRIFT"

        lines = [
            f"━━━ {self.feature_name} ━━━",
            f"Status: {drift_status}",
            f"Score ({self.drift_method}): {self.drift_score:.4f}",
            "",
            "📊 Central Tendency:",
            f"  Mean: {self.ref_mean:.4f} → {self.prod_mean:.4f} "
            f"({self.mean_shift_percent:+.2f}%)",
            "",
            "📈 Spread:",
            f"  Std: {self.ref_std:.4f} → {self.prod_std:.4f} "
            f"({self.std_change_percent:+.2f}%)",
            "",
            "📏 Range:",
            f"  Min: {self.ref_min:.4f} → {self.prod_min:.4f}",
            f"  Max: {self.ref_max:.4f} → {self.prod_max:.4f}",
            "",
            "📐 Quantiles:",
        ]

        for q in self.quantile_stats.quantiles:
            ref_val = self.quantile_stats.reference_values.get(q, 0)
            prod_val = self.quantile_stats.production_values.get(q, 0)
            rel_diff = self.quantile_stats.relative_diffs.get(q, 0)
            q_pct = int(q * 100)
            lines.append(
                f"  Q{q_pct}: {ref_val:.4f} → {prod_val:.4f} ({rel_diff:+.2f}%)"
            )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON export."""
        return {
            "feature_name": self.feature_name,
            "has_drift": self.has_drift,
            "drift_score": self.drift_score,
            "drift_method": self.drift_method,
            "central_tendency": {
                "ref_mean": self.ref_mean,
                "prod_mean": self.prod_mean,
                "mean_shift": self.mean_shift,
                "mean_shift_percent": self.mean_shift_percent,
            },
            "spread": {
                "ref_std": self.ref_std,
                "prod_std": self.prod_std,
                "std_change": self.std_change,
                "std_change_percent": self.std_change_percent,
            },
            "range": {
                "ref_min": self.ref_min,
                "prod_min": self.prod_min,
                "ref_max": self.ref_max,
                "prod_max": self.prod_max,
            },
            "quantiles": self.quantile_stats.to_dict(),
            "sample_sizes": {
                "reference": self.ref_count,
                "production": self.prod_count,
            },
        }


@dataclass
class DriftExplanation:
    """
    Complete drift explanation for all features.

    Aggregates FeatureExplanation objects with overall summary.
    """

    feature_explanations: list[FeatureExplanation]
    reference_size: int
    production_size: int

    def __getitem__(self, feature_name: str) -> FeatureExplanation | None:
        """Get explanation for a specific feature."""
        for exp in self.feature_explanations:
            if exp.feature_name == feature_name:
                return exp
        return None

    def drifted_features(self) -> list[FeatureExplanation]:
        """Return explanations for features with drift."""
        return [exp for exp in self.feature_explanations if exp.has_drift]

    def summary(self) -> str:
        """Generate a comprehensive summary of all feature explanations."""
        n_drifted = len(self.drifted_features())
        n_total = len(self.feature_explanations)

        lines = [
            "═" * 60,
            "DRIFT EXPLANATION REPORT",
            "═" * 60,
            f"Reference samples: {self.reference_size:,}",
            f"Production samples: {self.production_size:,}",
            f"Features with drift: {n_drifted}/{n_total}",
            "═" * 60,
            "",
        ]

        for exp in self.feature_explanations:
            lines.append(exp.summary())
            lines.append("")

        lines.append("═" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "reference_size": self.reference_size,
            "production_size": self.production_size,
            "feature_explanations": [
                exp.to_dict() for exp in self.feature_explanations
            ],
        }


class DriftExplainer:
    """
    Explains drift detection results with detailed statistics.

    The explainer takes reference and production DataFrames along with
    a DriftReport and provides detailed insights into why drift was
    detected and how distributions have shifted.

    Example:
        >>> from driftwatch import Monitor
        >>> from driftwatch.explain import DriftExplainer
        >>>
        >>> monitor = Monitor(reference_data=train_df)
        >>> report = monitor.check(prod_df)
        >>>
        >>> explainer = DriftExplainer(train_df, prod_df, report)
        >>> explanation = explainer.explain()
        >>> print(explanation.summary())
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        production_data: pd.DataFrame,
        report: DriftReport,
        quantiles: list[float] | None = None,
    ) -> None:
        """
        Initialize the DriftExplainer.

        Args:
            reference_data: Reference DataFrame (training data)
            production_data: Production DataFrame to explain
            report: DriftReport from Monitor.check()
            quantiles: List of quantiles to analyze (default: [0.25, 0.5, 0.75])
        """
        self.reference_data = reference_data
        self.production_data = production_data
        self.report = report
        self.quantiles = quantiles or [0.25, 0.5, 0.75]

    def explain(self) -> DriftExplanation:
        """
        Generate detailed explanations for all features.

        Returns:
            DriftExplanation containing per-feature statistical analysis
        """
        explanations: list[FeatureExplanation] = []

        for feature_result in self.report.feature_results:
            feature_name = feature_result.feature_name

            # Skip if feature not in both datasets
            if (
                feature_name not in self.reference_data.columns
                or feature_name not in self.production_data.columns
            ):
                continue

            ref_series = self.reference_data[feature_name].dropna()
            prod_series = self.production_data[feature_name].dropna()

            # Skip non-numeric features for now
            if not np.issubdtype(ref_series.dtype, np.number):
                continue

            explanation = self._explain_feature(
                feature_name=feature_name,
                ref_series=ref_series,
                prod_series=prod_series,
                has_drift=feature_result.has_drift,
                drift_score=feature_result.score,
                drift_method=feature_result.method,
            )
            explanations.append(explanation)

        return DriftExplanation(
            feature_explanations=explanations,
            reference_size=len(self.reference_data),
            production_size=len(self.production_data),
        )

    def explain_feature(self, feature_name: str) -> FeatureExplanation | None:
        """
        Generate detailed explanation for a single feature.

        Args:
            feature_name: Name of the feature to explain

        Returns:
            FeatureExplanation or None if feature not found
        """
        feature_result = self.report.feature_drift(feature_name)
        if feature_result is None:
            return None

        if (
            feature_name not in self.reference_data.columns
            or feature_name not in self.production_data.columns
        ):
            return None

        ref_series = self.reference_data[feature_name].dropna()
        prod_series = self.production_data[feature_name].dropna()

        if not np.issubdtype(ref_series.dtype, np.number):
            return None

        return self._explain_feature(
            feature_name=feature_name,
            ref_series=ref_series,
            prod_series=prod_series,
            has_drift=feature_result.has_drift,
            drift_score=feature_result.score,
            drift_method=feature_result.method,
        )

    def _explain_feature(
        self,
        feature_name: str,
        ref_series: pd.Series,
        prod_series: pd.Series,
        has_drift: bool,
        drift_score: float,
        drift_method: str,
    ) -> FeatureExplanation:
        """Internal method to compute feature explanation."""
        # Central tendency
        ref_mean = float(ref_series.mean())
        prod_mean = float(prod_series.mean())
        mean_shift = prod_mean - ref_mean
        mean_shift_percent = self._safe_percent_change(ref_mean, prod_mean)

        # Spread
        ref_std = float(ref_series.std())
        prod_std = float(prod_series.std())
        std_change = prod_std - ref_std
        std_change_percent = self._safe_percent_change(ref_std, prod_std)

        # Range
        ref_min = float(ref_series.min())
        prod_min = float(prod_series.min())
        ref_max = float(ref_series.max())
        prod_max = float(prod_series.max())

        # Quantiles
        quantile_stats = self._compute_quantile_stats(ref_series, prod_series)

        return FeatureExplanation(
            feature_name=feature_name,
            has_drift=has_drift,
            drift_score=drift_score,
            drift_method=drift_method,
            ref_mean=ref_mean,
            prod_mean=prod_mean,
            mean_shift=mean_shift,
            mean_shift_percent=mean_shift_percent,
            ref_std=ref_std,
            prod_std=prod_std,
            std_change=std_change,
            std_change_percent=std_change_percent,
            ref_min=ref_min,
            prod_min=prod_min,
            ref_max=ref_max,
            prod_max=prod_max,
            quantile_stats=quantile_stats,
            ref_count=len(ref_series),
            prod_count=len(prod_series),
        )

    def _compute_quantile_stats(
        self, ref_series: pd.Series, prod_series: pd.Series
    ) -> QuantileStats:
        """Compute quantile comparison statistics."""
        reference_values: dict[float, float] = {}
        production_values: dict[float, float] = {}
        absolute_diffs: dict[float, float] = {}
        relative_diffs: dict[float, float] = {}

        for q in self.quantiles:
            ref_val = float(ref_series.quantile(q))
            prod_val = float(prod_series.quantile(q))

            reference_values[q] = ref_val
            production_values[q] = prod_val
            absolute_diffs[q] = prod_val - ref_val
            relative_diffs[q] = self._safe_percent_change(ref_val, prod_val)

        return QuantileStats(
            quantiles=self.quantiles,
            reference_values=reference_values,
            production_values=production_values,
            absolute_diffs=absolute_diffs,
            relative_diffs=relative_diffs,
        )

    @staticmethod
    def _safe_percent_change(old: float, new: float) -> float:
        """Calculate percent change safely handling zero division."""
        if old == 0:
            if new == 0:
                return 0.0
            return float("inf") if new > 0 else float("-inf")
        return ((new - old) / abs(old)) * 100
