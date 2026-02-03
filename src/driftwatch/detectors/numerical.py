"""Numerical feature drift detectors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from driftwatch.detectors.base import BaseDetector, DetectionResult

if TYPE_CHECKING:
    import pandas as pd


class KSDetector(BaseDetector):
    """
    Kolmogorov-Smirnov test for numerical drift detection.

    The KS test measures the maximum distance between the cumulative
    distribution functions of two samples.

    Args:
        threshold: P-value threshold below which drift is detected.
            Default is 0.05 (95% confidence).

    Example:
        >>> detector = KSDetector(threshold=0.05)
        >>> result = detector.detect(reference_series, production_series)
        >>> print(f"Drift detected: {result.has_drift}")
    """

    def __init__(self, threshold: float = 0.05) -> None:
        super().__init__(threshold=threshold, name="ks_test")

    def detect(
        self,
        reference: pd.Series,
        production: pd.Series,
    ) -> DetectionResult:
        """
        Perform KS test between reference and production distributions.

        Returns:
            DetectionResult with KS statistic as score and p-value
        """
        self._validate_inputs(reference, production)

        statistic, p_value = stats.ks_2samp(
            reference.dropna(),
            production.dropna(),
        )

        return DetectionResult(
            has_drift=p_value < self.threshold,
            score=float(statistic),
            method=self.name,
            threshold=self.threshold,
            p_value=float(p_value),
        )


class PSIDetector(BaseDetector):
    """
    Population Stability Index (PSI) for numerical drift detection.

    PSI measures the shift in distribution between two populations.
    Commonly used thresholds:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.2: Minor shift
    - PSI >= 0.2: Significant shift (drift)

    Args:
        threshold: PSI value above which drift is detected.
            Default is 0.2.
        buckets: Number of buckets for binning. Default is 10.

    Example:
        >>> detector = PSIDetector(threshold=0.2, buckets=10)
        >>> result = detector.detect(reference_series, production_series)
    """

    def __init__(self, threshold: float = 0.2, buckets: int = 10) -> None:
        super().__init__(threshold=threshold, name="psi")
        self.buckets = buckets

    def detect(
        self,
        reference: pd.Series,
        production: pd.Series,
    ) -> DetectionResult:
        """
        Calculate PSI between reference and production distributions.

        Returns:
            DetectionResult with PSI score
        """
        self._validate_inputs(reference, production)

        psi_value = self._calculate_psi(
            np.asarray(reference.dropna().values),
            np.asarray(production.dropna().values),
        )

        return DetectionResult(
            has_drift=psi_value >= self.threshold,
            score=float(psi_value),
            method=self.name,
            threshold=self.threshold,
            p_value=None,
        )

    def _calculate_psi(
        self,
        reference: np.ndarray,
        production: np.ndarray,
    ) -> float:
        """
        Calculate PSI using percentile-based buckets.

        The reference distribution defines the bucket boundaries,
        and we compare the distribution of production data across
        these same buckets.
        """
        # Create buckets based on reference quantiles
        breakpoints = np.percentile(
            reference,
            np.linspace(0, 100, self.buckets + 1),
        )
        # Ensure unique breakpoints
        breakpoints = np.unique(breakpoints)

        if len(breakpoints) < 2:
            # Not enough variation, return 0
            return 0.0

        # Calculate distribution in each bucket
        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        prod_counts = np.histogram(production, bins=breakpoints)[0]

        # Convert to percentages, avoiding division by zero
        ref_pct = ref_counts / len(reference)
        prod_pct = prod_counts / len(production)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        ref_pct = np.clip(ref_pct, eps, 1)
        prod_pct = np.clip(prod_pct, eps, 1)

        # Calculate PSI
        psi: float = float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))

        return float(psi)


class WassersteinDetector(BaseDetector):
    """
    Wasserstein distance (Earth Mover's Distance) for drift detection.

    Measures the minimum "work" required to transform one distribution
    into another. More sensitive to subtle distributional changes.

    Args:
        threshold: Distance above which drift is detected.
    """

    def __init__(self, threshold: float = 0.1) -> None:
        super().__init__(threshold=threshold, name="wasserstein")

    def detect(
        self,
        reference: pd.Series,
        production: pd.Series,
    ) -> DetectionResult:
        """
        Calculate Wasserstein distance between distributions.

        Note: Values are normalized by the reference standard deviation
        to make the threshold more interpretable.
        """
        self._validate_inputs(reference, production)

        ref_clean = reference.dropna().values
        prod_clean = production.dropna().values

        distance = stats.wasserstein_distance(ref_clean, prod_clean)

        # Normalize by reference std for interpretability
        ref_std = np.std(ref_clean)
        normalized_distance = distance / ref_std if ref_std > 0 else distance

        return DetectionResult(
            has_drift=normalized_distance >= self.threshold,
            score=float(normalized_distance),
            method=self.name,
            threshold=self.threshold,
            p_value=None,
        )
