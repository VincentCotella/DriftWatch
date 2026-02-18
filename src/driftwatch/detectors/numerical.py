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


class JensenShannonDetector(BaseDetector):
    """
    Jensen-Shannon divergence for numerical drift detection.

    A symmetric and bounded (0-1) measure of similarity between
    two probability distributions. Based on the KL divergence but
    always finite and symmetric, making it more robust.

    Args:
        threshold: JSD value above which drift is detected.
            Default is 0.1 (range: 0 to ln(2) ≈ 0.693 for base-e,
            or 0 to 1 for base-2).
        buckets: Number of buckets for binning. Default is 50.
        base: Logarithm base (2 gives range [0,1]). Default is 2.

    Example:
        >>> detector = JensenShannonDetector(threshold=0.1)
        >>> result = detector.detect(reference_series, production_series)
        >>> print(f"JSD score: {result.score:.4f}")
    """

    def __init__(
        self,
        threshold: float = 0.1,
        buckets: int = 50,
        base: float = 2.0,
    ) -> None:
        super().__init__(threshold=threshold, name="jensen_shannon")
        self.buckets = buckets
        self.base = base

    def detect(
        self,
        reference: pd.Series,
        production: pd.Series,
    ) -> DetectionResult:
        """
        Calculate Jensen-Shannon divergence between distributions.

        The JSD is computed by binning both distributions and comparing
        their probability mass functions.

        Returns:
            DetectionResult with JSD score (0 = identical, 1 = maximally different)
        """
        self._validate_inputs(reference, production)

        ref_clean = reference.dropna().values
        prod_clean = production.dropna().values

        jsd = self._calculate_jsd(
            np.asarray(ref_clean),
            np.asarray(prod_clean),
        )

        return DetectionResult(
            has_drift=jsd >= self.threshold,
            score=float(jsd),
            method=self.name,
            threshold=self.threshold,
            p_value=None,
        )

    def _calculate_jsd(
        self,
        reference: np.ndarray,
        production: np.ndarray,
    ) -> float:
        """
        Calculate Jensen-Shannon divergence using histogram binning.

        JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
        where M = 0.5 * (P + Q)
        """
        # Determine bin edges from combined data
        combined = np.concatenate([reference, production])
        bin_edges = np.histogram_bin_edges(combined, bins=self.buckets)

        # Compute normalized histograms
        ref_hist = np.histogram(reference, bins=bin_edges)[0].astype(float)
        prod_hist = np.histogram(production, bins=bin_edges)[0].astype(float)

        # Normalize to probability distributions
        ref_prob = ref_hist / ref_hist.sum()
        prod_prob = prod_hist / prod_hist.sum()

        # Avoid log(0) by adding small epsilon
        eps = 1e-12
        ref_prob = np.clip(ref_prob, eps, 1.0)
        prod_prob = np.clip(prod_prob, eps, 1.0)

        # Mixture distribution M = 0.5 * (P + Q)
        m_prob = 0.5 * (ref_prob + prod_prob)

        # KL divergences
        kl_pm: float = float(np.sum(ref_prob * np.log(ref_prob / m_prob)))
        kl_qm: float = float(np.sum(prod_prob * np.log(prod_prob / m_prob)))

        # Jensen-Shannon divergence
        jsd = 0.5 * kl_pm + 0.5 * kl_qm

        # Convert to base-2 if requested (gives [0, 1] range)
        if self.base != np.e:
            jsd = jsd / np.log(self.base)

        return float(max(0.0, jsd))  # Ensure non-negative due to floating point


class AndersonDarlingDetector(BaseDetector):
    """
    Anderson-Darling test for numerical drift detection.

    A modification of the KS test that gives more weight to the tails
    of the distribution, making it more sensitive to tail differences.

    Uses the two-sample Anderson-Darling test from scipy.

    Args:
        threshold: Significance level below which drift is detected.
            Default is 0.05 (5% significance level).

    Example:
        >>> detector = AndersonDarlingDetector(threshold=0.05)
        >>> result = detector.detect(reference_series, production_series)
        >>> print(f"Drift detected: {result.has_drift}")
    """

    def __init__(self, threshold: float = 0.05) -> None:
        super().__init__(threshold=threshold, name="anderson_darling")

    def detect(
        self,
        reference: pd.Series,
        production: pd.Series,
    ) -> DetectionResult:
        """
        Perform two-sample Anderson-Darling test.

        Returns:
            DetectionResult with AD statistic as score and
            approximate significance level
        """
        self._validate_inputs(reference, production)

        ref_clean = reference.dropna().values
        prod_clean = production.dropna().values

        result = stats.anderson_ksamp([ref_clean, prod_clean])
        statistic = float(result.statistic)
        # anderson_ksamp returns pvalue directly (scipy >= 1.7)
        p_value = float(result.pvalue)

        return DetectionResult(
            has_drift=p_value < self.threshold,
            score=statistic,
            method=self.name,
            threshold=self.threshold,
            p_value=p_value,
        )


class CramerVonMisesDetector(BaseDetector):
    """
    Cramér-von Mises test for numerical drift detection.

    Unlike the KS test which uses the maximum CDF difference,
    the CvM test integrates the squared differences between
    the CDFs, making it sensitive to overall distributional changes.

    Args:
        threshold: P-value threshold below which drift is detected.
            Default is 0.05 (95% confidence).

    Example:
        >>> detector = CramerVonMisesDetector(threshold=0.05)
        >>> result = detector.detect(reference_series, production_series)
        >>> print(f"CvM statistic: {result.score:.4f}")
    """

    def __init__(self, threshold: float = 0.05) -> None:
        super().__init__(threshold=threshold, name="cramer_von_mises")

    def detect(
        self,
        reference: pd.Series,
        production: pd.Series,
    ) -> DetectionResult:
        """
        Perform two-sample Cramér-von Mises test.

        Returns:
            DetectionResult with CvM statistic and p-value
        """
        self._validate_inputs(reference, production)

        ref_clean = reference.dropna().values
        prod_clean = production.dropna().values

        result = stats.cramervonmises_2samp(ref_clean, prod_clean)
        statistic = float(result.statistic)
        p_value = float(result.pvalue)

        return DetectionResult(
            has_drift=p_value < self.threshold,
            score=statistic,
            method=self.name,
            threshold=self.threshold,
            p_value=p_value,
        )
