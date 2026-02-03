"""Categorical feature drift detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from driftwatch.detectors.base import BaseDetector, DetectionResult


class ChiSquaredDetector(BaseDetector):
    """
    Chi-Squared test for categorical drift detection.

    Tests whether the frequency distribution of categories
    has changed between reference and production data.

    Args:
        threshold: P-value threshold below which drift is detected.
            Default is 0.05 (95% confidence).

    Example:
        >>> detector = ChiSquaredDetector(threshold=0.05)
        >>> result = detector.detect(reference_series, production_series)
    """

    def __init__(self, threshold: float = 0.05) -> None:
        super().__init__(threshold=threshold, name="chi_squared")

    def detect(
        self,
        reference: pd.Series,
        production: pd.Series,
    ) -> DetectionResult:
        """
        Perform Chi-Squared test on category frequencies.

        Returns:
            DetectionResult with chi-squared statistic and p-value
        """
        self._validate_inputs(reference, production)

        # Get all categories from both datasets
        all_categories = set(reference.dropna().unique()) | set(
            production.dropna().unique()
        )

        # Count frequencies
        ref_counts = reference.value_counts()
        prod_counts = production.value_counts()

        # Align to same categories
        ref_freq = np.array([ref_counts.get(cat, 0) for cat in all_categories])
        prod_freq = np.array([prod_counts.get(cat, 0) for cat in all_categories])

        # Handle edge case of zero frequencies
        if ref_freq.sum() == 0 or prod_freq.sum() == 0:
            return DetectionResult(
                has_drift=True,
                score=float("inf"),
                method=self.name,
                threshold=self.threshold,
                p_value=0.0,
            )

        # Calculate expected frequencies based on reference proportions
        ref_proportions = ref_freq / ref_freq.sum()
        expected = ref_proportions * prod_freq.sum()

        # Add small epsilon to avoid division by zero
        expected = np.maximum(expected, 1e-10)

        # Chi-squared statistic
        statistic, p_value = stats.chisquare(prod_freq, f_exp=expected)

        return DetectionResult(
            has_drift=p_value < self.threshold,
            score=float(statistic),
            method=self.name,
            threshold=self.threshold,
            p_value=float(p_value),
        )


class FrequencyPSIDetector(BaseDetector):
    """
    PSI-based detector for categorical features.

    Calculates PSI using category frequency distributions
    instead of numerical buckets.

    Args:
        threshold: PSI value above which drift is detected.
            Default is 0.2.
    """

    def __init__(self, threshold: float = 0.2) -> None:
        super().__init__(threshold=threshold, name="frequency_psi")

    def detect(
        self,
        reference: pd.Series,
        production: pd.Series,
    ) -> DetectionResult:
        """
        Calculate PSI on category frequencies.

        Returns:
            DetectionResult with PSI score
        """
        self._validate_inputs(reference, production)

        # Get normalized frequencies
        ref_freq = reference.value_counts(normalize=True)
        prod_freq = production.value_counts(normalize=True)

        # Get all categories
        all_categories = set(ref_freq.index) | set(prod_freq.index)

        # Calculate PSI
        eps = 1e-10
        psi = 0.0

        for cat in all_categories:
            ref_pct = ref_freq.get(cat, eps)
            prod_pct = prod_freq.get(cat, eps)

            # Clip to avoid log(0)
            ref_pct = max(ref_pct, eps)
            prod_pct = max(prod_pct, eps)

            psi += (prod_pct - ref_pct) * np.log(prod_pct / ref_pct)

        return DetectionResult(
            has_drift=psi >= self.threshold,
            score=float(psi),
            method=self.name,
            threshold=self.threshold,
            p_value=None,
        )
