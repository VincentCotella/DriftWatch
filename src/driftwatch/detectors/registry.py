"""Detector registry for automatic selection based on dtype."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from driftwatch.detectors.categorical import ChiSquaredDetector
from driftwatch.detectors.numerical import KSDetector, PSIDetector, WassersteinDetector

if TYPE_CHECKING:
    from driftwatch.detectors.base import BaseDetector


def get_detector(dtype: np.dtype[Any], thresholds: dict[str, float]) -> BaseDetector:
    """
    Get appropriate detector based on data type.

    Args:
        dtype: NumPy dtype of the feature
        thresholds: Dictionary of threshold values

    Returns:
        Appropriate detector instance

    Note:
        - Numerical types use PSI by default
        - Categorical/object types use Chi-Squared
    """
    if np.issubdtype(dtype, np.number):
        # Use PSI for numerical features by default
        return PSIDetector(threshold=thresholds.get("psi", 0.2))
    else:
        # Use Chi-Squared for categorical features
        return ChiSquaredDetector(threshold=thresholds.get("chi2_pvalue", 0.05))


def get_detector_by_name(
    name: str,
    thresholds: dict[str, float],
) -> BaseDetector:
    """
    Get detector by explicit name.

    Args:
        name: Detector name ("ks", "psi", "wasserstein", "chi2")
        thresholds: Dictionary of threshold values

    Returns:
        Requested detector instance

    Raises:
        ValueError: If detector name is unknown
    """
    detectors = {
        "ks": lambda: KSDetector(threshold=thresholds.get("ks_pvalue", 0.05)),
        "psi": lambda: PSIDetector(threshold=thresholds.get("psi", 0.2)),
        "wasserstein": lambda: WassersteinDetector(
            threshold=thresholds.get("wasserstein", 0.1)
        ),
        "chi2": lambda: ChiSquaredDetector(
            threshold=thresholds.get("chi2_pvalue", 0.05)
        ),
    }

    if name not in detectors:
        available = ", ".join(detectors.keys())
        raise ValueError(f"Unknown detector '{name}'. Available: {available}")

    return detectors[name]()
