"""Detector registry for automatic selection based on dtype."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from driftwatch.detectors.categorical import ChiSquaredDetector
from driftwatch.detectors.numerical import (
    AndersonDarlingDetector,
    CramerVonMisesDetector,
    JensenShannonDetector,
    KSDetector,
    PSIDetector,
    WassersteinDetector,
)

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
    import pandas as pd

    # Handle pandas CategoricalDtype explicitly
    if isinstance(dtype, pd.CategoricalDtype):
        return ChiSquaredDetector(threshold=thresholds.get("chi2_pvalue", 0.05))

    # Handle pandas StringDtype explicitly
    if isinstance(dtype, pd.StringDtype):
        return ChiSquaredDetector(threshold=thresholds.get("chi2_pvalue", 0.05))

    # Handle object dtype (strings, mixed types)
    if dtype == np.object_ or dtype.name == "object":
        return ChiSquaredDetector(threshold=thresholds.get("chi2_pvalue", 0.05))

    # Handle string-like dtype names (e.g., 'string', 'String')
    if hasattr(dtype, "name") and dtype.name.lower().startswith("string"):
        return ChiSquaredDetector(threshold=thresholds.get("chi2_pvalue", 0.05))

    # Handle numerical types
    try:
        if np.issubdtype(dtype, np.number):
            # Use PSI for numerical features by default
            return PSIDetector(threshold=thresholds.get("psi", 0.2))
    except TypeError:
        # If issubdtype fails, treat as categorical
        pass

    # Default to categorical for any other type
    return ChiSquaredDetector(threshold=thresholds.get("chi2_pvalue", 0.05))


def get_detector_by_name(
    name: str,
    thresholds: dict[str, float],
) -> BaseDetector:
    """
    Get detector by explicit name.

    Args:
        name: Detector name ("ks", "psi", "wasserstein", "chi2",
              "jensen_shannon", "anderson_darling", "cramer_von_mises")
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
        "jensen_shannon": lambda: JensenShannonDetector(
            threshold=thresholds.get("jensen_shannon", 0.1)
        ),
        "anderson_darling": lambda: AndersonDarlingDetector(
            threshold=thresholds.get("anderson_darling_pvalue", 0.05)
        ),
        "cramer_von_mises": lambda: CramerVonMisesDetector(
            threshold=thresholds.get("cramer_von_mises_pvalue", 0.05)
        ),
    }

    if name not in detectors:
        available = ", ".join(detectors.keys())
        raise ValueError(f"Unknown detector '{name}'. Available: {available}")

    return detectors[name]()
