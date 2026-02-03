"""Base class for drift detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class DetectionResult:
    """Result from a drift detection test."""

    has_drift: bool
    score: float
    method: str
    threshold: float
    p_value: float | None = None


class BaseDetector(ABC):
    """
    Abstract base class for drift detectors.

    All drift detection methods should inherit from this class
    and implement the `detect` method.

    Args:
        threshold: Threshold value for determining drift
        name: Human-readable name for the detector
    """

    def __init__(self, threshold: float, name: str) -> None:
        self.threshold = threshold
        self.name = name

    @abstractmethod
    def detect(
        self,
        reference: pd.Series,
        production: pd.Series,
    ) -> DetectionResult:
        """
        Detect drift between reference and production data.

        Args:
            reference: Reference data series
            production: Production data series

        Returns:
            DetectionResult with drift status and metrics
        """
        ...

    def _validate_inputs(
        self,
        reference: pd.Series,
        production: pd.Series,
    ) -> None:
        """Validate input series are not empty."""
        if reference.empty:
            raise ValueError("Reference series cannot be empty")
        if production.empty:
            raise ValueError("Production series cannot be empty")
