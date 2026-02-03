"""Tests for categorical drift detectors."""

import numpy as np
import pandas as pd
import pytest

from driftwatch.detectors.categorical import ChiSquaredDetector, FrequencyPSIDetector


class TestChiSquaredDetector:
    """Tests for Chi-Squared detector."""

    @pytest.fixture
    def detector(self) -> ChiSquaredDetector:
        return ChiSquaredDetector(threshold=0.05)

    def test_no_drift_identical_data(self, detector: ChiSquaredDetector) -> None:
        """Should not detect drift for identical distributions."""
        data = pd.Series(["A", "B", "C", "A", "B", "A"] * 100)

        result = detector.detect(data, data)

        assert not result.has_drift
        assert result.p_value is not None
        assert result.p_value > detector.threshold

    def test_drift_different_distributions(self, detector: ChiSquaredDetector) -> None:
        """Should detect drift for different category distributions."""
        np.random.seed(42)
        reference = pd.Series(
            np.random.choice(["A", "B", "C"], 1000, p=[0.7, 0.2, 0.1])
        )
        production = pd.Series(
            np.random.choice(["A", "B", "C"], 1000, p=[0.2, 0.5, 0.3])
        )

        result = detector.detect(reference, production)

        assert result.has_drift
        assert result.p_value is not None
        assert result.p_value < detector.threshold

    def test_empty_reference_raises(self, detector: ChiSquaredDetector) -> None:
        """Should raise ValueError for empty reference data."""
        with pytest.raises(ValueError, match="Reference series cannot be empty"):
            detector.detect(pd.Series(dtype=str), pd.Series(["A", "B"]))

    def test_empty_production_raises(self, detector: ChiSquaredDetector) -> None:
        """Should raise ValueError for empty production data."""
        with pytest.raises(ValueError, match="Production series cannot be empty"):
            detector.detect(pd.Series(["A", "B"]), pd.Series(dtype=str))

    def test_new_category_in_production(self, detector: ChiSquaredDetector) -> None:
        """Should handle new categories appearing in production."""
        reference = pd.Series(["A", "B"] * 100)
        production = pd.Series(["A", "B", "C"] * 67)  # New category C

        result = detector.detect(reference, production)

        # Should complete without error
        assert result.method == "chi_squared"

    def test_missing_category_in_production(self, detector: ChiSquaredDetector) -> None:
        """Should handle categories missing in production."""
        reference = pd.Series(["A", "B", "C"] * 100)
        production = pd.Series(["A", "B"] * 150)  # Missing category C

        result = detector.detect(reference, production)

        assert result.method == "chi_squared"


class TestFrequencyPSIDetector:
    """Tests for Frequency PSI detector."""

    @pytest.fixture
    def detector(self) -> FrequencyPSIDetector:
        return FrequencyPSIDetector(threshold=0.2)

    def test_no_drift_identical_data(self, detector: FrequencyPSIDetector) -> None:
        """PSI should be ~0 for identical distributions."""
        data = pd.Series(["A", "B", "C"] * 100)

        result = detector.detect(data, data)

        assert not result.has_drift
        assert result.score < 0.01  # Should be very close to 0

    def test_drift_different_distributions(
        self, detector: FrequencyPSIDetector
    ) -> None:
        """Should detect drift for different category distributions."""
        np.random.seed(42)
        reference = pd.Series(
            np.random.choice(["A", "B", "C"], 1000, p=[0.7, 0.2, 0.1])
        )
        production = pd.Series(
            np.random.choice(["A", "B", "C"], 1000, p=[0.2, 0.5, 0.3])
        )

        result = detector.detect(reference, production)

        assert result.has_drift
        assert result.score >= detector.threshold

    def test_psi_value_interpretation(self, detector: FrequencyPSIDetector) -> None:
        """Test PSI score is reasonable."""
        np.random.seed(42)
        reference = pd.Series(np.random.choice(["A", "B"], 1000, p=[0.5, 0.5]))
        # Small shift
        production = pd.Series(np.random.choice(["A", "B"], 1000, p=[0.55, 0.45]))

        result = detector.detect(reference, production)

        # Small shift should have small PSI
        assert result.score < 0.1
