"""Tests for numerical drift detectors."""

import numpy as np
import pandas as pd
import pytest

from driftwatch.detectors.numerical import KSDetector, PSIDetector, WassersteinDetector


class TestKSDetector:
    """Tests for Kolmogorov-Smirnov detector."""

    @pytest.fixture
    def detector(self) -> KSDetector:
        return KSDetector(threshold=0.05)

    def test_no_drift_identical_data(self, detector: KSDetector) -> None:
        """Should not detect drift for identical distributions."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 1000))

        result = detector.detect(data, data)

        assert not result.has_drift
        assert result.p_value is not None
        assert result.p_value > detector.threshold

    def test_drift_different_distributions(self, detector: KSDetector) -> None:
        """Should detect drift for different distributions."""
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0, 1, 1000))
        production = pd.Series(np.random.normal(3, 1, 1000))  # Shifted mean

        result = detector.detect(reference, production)

        assert result.has_drift
        assert result.p_value is not None
        assert result.p_value < detector.threshold

    def test_empty_reference_raises(self, detector: KSDetector) -> None:
        """Should raise ValueError for empty reference data."""
        with pytest.raises(ValueError, match="Reference series cannot be empty"):
            detector.detect(pd.Series(dtype=float), pd.Series([1, 2, 3]))

    def test_empty_production_raises(self, detector: KSDetector) -> None:
        """Should raise ValueError for empty production data."""
        with pytest.raises(ValueError, match="Production series cannot be empty"):
            detector.detect(pd.Series([1, 2, 3]), pd.Series(dtype=float))


class TestPSIDetector:
    """Tests for Population Stability Index detector."""

    @pytest.fixture
    def detector(self) -> PSIDetector:
        return PSIDetector(threshold=0.2, buckets=10)

    def test_no_drift_identical_data(self, detector: PSIDetector) -> None:
        """PSI should be 0 for identical distributions."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 1000))

        result = detector.detect(data, data)

        assert not result.has_drift
        assert result.score < detector.threshold

    def test_drift_shifted_distribution(self, detector: PSIDetector) -> None:
        """PSI should detect significant distribution shift."""
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0, 1, 1000))
        production = pd.Series(np.random.normal(2, 2, 1000))

        result = detector.detect(reference, production)

        assert result.has_drift
        assert result.score >= detector.threshold

    def test_psi_interpretation(self, detector: PSIDetector) -> None:
        """Test PSI score interpretation."""
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0, 1, 1000))

        # Small shift - should be < 0.1
        small_shift = pd.Series(np.random.normal(0.1, 1, 1000))
        result_small = detector.detect(reference, small_shift)
        assert result_small.score < 0.1, "Small shift should have PSI < 0.1"


class TestWassersteinDetector:
    """Tests for Wasserstein distance detector."""

    @pytest.fixture
    def detector(self) -> WassersteinDetector:
        return WassersteinDetector(threshold=0.5)

    def test_no_drift_identical_data(self, detector: WassersteinDetector) -> None:
        """Distance should be 0 for identical data."""
        data = pd.Series([1, 2, 3, 4, 5])

        result = detector.detect(data, data)

        assert not result.has_drift
        assert result.score == 0.0

    def test_drift_shifted_distribution(self, detector: WassersteinDetector) -> None:
        """Should detect significant distribution shift."""
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0, 1, 1000))
        production = pd.Series(np.random.normal(2, 1, 1000))  # 2 std shift

        result = detector.detect(reference, production)

        assert result.has_drift
