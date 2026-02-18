"""Tests for new numerical drift detectors (Jensen-Shannon, Anderson-Darling, Cramér-von Mises)."""

import numpy as np
import pandas as pd
import pytest

from driftwatch.detectors.numerical import (
    AndersonDarlingDetector,
    CramerVonMisesDetector,
    JensenShannonDetector,
)


class TestJensenShannonDetector:
    """Tests for Jensen-Shannon divergence detector."""

    @pytest.fixture
    def detector(self) -> JensenShannonDetector:
        return JensenShannonDetector(threshold=0.1)

    def test_no_drift_identical_data(self, detector: JensenShannonDetector) -> None:
        """JSD should be ~0 for identical distributions."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 1000))

        result = detector.detect(data, data)

        assert not result.has_drift
        assert result.score < 0.01  # Should be very close to 0
        assert result.method == "jensen_shannon"

    def test_drift_different_distributions(
        self, detector: JensenShannonDetector
    ) -> None:
        """JSD should detect drift for shifted distributions."""
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0, 1, 1000))
        production = pd.Series(np.random.normal(3, 1, 1000))

        result = detector.detect(reference, production)

        assert result.has_drift
        assert result.score >= detector.threshold

    def test_jsd_bounded(self, detector: JensenShannonDetector) -> None:
        """JSD with base 2 should be bounded between 0 and 1."""
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0, 1, 1000))
        production = pd.Series(np.random.normal(10, 1, 1000))

        result = detector.detect(reference, production)

        assert 0.0 <= result.score <= 1.0

    def test_jsd_symmetric(self, detector: JensenShannonDetector) -> None:
        """JSD should be symmetric: JSD(P||Q) == JSD(Q||P)."""
        np.random.seed(42)
        ref = pd.Series(np.random.normal(0, 1, 1000))
        prod = pd.Series(np.random.normal(1, 1, 1000))

        result_forward = detector.detect(ref, prod)
        result_backward = detector.detect(prod, ref)

        assert abs(result_forward.score - result_backward.score) < 0.01

    def test_jsd_custom_buckets(self) -> None:
        """Test JSD with custom number of buckets."""
        detector = JensenShannonDetector(threshold=0.1, buckets=20)
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0, 1, 1000))
        production = pd.Series(np.random.normal(2, 1, 1000))

        result = detector.detect(reference, production)

        assert result.has_drift
        assert result.p_value is None

    def test_jsd_small_shift(self) -> None:
        """Small shift should result in low JSD."""
        detector = JensenShannonDetector(threshold=0.1)
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0, 1, 2000))
        production = pd.Series(np.random.normal(0.1, 1, 2000))

        result = detector.detect(reference, production)

        assert result.score < 0.1

    def test_empty_reference_raises(self, detector: JensenShannonDetector) -> None:
        """Should raise ValueError for empty reference data."""
        with pytest.raises(ValueError, match="Reference series cannot be empty"):
            detector.detect(pd.Series(dtype=float), pd.Series([1, 2, 3]))

    def test_empty_production_raises(self, detector: JensenShannonDetector) -> None:
        """Should raise ValueError for empty production data."""
        with pytest.raises(ValueError, match="Production series cannot be empty"):
            detector.detect(pd.Series([1, 2, 3]), pd.Series(dtype=float))


class TestAndersonDarlingDetector:
    """Tests for Anderson-Darling detector."""

    @pytest.fixture
    def detector(self) -> AndersonDarlingDetector:
        return AndersonDarlingDetector(threshold=0.05)

    def test_no_drift_identical_data(self, detector: AndersonDarlingDetector) -> None:
        """Should not detect drift for identical distributions."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 1000))

        result = detector.detect(data, data)

        assert not result.has_drift
        assert result.p_value is not None
        assert result.p_value > detector.threshold
        assert result.method == "anderson_darling"

    def test_drift_different_distributions(
        self, detector: AndersonDarlingDetector
    ) -> None:
        """Should detect drift for clearly different distributions."""
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0, 1, 1000))
        production = pd.Series(np.random.normal(3, 1, 1000))

        result = detector.detect(reference, production)

        assert result.has_drift
        assert result.p_value is not None
        assert result.p_value < detector.threshold

    def test_tail_sensitivity(self, detector: AndersonDarlingDetector) -> None:
        """AD test should be more sensitive to tail differences."""
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0, 1, 2000))
        # Same center but heavier tails (t-distribution with low df)
        production = pd.Series(np.random.standard_t(3, 2000))

        result = detector.detect(reference, production)

        # AD should detect this tail difference
        assert result.has_drift

    def test_empty_reference_raises(self, detector: AndersonDarlingDetector) -> None:
        """Should raise ValueError for empty reference data."""
        with pytest.raises(ValueError, match="Reference series cannot be empty"):
            detector.detect(pd.Series(dtype=float), pd.Series([1, 2, 3]))

    def test_empty_production_raises(self, detector: AndersonDarlingDetector) -> None:
        """Should raise ValueError for empty production data."""
        with pytest.raises(ValueError, match="Production series cannot be empty"):
            detector.detect(pd.Series([1, 2, 3]), pd.Series(dtype=float))


class TestCramerVonMisesDetector:
    """Tests for Cramér-von Mises detector."""

    @pytest.fixture
    def detector(self) -> CramerVonMisesDetector:
        return CramerVonMisesDetector(threshold=0.05)

    def test_no_drift_identical_data(self, detector: CramerVonMisesDetector) -> None:
        """Should not detect drift for identical distributions."""
        np.random.seed(42)
        data = pd.Series(np.random.normal(0, 1, 500))

        result = detector.detect(data, data)

        assert not result.has_drift
        assert result.p_value is not None
        assert result.p_value > detector.threshold
        assert result.method == "cramer_von_mises"

    def test_drift_different_distributions(
        self, detector: CramerVonMisesDetector
    ) -> None:
        """Should detect drift for different distributions."""
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0, 1, 1000))
        production = pd.Series(np.random.normal(3, 1, 1000))

        result = detector.detect(reference, production)

        assert result.has_drift
        assert result.p_value is not None
        assert result.p_value < detector.threshold

    def test_overall_shape_sensitivity(self, detector: CramerVonMisesDetector) -> None:
        """CvM should detect overall distribution shape changes."""
        np.random.seed(42)
        reference = pd.Series(np.random.normal(0, 1, 1000))
        # Different variance (wider distribution)
        production = pd.Series(np.random.normal(0, 3, 1000))

        result = detector.detect(reference, production)

        assert result.has_drift

    def test_empty_reference_raises(self, detector: CramerVonMisesDetector) -> None:
        """Should raise ValueError for empty reference data."""
        with pytest.raises(ValueError, match="Reference series cannot be empty"):
            detector.detect(pd.Series(dtype=float), pd.Series([1, 2, 3]))

    def test_empty_production_raises(self, detector: CramerVonMisesDetector) -> None:
        """Should raise ValueError for empty production data."""
        with pytest.raises(ValueError, match="Production series cannot be empty"):
            detector.detect(pd.Series([1, 2, 3]), pd.Series(dtype=float))


class TestRegistryWithNewDetectors:
    """Test that new detectors are properly registered."""

    def test_get_jensen_shannon_by_name(self) -> None:
        """Jensen-Shannon should be available by name."""
        from driftwatch.detectors.registry import get_detector_by_name

        detector = get_detector_by_name("jensen_shannon", {"jensen_shannon": 0.15})
        assert isinstance(detector, JensenShannonDetector)
        assert detector.threshold == 0.15

    def test_get_anderson_darling_by_name(self) -> None:
        """Anderson-Darling should be available by name."""
        from driftwatch.detectors.registry import get_detector_by_name

        detector = get_detector_by_name(
            "anderson_darling", {"anderson_darling_pvalue": 0.01}
        )
        assert isinstance(detector, AndersonDarlingDetector)
        assert detector.threshold == 0.01

    def test_get_cramer_von_mises_by_name(self) -> None:
        """Cramér-von Mises should be available by name."""
        from driftwatch.detectors.registry import get_detector_by_name

        detector = get_detector_by_name(
            "cramer_von_mises", {"cramer_von_mises_pvalue": 0.01}
        )
        assert isinstance(detector, CramerVonMisesDetector)
        assert detector.threshold == 0.01
