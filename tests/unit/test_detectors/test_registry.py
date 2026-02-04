"""Tests for detector registry."""

import numpy as np
import pytest

from driftwatch.detectors.categorical import ChiSquaredDetector
from driftwatch.detectors.numerical import KSDetector, PSIDetector, WassersteinDetector
from driftwatch.detectors.registry import get_detector, get_detector_by_name


class TestGetDetector:
    """Tests for automatic detector selection."""

    def test_numerical_dtype_returns_psi(self) -> None:
        """Should return PSI detector for numerical dtypes."""
        thresholds = {"psi": 0.15}

        detector = get_detector(np.dtype("float64"), thresholds)

        assert isinstance(detector, PSIDetector)
        assert detector.threshold == 0.15

    def test_int_dtype_returns_psi(self) -> None:
        """Should return PSI detector for integer dtypes."""
        thresholds = {"psi": 0.2}

        detector = get_detector(np.dtype("int32"), thresholds)

        assert isinstance(detector, PSIDetector)

    def test_object_dtype_returns_chi2(self) -> None:
        """Should return Chi-Squared detector for object dtypes."""
        thresholds = {"chi2_pvalue": 0.01}

        detector = get_detector(np.dtype("object"), thresholds)

        assert isinstance(detector, ChiSquaredDetector)
        assert detector.threshold == 0.01

    def test_string_dtype_returns_chi2(self) -> None:
        """Should return Chi-Squared detector for string dtypes."""
        thresholds = {}

        detector = get_detector(np.dtype("U10"), thresholds)

        assert isinstance(detector, ChiSquaredDetector)

    def test_default_thresholds(self) -> None:
        """Should use default thresholds when not specified."""
        detector = get_detector(np.dtype("float64"), {})

        assert isinstance(detector, PSIDetector)
        assert detector.threshold == 0.2  # default PSI threshold


class TestGetDetectorByName:
    """Tests for explicit detector selection by name."""

    def test_get_ks_detector(self) -> None:
        """Should return KS detector when requested."""
        thresholds = {"ks_pvalue": 0.01}

        detector = get_detector_by_name("ks", thresholds)

        assert isinstance(detector, KSDetector)
        assert detector.threshold == 0.01

    def test_get_psi_detector(self) -> None:
        """Should return PSI detector when requested."""
        thresholds = {"psi": 0.15}

        detector = get_detector_by_name("psi", thresholds)

        assert isinstance(detector, PSIDetector)
        assert detector.threshold == 0.15

    def test_get_chi2_detector(self) -> None:
        """Should return Chi-Squared detector when requested."""
        thresholds = {"chi2_pvalue": 0.1}

        detector = get_detector_by_name("chi2", thresholds)

        assert isinstance(detector, ChiSquaredDetector)
        assert detector.threshold == 0.1

    def test_get_wasserstein_detector(self) -> None:
        """Should return Wasserstein detector when requested."""
        thresholds = {"wasserstein": 0.2}

        detector = get_detector_by_name("wasserstein", thresholds)

        assert isinstance(detector, WassersteinDetector)
        assert detector.threshold == 0.2

    def test_unknown_detector_raises(self) -> None:
        """Should raise ValueError for unknown detector name."""
        with pytest.raises(ValueError, match="Unknown detector 'invalid'"):
            get_detector_by_name("invalid", {})

    def test_error_message_shows_available(self) -> None:
        """Error message should show available detectors."""
        with pytest.raises(ValueError) as exc_info:
            get_detector_by_name("invalid", {})

        error_msg = str(exc_info.value)
        assert "ks" in error_msg
        assert "psi" in error_msg
        assert "chi2" in error_msg
        assert "wasserstein" in error_msg
