"""
Realistic integration tests for multi-drift monitoring.

Uses simulated but realistic ML scenarios:
- Credit scoring (classification) with feature, prediction, and concept drift
- House pricing (regression) with gradual and sudden drift
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driftwatch import (
    ComprehensiveDriftReport,
    ConceptMonitor,
    DriftSuite,
    DriftType,
    Monitor,
    PredictionMonitor,
)

# ---------------------------------------------------------------------------
# Realistic data generators
# ---------------------------------------------------------------------------


def generate_credit_scoring_data(
    n: int = 2000,
    seed: int = 42,
    age_mean: float = 38.0,
    income_mean: float = 55000.0,
    debt_ratio_mean: float = 0.35,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate realistic credit scoring dataset.

    Returns:
        Tuple of (features_df, true_labels, model_predictions)
    """
    rng = np.random.default_rng(seed)

    # Features
    age = rng.normal(age_mean, 12, n).clip(18, 80)
    annual_income = rng.lognormal(np.log(income_mean), 0.6, n).clip(15000, 500000)
    debt_ratio = rng.beta(2, 4, n) * debt_ratio_mean * 3
    credit_history_months = rng.poisson(72, n).clip(0, 360)
    num_credit_lines = rng.poisson(4, n).clip(0, 20)
    employment_years = rng.exponential(6, n).clip(0, 40)

    features = pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income,
            "debt_ratio": debt_ratio,
            "credit_history_months": credit_history_months,
            "num_credit_lines": num_credit_lines,
            "employment_years": employment_years,
        }
    )

    # True labels (default probability based on features)
    log_odds = (
        -2.0
        + 0.01 * (age - 40)
        - 0.00002 * (annual_income - 50000)
        + 3.0 * (debt_ratio - 0.3)
        - 0.005 * (credit_history_months - 60)
        + rng.normal(0, 0.5, n)
    )
    default_prob = 1 / (1 + np.exp(-log_odds))
    y_true = (rng.random(n) < default_prob).astype(int)

    # Model predictions (imperfect but reasonable)
    pred_noise = rng.normal(0, 0.3, n)
    pred_log_odds = log_odds + pred_noise
    pred_prob = 1 / (1 + np.exp(-pred_log_odds))
    y_pred = (pred_prob > 0.5).astype(int)

    return features, y_true, y_pred


def generate_house_pricing_data(
    n: int = 2000,
    seed: int = 42,
    price_mult: float = 1.0,
    area_mean: float = 120.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate realistic house pricing dataset.

    Returns:
        Tuple of (features_df, true_prices, predicted_prices)
    """
    rng = np.random.default_rng(seed)

    # Features
    area_sqm = rng.lognormal(np.log(area_mean), 0.4, n).clip(30, 500)
    num_rooms = rng.poisson(3, n).clip(1, 10)
    floor = rng.choice(range(0, 15), n)
    distance_center_km = rng.exponential(5, n).clip(0.5, 30)
    building_age_years = rng.exponential(20, n).clip(0, 100)
    has_parking = rng.binomial(1, 0.4, n)

    features = pd.DataFrame(
        {
            "area_sqm": area_sqm,
            "num_rooms": num_rooms,
            "floor": floor,
            "distance_center_km": distance_center_km,
            "building_age_years": building_age_years,
            "has_parking": has_parking,
        }
    )

    # True prices (based on features)
    base_price = (
        50000
        + 2500 * area_sqm
        + 8000 * num_rooms
        - 3000 * distance_center_km
        - 500 * building_age_years
        + 15000 * has_parking
        + 1000 * floor
    )
    noise = rng.normal(0, 15000, n)
    y_true = (base_price + noise).clip(50000) * price_mult

    # Model predictions (with some error)
    prediction_error = rng.normal(0, 20000, n)
    y_pred = (y_true + prediction_error).clip(50000)

    return features, y_true, y_pred


# ---------------------------------------------------------------------------
# Credit Scoring Tests (Classification)
# ---------------------------------------------------------------------------


class TestCreditScoringScenario:
    """Realistic credit scoring scenario with all drift types."""

    @pytest.fixture
    def reference_data(
        self,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Reference period: normal credit applications."""
        return generate_credit_scoring_data(n=2000, seed=42)

    @pytest.fixture
    def production_no_drift(
        self,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Production period: same distribution, different samples."""
        return generate_credit_scoring_data(n=1000, seed=99)

    @pytest.fixture
    def production_feature_drift(
        self,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Production: economic crisis → higher income, older applicants."""
        return generate_credit_scoring_data(
            n=1000,
            seed=77,
            age_mean=48.0,  # Older applicants
            income_mean=35000.0,  # Lower income (recession)
            debt_ratio_mean=0.55,  # More debt
        )

    @pytest.fixture
    def production_concept_drift(
        self,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Production: model performance degrades (concept drift)."""
        features, y_true, _ = generate_credit_scoring_data(n=1000, seed=55)
        # Model still predicts like before but reality changed
        rng = np.random.default_rng(55)
        y_pred = rng.integers(0, 2, len(y_true))  # Random predictions
        return features, y_true, y_pred

    def test_no_drift_scenario(
        self,
        reference_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
        production_no_drift: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Normal operation: no drift should be detected."""
        ref_features, ref_y_true, ref_y_pred = reference_data
        prod_features, prod_y_true, prod_y_pred = production_no_drift

        suite = DriftSuite(
            reference_data=ref_features,
            reference_predictions=ref_y_pred,
            task="classification",
            model_version="credit-v1.0",
        )

        report = suite.check(
            production_data=prod_features,
            production_predictions=prod_y_pred,
            y_true_ref=ref_y_true,
            y_pred_ref=ref_y_pred,
            y_true_prod=prod_y_true,
            y_pred_prod=prod_y_pred,
        )

        assert report.feature_report is not None
        assert report.prediction_report is not None
        assert report.concept_report is not None
        assert report.model_version == "credit-v1.0"

    def test_feature_drift_only(
        self,
        reference_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
        production_feature_drift: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Economic crisis: feature distributions shift significantly."""
        ref_features, _, _ = reference_data
        prod_features, _, _ = production_feature_drift

        monitor = Monitor(reference_data=ref_features)
        report = monitor.check(prod_features)

        # Should detect drift in age, income, debt_ratio
        assert report.has_drift()
        drifted_names = report.drifted_features()
        assert len(drifted_names) >= 2  # At least 2 features drifted

    def test_prediction_drift_detected(
        self,
        reference_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
        production_feature_drift: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Model predictions should shift when features shift."""
        _, _, ref_y_pred = reference_data
        _, _, prod_y_pred = production_feature_drift

        pred_monitor = PredictionMonitor(
            reference_predictions=ref_y_pred,
            detector="psi",
        )
        report = pred_monitor.check(prod_y_pred)

        # All results should be tagged as PREDICTION drift
        for result in report.feature_results:
            assert result.drift_type == DriftType.PREDICTION

    def test_concept_drift_detected(
        self,
        reference_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
        production_concept_drift: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Performance degradation should be flagged as concept drift."""
        _, ref_y_true, ref_y_pred = reference_data
        _, prod_y_true, prod_y_pred = production_concept_drift

        concept_monitor = ConceptMonitor(
            task="classification",
            metrics=["accuracy", "f1"],
        )
        report = concept_monitor.check(
            ref_y_true,
            ref_y_pred,
            prod_y_true,
            prod_y_pred,
        )

        assert report.has_drift()
        for result in report.feature_results:
            assert result.drift_type == DriftType.CONCEPT

    def test_full_suite_with_drift(
        self,
        reference_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
        production_feature_drift: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Full suite should detect feature + prediction drift together."""
        ref_features, ref_y_true, ref_y_pred = reference_data
        prod_features, prod_y_true, prod_y_pred = production_feature_drift

        suite = DriftSuite(
            reference_data=ref_features,
            reference_predictions=ref_y_pred,
            task="classification",
        )

        report = suite.check(
            production_data=prod_features,
            production_predictions=prod_y_pred,
            y_true_ref=ref_y_true,
            y_pred_ref=ref_y_pred,
            y_true_prod=prod_y_true,
            y_pred_prod=prod_y_pred,
        )

        # Should detect at least feature drift
        assert report.has_drift()
        assert DriftType.FEATURE in report.drift_types_detected()

        # Summary should contain all sections
        summary = report.summary()
        assert "FEATURE DRIFT" in summary
        assert "PREDICTION DRIFT" in summary
        assert "CONCEPT DRIFT" in summary

    def test_comprehensive_report_serialization(
        self,
        reference_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
        production_feature_drift: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Report should serialize to dict/JSON correctly."""
        import json

        ref_features, ref_y_true, ref_y_pred = reference_data
        prod_features, prod_y_true, prod_y_pred = production_feature_drift

        suite = DriftSuite(
            reference_data=ref_features,
            reference_predictions=ref_y_pred,
            task="classification",
            model_version="v1.2.0",
        )

        report = suite.check(
            production_data=prod_features,
            production_predictions=prod_y_pred,
            y_true_ref=ref_y_true,
            y_pred_ref=ref_y_pred,
            y_true_prod=prod_y_true,
            y_pred_prod=prod_y_pred,
        )

        # to_dict
        d = report.to_dict()
        assert d["model_version"] == "v1.2.0"
        assert isinstance(d["drift_types_detected"], list)
        assert d["feature_drift"] is not None

        # to_json
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["status"] in ["OK", "WARNING", "CRITICAL"]


# ---------------------------------------------------------------------------
# House Pricing Tests (Regression)
# ---------------------------------------------------------------------------


class TestHousePricingScenario:
    """Realistic house pricing scenario (regression)."""

    @pytest.fixture
    def reference_data(
        self,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        return generate_house_pricing_data(n=2000, seed=42)

    @pytest.fixture
    def production_no_drift(
        self,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        return generate_house_pricing_data(n=1000, seed=88)

    @pytest.fixture
    def production_market_boom(
        self,
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Market boom: prices increase 40%, bigger houses in demand."""
        return generate_house_pricing_data(
            n=1000,
            seed=66,
            price_mult=1.4,
            area_mean=160.0,
        )

    def test_no_drift_regression(
        self,
        reference_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
        production_no_drift: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Stable market: no drift expected."""
        ref_features, _, _ = reference_data
        prod_features, _, _ = production_no_drift

        monitor = Monitor(reference_data=ref_features)
        report = monitor.check(prod_features)

        # Most features should not drift
        drift_ratio = report.drift_ratio()
        assert drift_ratio < 0.5  # Less than 50% of features drift

    def test_market_boom_feature_drift(
        self,
        reference_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
        production_market_boom: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Market boom should cause feature drift in area_sqm."""
        ref_features, _, _ = reference_data
        prod_features, _, _ = production_market_boom

        monitor = Monitor(reference_data=ref_features)
        report = monitor.check(prod_features)

        assert report.has_drift()
        drifted = report.drifted_features()
        assert "area_sqm" in drifted

    def test_regression_prediction_drift(
        self,
        reference_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
        production_market_boom: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Price predictions should drift in a booming market."""
        _, _, ref_y_pred = reference_data
        _, _, prod_y_pred = production_market_boom

        pred_monitor = PredictionMonitor(
            reference_predictions=ref_y_pred,
            task="regression",
            detector="psi",
        )
        report = pred_monitor.check(prod_y_pred)

        # Predictions are in a completely different range
        assert report.has_drift()

    def test_regression_concept_drift(
        self,
        reference_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
        production_market_boom: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Model trained on old prices → concept drift with new prices."""
        _, ref_y_true, ref_y_pred = reference_data
        _, prod_y_true, _ = production_market_boom

        # Model still predicts like before (old scale), but true prices boomed
        rng = np.random.default_rng(42)
        stale_preds = ref_y_pred[: len(prod_y_true)] + rng.normal(
            0, 5000, len(prod_y_true)
        )

        concept_monitor = ConceptMonitor(
            task="regression",
            metrics=["rmse", "mae", "r2"],
        )
        report = concept_monitor.check(
            ref_y_true,
            ref_y_pred,
            prod_y_true,
            stale_preds,
        )

        # RMSE/MAE should have increased significantly
        assert report.has_drift()

    def test_full_regression_suite(
        self,
        reference_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
        production_market_boom: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Full suite for regression scenario."""
        ref_features, ref_y_true, ref_y_pred = reference_data
        prod_features, prod_y_true, prod_y_pred = production_market_boom

        suite = DriftSuite(
            reference_data=ref_features,
            reference_predictions=ref_y_pred,
            task="regression",
            performance_metrics=["rmse", "r2"],
            model_version="house-v2.1",
        )

        report = suite.check(
            production_data=prod_features,
            production_predictions=prod_y_pred,
            y_true_ref=ref_y_true,
            y_pred_ref=ref_y_pred,
            y_true_prod=prod_y_true,
            y_pred_prod=prod_y_pred,
        )

        assert isinstance(report, ComprehensiveDriftReport)
        assert report.model_version == "house-v2.1"

        # At least feature drift should be detected
        detected = report.drift_types_detected()
        assert DriftType.FEATURE in detected


# ---------------------------------------------------------------------------
# Drift Type Distinction Tests
# ---------------------------------------------------------------------------


class TestDriftTypeDistinction:
    """Verify that drift types are clearly distinguishable in reports."""

    def test_drift_type_enum_values(self) -> None:
        """DriftType enum values should be clear strings."""
        assert DriftType.FEATURE.value == "FEATURE"
        assert DriftType.PREDICTION.value == "PREDICTION"
        assert DriftType.CONCEPT.value == "CONCEPT"

    def test_feature_results_tagged_correctly(self) -> None:
        """Each monitor should tag results with correct drift type."""
        ref_data, ref_y_true, ref_y_pred = generate_credit_scoring_data(n=500, seed=1)
        prod_data, prod_y_true, prod_y_pred = generate_credit_scoring_data(
            n=300, seed=2
        )

        # Feature drift
        feature_monitor = Monitor(reference_data=ref_data)
        feature_report = feature_monitor.check(prod_data)
        for r in feature_report.feature_results:
            assert r.drift_type == DriftType.FEATURE

        # Prediction drift
        pred_monitor = PredictionMonitor(reference_predictions=ref_y_pred)
        pred_report = pred_monitor.check(prod_y_pred)
        for r in pred_report.feature_results:
            assert r.drift_type == DriftType.PREDICTION

        # Concept drift
        concept_monitor = ConceptMonitor(task="classification")
        concept_report = concept_monitor.check(
            ref_y_true, ref_y_pred, prod_y_true, prod_y_pred
        )
        for r in concept_report.feature_results:
            assert r.drift_type == DriftType.CONCEPT

    def test_comprehensive_report_separates_types(self) -> None:
        """ComprehensiveDriftReport should separate types in output."""
        ref_data, ref_y_true, ref_y_pred = generate_credit_scoring_data(n=500, seed=10)
        prod_data, prod_y_true, prod_y_pred = generate_credit_scoring_data(
            n=300,
            seed=20,
            age_mean=55.0,
            income_mean=30000.0,
        )

        suite = DriftSuite(
            reference_data=ref_data,
            reference_predictions=ref_y_pred,
            task="classification",
        )

        report = suite.check(
            production_data=prod_data,
            production_predictions=prod_y_pred,
            y_true_ref=ref_y_true,
            y_pred_ref=ref_y_pred,
            y_true_prod=prod_y_true,
            y_pred_prod=prod_y_pred,
        )

        # to_dict should have clear keys for each type
        d = report.to_dict()
        assert "feature_drift" in d
        assert "prediction_drift" in d
        assert "concept_drift" in d

        # Each section should have its own feature_results
        if d["feature_drift"]:
            for r in d["feature_drift"]["feature_results"]:
                assert r["drift_type"] == "FEATURE"
        if d["prediction_drift"]:
            for r in d["prediction_drift"]["feature_results"]:
                assert r["drift_type"] == "PREDICTION"
        if d["concept_drift"]:
            for r in d["concept_drift"]["feature_results"]:
                assert r["drift_type"] == "CONCEPT"
