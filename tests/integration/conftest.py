"""Fixtures for integration tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris, load_wine


@pytest.fixture
def iris_reference_df() -> pd.DataFrame:
    """Load Iris dataset as reference DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical([iris.target_names[t] for t in iris.target])
    return df


@pytest.fixture
def iris_production_no_drift(iris_reference_df: pd.DataFrame) -> pd.DataFrame:
    """Iris data with same distribution (no drift) - identical copy."""
    # Return exact copy for true no-drift scenario
    return iris_reference_df.copy()


@pytest.fixture
def iris_production_with_drift(iris_reference_df: pd.DataFrame) -> pd.DataFrame:
    """Iris data with intentional drift (mean shift on all numerical features)."""
    np.random.seed(123)
    df = iris_reference_df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Shift mean by 1-2 standard deviations
        shift = df[col].std() * 1.5
        df[col] = df[col] + shift
    return df


@pytest.fixture
def wine_reference_df() -> pd.DataFrame:
    """Load Wine dataset as reference DataFrame (larger dataset)."""
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df["wine_class"] = pd.Categorical([wine.target_names[t] for t in wine.target])
    return df


@pytest.fixture
def large_synthetic_reference_df() -> pd.DataFrame:
    """Generate large synthetic dataset with mixed types (1000+ samples)."""
    np.random.seed(42)
    n_samples = 2000

    return pd.DataFrame(
        {
            # Numerical features
            "age": np.random.normal(35, 10, n_samples),
            "income": np.random.lognormal(10, 1, n_samples),
            "score": np.random.uniform(0, 100, n_samples),
            "transactions": np.random.poisson(5, n_samples),
            # Categorical features
            "category": pd.Categorical(
                np.random.choice(
                    ["A", "B", "C", "D"], n_samples, p=[0.4, 0.3, 0.2, 0.1]
                )
            ),
            "region": pd.Categorical(
                np.random.choice(["North", "South", "East", "West"], n_samples)
            ),
            "status": pd.Categorical(
                np.random.choice(
                    ["active", "inactive", "pending"], n_samples, p=[0.6, 0.3, 0.1]
                )
            ),
        }
    )


@pytest.fixture
def large_synthetic_drifted_df() -> pd.DataFrame:
    """Generate drifted version of synthetic dataset."""
    np.random.seed(999)
    n_samples = 2000

    return pd.DataFrame(
        {
            # Numerical features with drift
            "age": np.random.normal(45, 15, n_samples),  # Mean +10, variance increased
            "income": np.random.lognormal(11, 1.5, n_samples),  # Shifted up
            "score": np.random.uniform(20, 120, n_samples),  # Range shifted
            "transactions": np.random.poisson(10, n_samples),  # Lambda doubled
            # Categorical features with distribution change
            "category": pd.Categorical(
                np.random.choice(
                    ["A", "B", "C", "D"], n_samples, p=[0.1, 0.2, 0.3, 0.4]
                )
            ),
            "region": pd.Categorical(
                np.random.choice(
                    ["North", "South", "East", "West"],
                    n_samples,
                    p=[0.7, 0.1, 0.1, 0.1],
                )
            ),
            "status": pd.Categorical(
                np.random.choice(
                    ["active", "inactive", "pending"], n_samples, p=[0.2, 0.5, 0.3]
                )
            ),
        }
    )


@pytest.fixture
def single_sample_df() -> pd.DataFrame:
    """DataFrame with only one sample."""
    return pd.DataFrame(
        {
            "age": [35.0],
            "income": [50000.0],
            "category": pd.Categorical(["A"]),
        }
    )


@pytest.fixture
def df_with_nans() -> pd.DataFrame:
    """DataFrame with NaN values."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "age": np.random.normal(35, 10, 100),
            "income": np.random.lognormal(10, 1, 100),
        }
    )
    # Inject NaNs
    df.loc[::10, "age"] = np.nan  # 10% NaNs
    df.loc[::5, "income"] = np.nan  # 20% NaNs
    return df


@pytest.fixture
def df_with_outliers() -> pd.DataFrame:
    """DataFrame with extreme outliers."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "value": np.random.normal(100, 10, 1000),
        }
    )
    # Add extreme outliers
    df.loc[0, "value"] = 1_000_000  # Extreme high
    df.loc[1, "value"] = -1_000_000  # Extreme low
    return df
