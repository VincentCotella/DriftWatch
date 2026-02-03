"""Pytest configuration and fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_numerical_df() -> pd.DataFrame:
    """Create a sample DataFrame with numerical features."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": np.random.normal(35, 10, 1000),
            "income": np.random.lognormal(10, 1, 1000),
            "score": np.random.uniform(0, 100, 1000),
        }
    )


@pytest.fixture
def drifted_numerical_df() -> pd.DataFrame:
    """Create a DataFrame with shifted distributions (drift)."""
    np.random.seed(123)
    return pd.DataFrame(
        {
            "age": np.random.normal(45, 15, 1000),  # Mean shifted
            "income": np.random.lognormal(11, 1.5, 1000),  # Shifted
            "score": np.random.uniform(20, 120, 1000),  # Range shifted
        }
    )


@pytest.fixture
def sample_categorical_df() -> pd.DataFrame:
    """Create a sample DataFrame with categorical features."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "category": np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2]),
            "status": np.random.choice(["active", "inactive"], 1000, p=[0.7, 0.3]),
        }
    )


@pytest.fixture
def drifted_categorical_df() -> pd.DataFrame:
    """Create a DataFrame with changed category distributions."""
    np.random.seed(123)
    return pd.DataFrame(
        {
            "category": np.random.choice(["A", "B", "C"], 1000, p=[0.2, 0.5, 0.3]),
            "status": np.random.choice(["active", "inactive"], 1000, p=[0.4, 0.6]),
        }
    )
