"""
Basic usage example for DriftWatch.

This example demonstrates how to:
1. Initialize a Monitor with reference data (training set).
2. Check production data for potential distribution drift.
3. Interpret the drift report and metrics (PSI, KS, Chi-Square).
"""

import numpy as np
import pandas as pd

from driftwatch import Monitor


def create_training_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic training data."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": np.random.normal(35, 10, n_samples),
            "income": np.random.lognormal(10.5, 0.5, n_samples),
            "credit_score": np.random.normal(700, 50, n_samples),
            "category": np.random.choice(["A", "B", "C"], n_samples, p=[0.5, 0.3, 0.2]),
        }
    )


def create_production_data_no_drift(n_samples: int = 500) -> pd.DataFrame:
    """Generate production data similar to training (no drift)."""
    np.random.seed(123)
    return pd.DataFrame(
        {
            "age": np.random.normal(35, 10, n_samples),
            "income": np.random.lognormal(10.5, 0.5, n_samples),
            "credit_score": np.random.normal(700, 50, n_samples),
            "category": np.random.choice(["A", "B", "C"], n_samples, p=[0.5, 0.3, 0.2]),
        }
    )


def create_production_data_with_drift(n_samples: int = 500) -> pd.DataFrame:
    """Generate production data with drift."""
    np.random.seed(456)
    return pd.DataFrame(
        {
            # Age distribution shifted (older population)
            "age": np.random.normal(45, 15, n_samples),
            # Income distribution shifted (higher incomes)
            "income": np.random.lognormal(11, 0.7, n_samples),
            # Credit score similar (no drift)
            "credit_score": np.random.normal(700, 50, n_samples),
            # Category distribution changed
            "category": np.random.choice(["A", "B", "C"], n_samples, p=[0.2, 0.5, 0.3]),
        }
    )


def main() -> None:
    """Run the example."""
    print("=" * 60)
    print("DriftWatch Basic Example")
    print("=" * 60)

    # Create datasets
    train_df = create_training_data()
    prod_no_drift = create_production_data_no_drift()
    prod_with_drift = create_production_data_with_drift()

    print(f"\nTraining data shape: {train_df.shape}")
    print(f"Production data shape: {prod_no_drift.shape}")

    # Initialize monitor
    monitor = Monitor(
        reference_data=train_df,
        thresholds={
            "psi": 0.2,
            "ks_pvalue": 0.05,
            "chi2_pvalue": 0.05,
        },
    )

    print(f"\nMonitoring features: {monitor.monitored_features}")

    # Check 1: Production data without drift
    print("\n" + "=" * 60)
    print("CHECK 1: Production data WITHOUT drift")
    print("=" * 60)

    report1 = monitor.check(prod_no_drift)
    print(report1.summary())

    # Check 2: Production data with drift
    print("\n" + "=" * 60)
    print("CHECK 2: Production data WITH drift")
    print("=" * 60)

    report2 = monitor.check(prod_with_drift)
    print(report2.summary())

    # Export as JSON
    print("\n" + "=" * 60)
    print("JSON EXPORT")
    print("=" * 60)
    print(report2.to_json())


if __name__ == "__main__":
    main()
