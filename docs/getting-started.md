# Getting Started with DriftWatch

This guide will help you get started with DriftWatch for monitoring data and model drift in your ML systems.

## Installation

```bash
pip install driftwatch
```

With optional dependencies:

```bash
# CLI support
pip install driftwatch[cli]

# FastAPI middleware
pip install driftwatch[fastapi]

# All integrations
pip install driftwatch[all]
```

## Quick Example

```python
import pandas as pd
from driftwatch import Monitor

# Your training data
train_df = pd.DataFrame({
    "age": [25, 30, 35, 40, 45],
    "income": [50000, 60000, 70000, 80000, 90000],
    "category": ["A", "B", "A", "C", "B"],
})

# Initialize monitor with reference data
monitor = Monitor(
    reference_data=train_df,
    features=["age", "income", "category"],
    thresholds={
        "psi": 0.2,        # PSI threshold for numerical
        "ks_pvalue": 0.05, # KS test p-value threshold
        "chi2_pvalue": 0.05, # Chi-squared p-value for categorical
    }
)

# Check production data for drift
production_df = pd.DataFrame({
    "age": [50, 55, 60, 65, 70],  # Older than training
    "income": [100000, 110000, 120000, 130000, 140000],
    "category": ["C", "C", "C", "A", "A"],
})

report = monitor.check(production_df)

# View results
print(report.summary())
print(f"Status: {report.status}")
print(f"Drifted features: {report.drifted_features()}")

# Export as JSON
print(report.to_json())
```

## Understanding the Report

The `DriftReport` provides several methods:

| Method | Description |
|--------|-------------|
| `has_drift()` | Returns `True` if any feature has drift |
| `drifted_features()` | List of feature names with detected drift |
| `status` | Overall status: OK, WARNING, or CRITICAL |
| `summary()` | Human-readable summary string |
| `to_dict()` | Dictionary representation |
| `to_json()` | JSON string representation |

## Drift Status Levels

| Status | Meaning |
|--------|---------|
| `OK` | No drift detected |
| `WARNING` | <50% of features have drift |
| `CRITICAL` | ≥50% of features have drift |

## Next Steps

- [Available Drift Tests](./drift-tests.md)
- [CLI Usage](./cli-guide.md)
- [FastAPI Integration](./integrations.md)
- [API Reference](./api-reference.md)
