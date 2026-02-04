# Monitor API

The `Monitor` class is the main entry point for drift detection.

## Overview

The Monitor analyzes feature distributions between reference (training) and production data to detect drift.

## Quick Example

```python
from driftwatch import Monitor
import pandas as pd

# Load training data
train_df = pd.read_parquet("train.parquet")

# Create monitor
monitor = Monitor(
    reference_data=train_df,
    thresholds={
        "psi": 0.2,
        "ks_pvalue": 0.05,
    }
)

# Check production data
prod_df = pd.read_parquet("prod.parquet")
report = monitor.check(prod_df)
```

---

## API Reference

::: driftwatch.core.monitor.Monitor
    options:
      show_root_heading: true
      show_source: true
      members:
        - __init__
        - check
        - update_reference
        - get_thresholds

---

## Configuration

### Thresholds

Control sensitivity of drift detection:

```python
monitor = Monitor(
    reference_data=train_df,
    thresholds={
        "psi": 0.15,           # More sensitive than default 0.2
        "ks_pvalue": 0.01,     # More strict than default 0.05
        "chi2_pvalue": 0.05,   # Default
        "wasserstein": 0.2,    # For Wasserstein detector
    }
)
```

### Model Version Tracking

Track which model version is being monitored:

```python
monitor = Monitor(
    reference_data=train_df,
    model_version="v1.2.3"
)

report = monitor.check(prod_df)
print(report.model_version)  # "v1.2.3"
```

---

## Methods

### check()

Detect drift in production data:

```python
report = monitor.check(
    production_data=prod_df,
    # Optional: override thresholds per feature
    feature_thresholds={
        "age": {"psi": 0.1},  # More sensitive for age
    }
)
```

### update_reference()

Update reference data (e.g., after model retraining):

```python
# Retrain model with new data
new_train_df = pd.read_parquet("retrain_data.parquet")

# Update monitor
monitor.update_reference(new_train_df)
```

---

## Best Practices

### 1. Choose Appropriate Reference Data

Reference data should represent your model's training distribution:

```python
# ✓ Good: Use actual training data
monitor = Monitor(reference_data=train_df)

# ✗ Bad: Using validation data with different distribution
monitor = Monitor(reference_data=val_df)
```

### 2. Set Thresholds Based on Business Impact

```python
# High-stakes model: strict thresholds
critical_monitor = Monitor(
    reference_data=train_df,
    thresholds={"psi": 0.1, "ks_pvalue": 0.01}
)

# Exploratory model: relaxed thresholds
exploratory_monitor = Monitor(
    reference_data=train_df,
    thresholds={"psi": 0.3}
)
```

### 3. Version Your Reference Data

```python
import joblib

# Save monitor for reproducibility
joblib.dump(monitor, f"monitor_v{model_version}.pkl")

# Load later
monitor = joblib.load("monitor_v1.2.3.pkl")
```

---

## See Also

- [Drift Detectors →](detectors.md) - Available detection methods
- [Reports →](reports.md) - Working with drift reports
- [Thresholds Guide →](../user-guide/thresholds.md) - Tuning sensitivity
