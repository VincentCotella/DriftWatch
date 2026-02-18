# Multi-Drift Monitoring

This example demonstrates how to use `DriftSuite` to monitor all three
drift types in a single pipeline.

## Credit Scoring Example

### Setup

```python
import numpy as np
import pandas as pd
from driftwatch import DriftSuite, DriftType

# Reference data (from training/validation)
np.random.seed(42)
n_ref = 2000

X_train = pd.DataFrame({
    "age": np.random.normal(38, 12, n_ref).clip(18, 80),
    "annual_income": np.random.lognormal(10.9, 0.6, n_ref).clip(15000, 500000),
    "debt_ratio": np.random.beta(2, 4, n_ref),
    "credit_history_months": np.random.poisson(72, n_ref),
})

# Validation predictions and labels
y_val = np.random.randint(0, 2, n_ref)
y_val_pred = y_val.copy()
y_val_pred[:100] = 1 - y_val_pred[:100]  # 5% error

# Create the suite
suite = DriftSuite(
    reference_data=X_train,
    reference_predictions=y_val_pred,
    task="classification",
    performance_metrics=["accuracy", "f1", "precision"],
    model_version="credit-v1.0",
)
```

### Check Feature Drift Only

The simplest check — no labels or predictions needed:

```python
X_prod = pd.DataFrame({
    "age": np.random.normal(45, 15, 500).clip(18, 80),  # Shifted!
    "annual_income": np.random.lognormal(10.5, 0.8, 500).clip(15000, 500000),
    "debt_ratio": np.random.beta(3, 3, 500),  # Changed!
    "credit_history_months": np.random.poisson(72, 500),
})

report = suite.check(production_data=X_prod)

print(f"Status: {report.status.value}")
print(f"Feature drift detected: {report.feature_report.has_drift()}")
print(f"Drifted features: {report.feature_report.drifted_features()}")
```

### Check Feature + Prediction Drift

Add production predictions for more insight:

```python
y_prod_pred = np.random.randint(0, 2, 500)

report = suite.check(
    production_data=X_prod,
    production_predictions=y_prod_pred,
)

print(f"\nDrift types detected: {report.drift_types_detected()}")
print(f"Feature drift: {report.feature_report.has_drift()}")
print(f"Prediction drift: {report.prediction_report.has_drift()}")
```

### Full Check (All Three Types)

When ground truth labels become available:

```python
y_prod_true = np.random.randint(0, 2, 500)

report = suite.check(
    production_data=X_prod,
    production_predictions=y_prod_pred,
    y_true_ref=y_val,
    y_pred_ref=y_val_pred,
    y_true_prod=y_prod_true,
    y_pred_prod=y_prod_pred,
)

# Print the comprehensive summary
print(report.summary())
```

Output:

```
============================================================
COMPREHENSIVE DRIFT REPORT
============================================================
Overall Status: WARNING
Timestamp: 2026-02-18T12:00:00+00:00
Drift Types Detected: FEATURE

------------------------------------------------------------
📊 FEATURE DRIFT (Data Distribution)
------------------------------------------------------------
  Status: WARNING
  Drift Ratio: 50.0%
  Affected: 2/4 features
    ⚠ age: psi=0.3521
    ⚠ debt_ratio: psi=0.2847

------------------------------------------------------------
🎯 PREDICTION DRIFT (Model Output Distribution)
------------------------------------------------------------
  Status: OK
  Drift Ratio: 0.0%

------------------------------------------------------------
🧠 CONCEPT DRIFT (Model Performance Degradation)
------------------------------------------------------------
  Status: OK
  Drift Ratio: 0.0%

============================================================
```

### Handling Results Programmatically

```python
# Access individual reports
if report.feature_report and report.feature_report.has_drift():
    for feat in report.feature_report.drifted_features():
        print(f"⚠ Feature '{feat}' has drifted")

# Check specific drift types
if DriftType.CONCEPT in report.drift_types_detected():
    print("🚨 Concept drift detected — consider retraining!")

# Serialize for logging/storage
import json
print(json.dumps(report.to_dict(), indent=2, default=str))
```

## Regression Example

```python
from driftwatch import DriftSuite

# House pricing model
suite = DriftSuite(
    reference_data=X_train,
    reference_predictions=y_val_price_pred,
    task="regression",
    performance_metrics=["rmse", "r2", "mae"],
    model_version="pricing-v2.1",
)

report = suite.check(
    production_data=X_prod,
    production_predictions=y_prod_price_pred,
    y_true_ref=y_val_prices,
    y_pred_ref=y_val_price_pred,
    y_true_prod=y_prod_prices,
    y_pred_prod=y_prod_price_pred,
)
```

## Using Individual Monitors

You can also use each monitor independently:

```python
from driftwatch import Monitor, PredictionMonitor, ConceptMonitor

# Feature drift only
feature_monitor = Monitor(reference_data=X_train)
feature_report = feature_monitor.check(X_prod)

# Prediction drift only
pred_monitor = PredictionMonitor(
    reference_predictions=y_val_pred,
    detector="ks",  # Use KS test instead of PSI
)
pred_report = pred_monitor.check(y_prod_pred)

# Concept drift only
concept_monitor = ConceptMonitor(
    task="classification",
    metrics=["accuracy", "f1"],
    degradation_mode="relative",  # Use relative change
)
concept_report = concept_monitor.check(
    y_val, y_val_pred, y_prod_true, y_prod_pred
)
```

## See Also

- [Drift Types →](../user-guide/drift-types.md) — Understanding each drift type
- [Monitoring Guide →](../user-guide/monitoring.md) — Production strategies
- [Drift Detectors →](../user-guide/detectors.md) — Statistical tests available
