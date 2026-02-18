# Monitoring Guide

Learn how to monitor your ML models in production with DriftWatch.

## Types of Drift

DriftWatch supports three types of drift monitoring:

| Drift Type | What Changes | Monitor Class | Needs Labels? |
|------------|-------------|---------------|:---:|
| **Feature Drift** | Input data distribution P(X) | `Monitor` | ❌ |
| **Prediction Drift** | Model output distribution P(Ŷ) | `PredictionMonitor` | ❌ |
| **Concept Drift** | Input-output relationship P(Y\|X) | `ConceptMonitor` | ✅ |

### Feature Drift (Data Drift)

The input data distribution has changed compared to training data. This is the most common and easiest to detect since it requires no ground truth labels.

```python
from driftwatch import Monitor

monitor = Monitor(reference_data=X_train, features=["age", "income", "score"])
report = monitor.check(X_production)

if report.has_drift():
    print(f"Feature drift detected in: {report.drifted_features()}")
```

### Prediction Drift

The distribution of model predictions has changed. This can happen when feature drift causes the model to produce different output patterns.

```python
from driftwatch import PredictionMonitor

# Regression
monitor = PredictionMonitor(reference_predictions=y_val_pred)
report = monitor.check(y_prod_pred)

# Classification (probabilities)
monitor = PredictionMonitor(
    reference_predictions=y_val_proba,  # shape (n, n_classes)
    task="classification",
    class_names=["negative", "positive"],
)
report = monitor.check(y_prod_proba)
```

### Concept Drift

The relationship between inputs and outputs has changed. The model's learned patterns are no longer valid. This requires ground truth labels (delayed feedback).

```python
from driftwatch import ConceptMonitor

# Classification
monitor = ConceptMonitor(
    task="classification",
    metrics=["accuracy", "f1", "precision", "recall"],
)
report = monitor.check(
    y_true_ref=y_val, y_pred_ref=y_val_pred,
    y_true_prod=y_prod, y_pred_prod=y_prod_pred,
)

# Regression
monitor = ConceptMonitor(
    task="regression",
    metrics=["rmse", "r2", "mae"],
)
```

## Unified Monitoring with DriftSuite

`DriftSuite` combines all three drift types into a single interface:

```python
from driftwatch import DriftSuite

suite = DriftSuite(
    reference_data=X_train,
    reference_predictions=y_val_pred,
    task="classification",
    model_version="v1.2.0",
)

# Full check (all three drift types)
report = suite.check(
    production_data=X_prod,
    production_predictions=y_prod_pred,
    y_true_ref=y_val,
    y_pred_ref=y_val_pred,
    y_true_prod=y_prod,
    y_pred_prod=y_prod_pred,
)

print(report.summary())
```

### Partial Checks

You don't have to provide all data at once. DriftSuite checks only the drift types for which data is available:

```python
# Feature drift only (most common)
report = suite.check(production_data=X_prod)

# Feature + Prediction (no labels needed)
report = suite.check(
    production_data=X_prod,
    production_predictions=y_prod_pred,
)
```

## DriftType Enum

Every result carries a `drift_type` field for clear identification:

```python
from driftwatch import DriftType

for result in report.feature_results:
    if result.drift_type == DriftType.FEATURE:
        print(f"Feature drift: {result.feature_name}")
    elif result.drift_type == DriftType.PREDICTION:
        print(f"Prediction drift: {result.feature_name}")
    elif result.drift_type == DriftType.CONCEPT:
        print(f"Concept drift: {result.feature_name}")
```

## ComprehensiveDriftReport

The `ComprehensiveDriftReport` from `DriftSuite` provides:

```python
# Which drift types were detected
report.drift_types_detected()
# → [DriftType.FEATURE, DriftType.PREDICTION]

# Overall status (worst across all types)
report.status
# → DriftStatus.CRITICAL

# Access individual reports
report.feature_report      # DriftReport or None
report.prediction_report   # DriftReport or None
report.concept_report      # DriftReport or None

# Human-readable summary
print(report.summary())

# Serialization
d = report.to_dict()
json_str = report.to_json()
```

## Monitoring Strategy

### Recommended Monitoring Levels

```
Level 1: Feature Drift Only (always available)
        → Run on every batch
        → Alert if drift_ratio > 30%

Level 2: Feature + Prediction Drift (no labels needed)
        → Run on every batch
        → Correlate feature and prediction drift

Level 3: Full Suite (requires delayed labels)
        → Run when ground truth becomes available
        → Highest confidence: concept drift = model must be retrained
```

### Production Pipeline Example

```python
from driftwatch import DriftSuite, DriftType

suite = DriftSuite(
    reference_data=X_train,
    reference_predictions=y_val_pred,
    task="classification",
    model_version="v1.2.0",
)

# On every inference batch
report = suite.check(
    production_data=X_batch,
    production_predictions=y_batch_pred,
)

if DriftType.FEATURE in report.drift_types_detected():
    send_alert("Feature drift detected — investigate input data pipeline")

if DriftType.PREDICTION in report.drift_types_detected():
    send_alert("Prediction drift detected — model outputs are changing")

# When labels arrive (delayed)
if labels_available:
    full_report = suite.check(
        production_data=X_batch,
        production_predictions=y_batch_pred,
        y_true_ref=y_val,
        y_pred_ref=y_val_pred,
        y_true_prod=y_batch_true,
        y_pred_prod=y_batch_pred,
    )

    if DriftType.CONCEPT in full_report.drift_types_detected():
        send_alert("CRITICAL: Concept drift — model retraining required")
```

## See Also

- [Drift Detectors →](detectors.md) — Choose the right statistical test
- [Reports →](reports.md) — Understanding drift reports
- [Thresholds →](thresholds.md) — Tuning sensitivity
