# Drift Types

DriftWatch monitors three distinct types of drift, each representing a different
kind of change in your ML system.

## Overview

```
Training Data ──→ Model ──→ Predictions ──→ Outcomes
     ↕                         ↕                ↕
Feature Drift          Prediction Drift    Concept Drift
   P(X) changes         P(Ŷ) changes       P(Y|X) changes
```

## Feature Drift (Data Drift)

**What**: The distribution of input features has changed.

**Why it matters**: If the model was trained on data with certain patterns
(e.g., age ~ Normal(35, 10)) and production data shifts (age ~ Normal(50, 15)),
model predictions may become unreliable.

**Example scenarios**:
- Economic recession changes customer income distributions
- Seasonal effects alter user behavior patterns
- Data pipeline bug produces different value ranges

**Monitor**: `Monitor` (the standard DriftWatch monitor)

```python
from driftwatch import Monitor

monitor = Monitor(reference_data=X_train)
report = monitor.check(X_production)
# report.drifted_features() → ["age", "income", "debt_ratio"]
```

## Prediction Drift (Output Drift)

**What**: The distribution of model predictions has changed.

**Why it matters**: Even without access to ground truth labels, you can
detect that your model is behaving differently. This is an early warning
signal.

**Example scenarios**:
- A classification model that usually predicts 20% positive suddenly predicts 50%
- A pricing model starts predicting systematically higher values
- Model confidence scores shift to extremes

**Monitor**: `PredictionMonitor`

```python
from driftwatch import PredictionMonitor

# Regression predictions
monitor = PredictionMonitor(reference_predictions=y_val_pred)
report = monitor.check(y_prod_pred)

# Classification probabilities (per-class)
monitor = PredictionMonitor(
    reference_predictions=y_val_proba,  # shape (n, n_classes)
    task="classification",
    class_names=["no_default", "default"],
)
report = monitor.check(y_prod_proba)
```

## Concept Drift

**What**: The relationship between features and target has changed (P(Y|X)).

**Why it matters**: This is the most serious drift type. The model's learned
patterns are wrong — the same inputs should now produce different outputs.
**The model must be retrained.**

**Example scenarios**:
- Customer default patterns change during economic crisis
- Medical diagnosis criteria evolve with new research
- Fraud patterns shift as attackers adapt

**Monitor**: `ConceptMonitor`

```python
from driftwatch import ConceptMonitor

# Classification
monitor = ConceptMonitor(
    task="classification",
    metrics=["accuracy", "f1"],
    thresholds={"accuracy": 0.05, "f1": 0.05},
)
report = monitor.check(
    y_true_ref=y_val, y_pred_ref=y_val_pred,
    y_true_prod=y_prod, y_pred_prod=y_prod_pred,
)

# Show performance degradation details
for detail in monitor.performance_details:
    print(f"{detail.metric_name}: {detail.reference_value:.3f} → "
          f"{detail.production_value:.3f} (Δ{detail.absolute_change:+.3f})")
```

**Available Metrics**:

| Task | Metrics | Higher is Better |
|------|---------|:---:|
| Classification | `accuracy`, `precision`, `recall`, `f1`, `auc_roc` | ✅ |
| Regression | `mae`, `mse`, `rmse`, `mape` | ❌ |
| Regression | `r2` | ✅ |

## Interaction Between Drift Types

The three drift types are related but independent:

| Scenario | Feature | Prediction | Concept |
|----------|:---:|:---:|:---:|
| New data population | ✅ | ✅ | ❌ |
| Model is fine, data changed | ✅ | ✅ | ❌ |
| Model degraded, data unchanged | ❌ | ❌ | ✅ |
| Everything changed | ✅ | ✅ | ✅ |
| Data pipeline bug (values) | ✅ | ✅ | ⚠️ |
| Gradual performance decay | ❌ | ❌ | ✅ |

### Key Insight

- **Feature + Prediction drift** without concept drift → Model may still be OK, but verify
- **Concept drift** alone → Model must be retrained, even if data looks similar
- **All three** → Urgent: data and model have diverged significantly

## The DriftType Enum

```python
from driftwatch import DriftType

DriftType.FEATURE     # Input data distribution shift
DriftType.PREDICTION  # Model output distribution shift
DriftType.CONCEPT     # Input-output relationship shift
```

Every `FeatureDriftResult` carries a `drift_type` field, so you always know
which category a detection belongs to.
