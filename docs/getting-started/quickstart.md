# Quickstart

Get started with DriftWatch in 5 minutes.

## Your First Drift Check

### 1. Generate Sample Data

```python
import pandas as pd
import numpy as np

# Training data (reference)
np.random.seed(42)
train_df = pd.DataFrame({
    "age": np.random.normal(35, 10, 1000),
    "income": np.random.lognormal(10.5, 0.5, 1000),
    "credit_score": np.random.normal(700, 50, 1000),
})

# Production data (drifted)
prod_df = pd.DataFrame({
    "age": np.random.normal(45, 12, 500),  # Mean shifted!
    "income": np.random.lognormal(10.5, 0.5, 500),
    "credit_score": np.random.normal(685, 55, 500),  # Mean and variance changed!
})
```

### 2. Create a Monitor

```python
from driftwatch import Monitor

monitor = Monitor(
    reference_data=train_df,
    thresholds={
        "psi": 0.2,  # PSI threshold
        "ks_pvalue": 0.05,  # KS p-value threshold
    }
)
```

### 3. Check for Drift

```python
report = monitor.check(prod_df)

# Quick check
if report.has_drift():
    print(f"⚠️  Drift detected!")
    print(f"Drifted features: {', '.join(report.drifted_features())}")
else:
    print("✓ No drift detected")
```

### 4. Inspect the Report

```python
# Get detailed metrics
print(f"Status: {report.status.value}")
print(f"Drift ratio: {report.drift_ratio():.1%}")

# Feature-level details
for result in report.feature_results:
    status = "⚠️ DRIFT" if result.has_drift else "✓ OK"
    print(f"{result.feature_name}: {status} (score={result.score:.4f})")
```

**Expected Output:**

```
⚠️  Drift detected!
Drifted features: age, credit_score
Status: WARNING
Drift ratio: 66.7%
age: ⚠️ DRIFT (score=0.9521)
income: ✓ OK (score=0.0234)
credit_score: ⚠️ DRIFT (score=0.4582)
```

---

## Export Report

### Save to JSON

```python
import json

# Export full report
with open("drift_report.json", "w") as f:
    json.dump(report.to_dict(), f, indent=2)
```

### Use in CI/CD

```python
import sys

# Exit with error code if drift detected
if report.status.value == "CRITICAL":
    sys.exit(2)  # Critical drift
elif report.status.value == "WARNING":
    sys.exit(1)  # Warning
else:
    sys.exit(0)  # No drift
```

---

## Add Alerts

### Slack Integration

```python
from driftwatch.integrations.alerting import SlackAlerter

alerter = SlackAlerter(
    webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    throttle_minutes=60  # Max 1 alert per hour
)

if report.has_drift():
    alerter.send(report)
```

_<!-- Screenshot placeholder: Slack notification showing drift alert with feature details -->_

---

## Monitor Production API

### FastAPI Integration

```python
from fastapi import FastAPI
from driftwatch.integrations.fastapi import DriftMiddleware

app = FastAPI()

# Add drift monitoring middleware
app.add_middleware(
    DriftMiddleware,
    monitor=monitor,
    check_interval=100,  # Check every 100 requests
    min_samples=50,
)

@app.post("/predict")
async def predict(age: float, income: float, credit_score: float):
    # Your prediction logic
    return {"prediction": 0.85}
```

The middleware automatically:

- ✅ Collects input features
- ✅ Runs drift checks periodically
- ✅ Exposes `/drift/status` endpoint

_<!-- Screenshot placeholder: FastAPI /drift/status endpoint response -->_

---

## Use the CLI

### Generate Report

```bash
driftwatch check \
  --ref train.parquet \
  --prod prod.parquet \
  --output drift_report.json
```

_<!-- Screenshot placeholder: CLI output showing colorful drift report table -->_

### View Existing Report

```bash
driftwatch report drift_report.json --format table
```

---

## What's Next?

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } __Core Concepts__

    ---

    Understand drift detection fundamentals

    [:octicons-arrow-right-24: Learn More](concepts.md)

-   :material-chart-line:{ .lg .middle } __Drift Detectors__

    ---

    Choose the right detector for your use case

    [:octicons-arrow-right-24: Detectors Guide](../user-guide/detectors.md)

-   :material-cog:{ .lg .middle } __Thresholds__

    ---

    Configure sensitivity and alerting

    [:octicons-arrow-right-24: Threshold Guide](../user-guide/thresholds.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Full API documentation

    [:octicons-arrow-right-24: API Docs](../api/monitor.md)

</div>
