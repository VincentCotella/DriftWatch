# MLflow Integration

Log drift detection results to [MLflow](https://mlflow.org/) experiments for tracking, comparison, and alerting.

---

## Installation

```bash
pip install driftwatch[mlflow]
```

---

## Quick Start

```python
from driftwatch import Monitor
from driftwatch.integrations.mlflow import MLflowDriftTracker

# 1. Run drift detection
monitor = Monitor(reference_data=train_df, features=["age", "income"])
report = monitor.check(production_df)

# 2. Log to MLflow
tracker = MLflowDriftTracker(experiment_name="my-model-drift")
run_id = tracker.log_report(report)

print(f"Logged to MLflow run: {run_id}")
```

---

## What Gets Logged

### Metrics

| Metric | Description |
|--------|-------------|
| `drift.has_drift` | `1.0` if any feature drifted, else `0.0` |
| `drift.drift_ratio` | Ratio of drifted features (0.0 – 1.0) |
| `drift.num_features` | Total number of monitored features |
| `drift.num_drifted` | Number of features with detected drift |
| `drift.{feature}.score` | Drift score for each feature |
| `drift.{feature}.has_drift` | Whether this feature drifted |
| `drift.{feature}.threshold` | Threshold used for detection |
| `drift.{feature}.p_value` | P-value (if available) |

### Parameters

| Parameter | Description |
|-----------|-------------|
| `drift.reference_size` | Number of reference samples |
| `drift.production_size` | Number of production samples |
| `drift.status` | Overall status: `OK`, `WARNING`, or `CRITICAL` |
| `drift.model_version` | Model version (if set in the report) |

### Tags

| Tag | Description |
|-----|-------------|
| `driftwatch.status` | Overall drift status |
| `driftwatch.version` | DriftWatch library version |

### Artifacts

| Artifact | Description |
|----------|-------------|
| `driftwatch/drift_report.json` | Full JSON drift report (optional) |

---

## Configuration

### Custom Tracking URI

```python
tracker = MLflowDriftTracker(
    experiment_name="production-drift",
    tracking_uri="http://mlflow.example.com:5000",
)
```

### Custom Metric Prefix

Use a custom prefix to namespace metrics (useful when tracking multiple models):

```python
tracker = MLflowDriftTracker(
    experiment_name="production-drift",
    prefix="model_v2",  # Metrics: model_v2.has_drift, model_v2.age.score, etc.
)
```

### Disable Artifact Logging

```python
tracker = MLflowDriftTracker(
    experiment_name="production-drift",
    log_report_artifact=False,  # Skip JSON artifact upload
)
```

### Custom Tags

```python
tracker = MLflowDriftTracker(
    experiment_name="production-drift",
    tags={"env": "production", "team": "ml-platform"},
)
```

---

## Advanced Usage

### Log Into an Existing Run

Use `run_id` to append drift metrics to a training or evaluation run:

```python
import mlflow
from driftwatch.integrations.mlflow import MLflowDriftTracker

with mlflow.start_run() as run:
    # ... your training code ...
    model.fit(X_train, y_train)
    mlflow.log_metric("accuracy", 0.95)

    # Log drift alongside training metrics
    tracker = MLflowDriftTracker(experiment_name="my-model")
    tracker.log_report(report, run_id=run.info.run_id)
```

### Extra Parameters

Pass additional context alongside drift data:

```python
tracker.log_report(
    report,
    extra_params={
        "pipeline": "nightly-batch",
        "data_source": "s3://my-bucket/prod-data",
    },
    extra_tags={
        "triggered_by": "airflow",
    },
)
```

### Named Runs

```python
tracker.log_report(
    report,
    run_name="drift-check-2026-02-11",
)
```

---

## Integration with Pipelines

### Airflow Example

```python
from airflow.decorators import task

@task
def check_drift():
    from driftwatch import Monitor
    from driftwatch.integrations.mlflow import MLflowDriftTracker

    monitor = Monitor(reference_data=load_reference(), features=FEATURES)
    report = monitor.check(load_production_data())

    tracker = MLflowDriftTracker(
        experiment_name="production-drift",
        tracking_uri="http://mlflow:5000",
        tags={"pipeline": "airflow"},
    )
    tracker.log_report(report)

    if report.has_drift():
        raise ValueError(f"Drift detected: {report.drifted_features()}")
```

### Combined with Slack Alerting

```python
from driftwatch import Monitor
from driftwatch.integrations.mlflow import MLflowDriftTracker
from driftwatch.integrations.alerting import SlackAlerter

monitor = Monitor(reference_data=train_df, features=["age", "income"])
report = monitor.check(production_df)

# Log to MLflow
tracker = MLflowDriftTracker(experiment_name="production-drift")
tracker.log_report(report)

# Alert on Slack if drift detected
if report.has_drift():
    alerter = SlackAlerter(webhook_url="https://hooks.slack.com/...")
    alerter.send(report)
```

---

## API Reference

::: driftwatch.integrations.mlflow.MLflowDriftTracker
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
