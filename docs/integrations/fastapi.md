# FastAPI Integration

Monitor your FastAPI inference endpoints automatically with DriftWatch.

## Installation

```bash
pip install driftwatch[fastapi]
```

## Quick Setup

### 1. Add Middleware

```python
from fastapi import FastAPI
from driftwatch import Monitor
from driftwatch.integrations.fastapi import DriftMiddleware
import pandas as pd

# Load reference data
train_df = pd.read_parquet("train.parquet")

# Create monitor
monitor = Monitor(reference_data=train_df)

# Create FastAPI app
app = FastAPI()

# Add drift monitoring
app.add_middleware(
    DriftMiddleware,
    monitor=monitor,
    check_interval=100,  # Check every 100 requests
    min_samples=50,      # Minimum samples before checking
    enabled=True,
)
```

### 2. Add Endpoints

```python
from driftwatch.integrations.fastapi import add_drift_routes

# Add /drift/* endpoints
add_drift_routes(app, middleware)
```

This adds:

- `GET /drift/status` - Current drift status
- `GET /drift/report` - Full drift report
- `GET /drift/health` - Health check
- `POST /drift/check` - Manual drift check
- `POST /drift/reset` - Reset buffer

---

## Configuration

### Feature Extraction

Custom feature extractor for complex request formats:

```python
def extract_features(request_body: dict) -> dict:
    """Extract relevant features from request."""
    return {
        "age": request_body["user"]["age"],
        "income": request_body["user"]["income"],
        "credit_score": request_body["credit"]["score"],
    }

app.add_middleware(
    DriftMiddleware,
    monitor=monitor,
    feature_extractor=extract_features,
    check_interval=100,
)
```

### Prediction Collection

Collect predictions for model drift analysis:

```python
def extract_prediction(response_body: dict) -> dict:
    """Extract prediction from response."""
    return {"probability": response_body["prediction"]}

app.add_middleware(
    DriftMiddleware,
    monitor=monitor,
    prediction_extractor=extract_prediction,
    check_interval=100,
)
```

### Buffer Size

Control memory usage:

```python
app.add_middleware(
    DriftMiddleware,
    monitor=monitor,
    buffer_size=5000,  # Keep last 5000 samples
    check_interval=100,
)
```

---

## API Endpoints

### GET /drift/status

Get current drift status:

```bash
curl http://localhost:8000/drift/status
```

**Response:**

```json
{
  "status": "WARNING",
  "has_drift": true,
  "drift_ratio": 0.333,
  "drifted_features": ["age"],
  "last_check": "2024-01-15T14:30:00Z",
  "samples_collected": 150,
  "total_requests": 523
}
```

_<!-- Screenshot placeholder: /drift/status JSON response in browser -->_

### GET /drift/report

Full drift report with feature details:

```bash
curl http://localhost:8000/drift/report
```

**Response:**

```json
{
  "status": "WARNING",
  "timestamp": "2024-01-15T14:30:00Z",
  "feature_results": [
    {
      "feature_name": "age",
      "has_drift": true,
      "score": 0.3521,
      "method": "psi",
      "threshold": 0.2
    }
  ]
}
```

### POST /drift/check

Trigger manual drift check:

```bash
curl -X POST http://localhost:8000/drift/check
```

### POST /drift/reset

Reset sample buffer:

```bash
curl -X POST http://localhost:8000/drift/reset
```

---

## Complete Example

```python
from fastapi import FastAPI
from driftwatch import Monitor
from driftwatch.integrations.fastapi import DriftMiddleware, add_drift_routes
import pandas as pd

# Setup
train_df = pd.read_parquet("train.parquet")
monitor = Monitor(reference_data=train_df)
app = FastAPI(title="ML Inference API")

# Add drift monitoring
middleware = DriftMiddleware(
    app=app,
    monitor=monitor,
    check_interval=100,
    min_samples=50,
)
app.add_middleware(DriftMiddleware, **middleware.__dict__)
add_drift_routes(app, middleware)

# Your prediction endpoint
@app.post("/predict")
async def predict(
    age: float,
    income: float,
    credit_score: float
):
    # Predictions automatically monitored
    prediction = model.predict([[age, income, credit_score]])[0]
    
    return {
        "prediction": float(prediction),
        "confidence": 0.87,
    }

# Run with: uvicorn main:app --reload
```

---

## Production Tips

### 1. Disable in Development

```python
import os

app.add_middleware(
    DriftMiddleware,
    monitor=monitor,
    enabled=os.getenv("ENV") == "production",
)
```

### 2. Combine with Alerts

```python
from driftwatch.integrations.alerting import SlackAlerter

alerter = SlackAlerter(webhook_url="https://hooks.slack.com/...")

# Check periodically and alert
@app.middleware("http")
async def check_and_alert(request, call_next):
    response = await call_next(request)
    
    if middleware.state.last_report and middleware.state.last_report.has_drift():
        alerter.send(middleware.state.last_report)
    
    return response
```

### 3. Monitor Metrics

Export to Prometheus, DataDog, etc.:

```python
from prometheus_client import Gauge

drift_ratio_gauge = Gauge("drift_ratio", "Feature drift ratio")

@app.get("/metrics")
async def metrics():
    if middleware.state.last_report:
        drift_ratio_gauge.set(middleware.state.last_report.drift_ratio())
    # ... return Prometheus metrics
```

---

## Demo Application

A full demo is available in the repository:

```bash
git clone https://github.com/VincentCotella/DriftWatch
cd DriftWatch
python examples/fastapi_demo.py
```

Open [http://localhost:8000](http://localhost:8000) to see the interactive dashboard.

_<!-- Screenshot placeholder: FastAPI demo dashboard with drift visualization -->_

---

## Next Steps

- [CLI Integration →](cli.md)
- [Slack Alerts →](slack.md)
- [API Reference →](../api/integrations.md)
