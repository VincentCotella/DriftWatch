---
hide:
  - navigation
  - toc
---

<div class="hero-section">
  <div class="hero-title">DriftWatch</div>
  <div class="hero-subtitle">Lightweight ML drift monitoring, built for real-world pipelines</div>
  
  <div class="hero-badges">
    <a href="https://github.com/VincentCotella/DriftWatch/actions/workflows/ci.yml">
      <img src="https://github.com/VincentCotella/DriftWatch/actions/workflows/ci.yml/badge.svg" alt="CI">
    </a>
    <a href="https://pypi.org/project/driftwatch/">
      <img src="https://badge.fury.io/py/driftwatch.svg" alt="PyPI version">
    </a>
    <a href="https://www.python.org/downloads/">
      <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
    </a>
    <a href="https://opensource.org/licenses/MIT">
      <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
    </a>
    <a href="https://github.com/VincentCotella/DriftWatch">
      <img src="https://img.shields.io/badge/coverage-96%25-brightgreen" alt="Coverage">
    </a>
  </div>

  <div style="display: flex; gap: 1rem; justify-content: center; margin-top: 2rem;">
    <a href="getting-started/quickstart/" class="md-button md-button--primary">
      Get Started
    </a>
    <a href="https://github.com/VincentCotella/DriftWatch" class="md-button">
      View on GitHub
    </a>
  </div>
</div>

---

## What is DriftWatch?

DriftWatch is a **production-ready** drift detection library designed for ML engineers who need reliable, actionable drift monitoring without the overhead of complex platforms.

### Key Features

<div class="grid cards" markdown>

-   :material-flash:{ .lg .middle } __Simple API__

    ---

    Get started in 3 lines of code. No complex configuration required.

-   :material-factory:{ .lg .middle } __Production-Ready__

    ---

    Includes FastAPI middleware, Slack alerts, and comprehensive CLI tools.

-   :material-chart-bell-curve-cumulative:{ .lg .middle } __Multiple Detectors__

    ---

    PSI, KS-Test, Wasserstein Distance, and Chi-Squared tests included.

-   :material-shield-check:{ .lg .middle } __Reliable__

    ---

    Fully type-hinted, tested (96%+ coverage), and minimally dependent.

</div>

---

## Quick Example

```python
from driftwatch import Monitor
import pandas as pd

# Load your data
train_df = pd.read_parquet("train.parquet")
prod_df = pd.read_parquet("production.parquet")

# Create monitor
monitor = Monitor(reference_data=train_df)

# Check for drift
report = monitor.check(prod_df)

if report.has_drift():
    print(f"⚠️ Drift detected in {len(report.drifted_features())} features!")
    print(f"Drift ratio: {report.drift_ratio():.1%}")
```

---

## Use Cases

### 1. API Monitoring

Monitor your FastAPI ML service automatically:

```python
from fastapi import FastAPI
from driftwatch.integrations.fastapi import DriftMiddleware

app = FastAPI()
app.add_middleware(DriftMiddleware, monitor=monitor)
```

[Read FastAPI Guide →](integrations/fastapi.md){ .md-button }

### 2. CI/CD Integration

Check for drift in your deployment pipeline to prevent bad rollouts.

```bash
driftwatch check --ref train.parquet --prod new_data.parquet
```

[Read CLI Guide →](integrations/cli.md){ .md-button }

### 3. Slack Alerting

Get notified instantly when critical features drift.

```python
if report.has_drift():
    alerter.send(report)
```

[Read Slack Guide →](integrations/slack.md){ .md-button }

---

## What's Next?

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Get Started__

    ---

    Run your first drift check in minutes

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } __Core Concepts__

    ---

    Understand the math behind drift

    [:octicons-arrow-right-24: Concepts](getting-started/concepts.md)

-   :material-code-braces:{ .lg .middle } __API Reference__

    ---

    Explore the full API documentation

    [:octicons-arrow-right-24: API Docs](api/monitor.md)

</div>
