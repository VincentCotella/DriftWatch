# DriftWatch

<div align="center">

**Lightweight ML drift monitoring, built for real-world pipelines**

[![Documentation](https://img.shields.io/badge/docs-vincentcotella.github.io%2FDriftWatch-blue.svg)](https://vincentcotella.github.io/DriftWatch/)
[![CI](https://github.com/VincentCotella/DriftWatch/actions/workflows/ci.yml/badge.svg)](https://github.com/VincentCotella/DriftWatch/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/driftwatch.svg)](https://pypi.org/project/driftwatch/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📖 Documentation

**Read the full documentation here:** [vincentcotella.github.io/DriftWatch](https://vincentcotella.github.io/DriftWatch/)

## 🚀 Features

- **Simple API**: Detect drift in 3 lines of code.
- **Multiple Detectors**: **PSI**, **KS-Test**, **Wasserstein Distance**, **Chi-Squared**.
- **Production-Ready**:
    - ⚡ **FastAPI Integration** (Middleware included).
    - 🔔 **Slack Alerts** built-in.
    - 🛠️ **CLI** for batch processing.
- **Lightweight**: Minimal dependencies (`numpy`, `pandas`, `scipy`).
- **Type-Safe**: 100% typed code with `mypy` support.

## 📦 Installation

```bash
pip install driftwatch
```

For extras (CLI, FastAPI, Alerting):
```bash
pip install driftwatch[all]
```

## ⚡ Quick Start

```python
from driftwatch import Monitor
import pandas as pd

# 1. Initialize monitor with reference data (e.g., training set)
monitor = Monitor(reference_data=pd.read_parquet("train.parquet"))

# 2. Check production data for drift
report = monitor.check(pd.read_parquet("production.parquet"))

# 3. Act on results
if report.has_drift():
    print(f"⚠️ Drift detected! Ratio: {report.drift_ratio():.1%}")
    print(f"Drifted features: {report.drifted_features()}")
else:
    print("✅ All systems normal.")
```

## 🛠️ Usage Scenarios

| Scenario | Solution |
|----------|----------|
| **Real-time API** | Use `DriftMiddleware` in FastAPI to monitor every request. |
| **Batch Job** | Use `driftwatch check` CLI in your Airflow/Cron jobs. |
| **CI/CD** | Block deployments if validation data drifts from training data. |
| **Alerting** | Send Slack notifications automatically when drift is critical. |

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://vincentcotella.github.io/DriftWatch/contributing/) for details.

1. Fork the repo.
2. Install dev dependencies: `pip install -e ".[dev,all]"`
3. Run tests: `pytest`
4. Submit a PR!

## 📄 License

MIT © [Vincent Cotella](https://github.com/VincentCotella)
