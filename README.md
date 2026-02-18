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

- **Multi-Drift Monitoring**:
    - 📊 **Feature Drift**: Monitor input data distribution changes (P(X)).
    - 🎯 **Prediction Drift**: Monitor model output changes (P(Ŷ)).
    - 🧠 **Concept Drift**: Monitor model performance degradation (P(Y|X)).
- **Unified Interface**: `DriftSuite` combines all monitors in one simple API.
- **7 Statistical Detectors**:
    - **PSI**, **KS-Test**, **Wasserstein**, **Jensen-Shannon**, **Anderson-Darling**, **Cramér-von Mises**, **Chi-Squared**.
- **Explainability**: Built-in statistical explanation (`DriftExplainer`) and visualization (`DriftVisualizer`).
- **Production Integrations**:
    - ⚡ **FastAPI** Middleware
    - 📈 **MLflow** Tracking
    - 🔔 **Slack** & **Email** Alerts
- **Lightweight & Robust**: Minimal dependencies, 100% type-safe.

## 📦 Installation

```bash
pip install driftwatch
```

For specific extras:
```bash
pip install driftwatch[viz]     # Visualization support
pip install driftwatch[mlflow]  # MLflow integration
pip install driftwatch[all]     # CLI, API, Alerting, etc.
```

## ⚡ Quick Start

DriftWatch v0.4.0 introduces `DriftSuite` for unified monitoring:

```python
from driftwatch import DriftSuite, DriftType
import pandas as pd

# 1. Initialize suite with reference data (e.g., training set)
suite = DriftSuite(
    reference_data=X_train,
    reference_predictions=y_val_pred,
    task="classification",  # or "regression"
    model_version="v1.0"
)

# 2. Check production batch
report = suite.check(
    production_data=X_prod,
    production_predictions=y_prod_pred
)

# 3. Act on specific drift types
drift_types = report.drift_types_detected()

if DriftType.CONCEPT in drift_types:
    print("🚨 CRITICAL: Concept drift detected — Retrain model!")
elif DriftType.PREDICTION in drift_types:
    print("⚠️ WARNING: Prediction drift — Check model outputs.")
elif DriftType.FEATURE in drift_types:
    print(f"📊 INFO: Feature drift in {report.feature_report.drifted_features()}")
else:
    print("✅ All systems normal.")
```

## 🛠️ Usage Scenarios

| Scenario | Solution |
|----------|----------|
| **Unified Monitoring** | Use `DriftSuite` to track Feature, Prediction, and Concept drift in one go. |
| **Experiment Tracking** | Log all drift metrics to **MLflow** for long-term trend analysis. |
| **Real-time API** | Use `DriftMiddleware` in **FastAPI** to monitor every request. |
| **Alerting** | Send critical alerts via **Slack** or **Email** when model performance degrades. |
| **CI/CD** | Block deployments if `DriftType.PREDICTION` is detected in staging. |

## 📓 Interactive Tutorials

- [**Multi-Drift Tutorial**](examples/notebooks/multi_drift_tutorial.ipynb) — Step-by-step guide to Feature, Prediction, and Concept drift.
- [**Complete Showcase**](examples/notebooks/complete_showcase.ipynb) — Tour of all detectors, visualizers, and integrations.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://vincentcotella.github.io/DriftWatch/contributing/) for details.

1. Fork the repo.
2. Install dev dependencies: `pip install -e ".[dev,all]"`
3. Run tests: `pytest`
4. Submit a PR!

## 📄 License

MIT © [Vincent Cotella](https://github.com/VincentCotella)
