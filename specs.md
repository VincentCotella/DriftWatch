# 🏷️ Nom du projet

## **DriftWatch**

**Tagline :** *Lightweight ML drift monitoring, built for real-world pipelines.*

Pourquoi ce nom :

* clair et pro
* explicite pour ML / MLOps
* crédible en open source
* pas gadget, pas buzzword IA

Repo :

```
driftwatch/
```

---

# 📄 `SPECIFICATIONS.md`

Voici un **fichier de specs complet**, orienté **outil OSS sérieux**, mais **réaliste à implémenter**.

---

````md
# DriftWatch – Specifications

## 1. Overview

DriftWatch is a lightweight, open-source Python library designed to detect and monitor
data drift and model drift in machine learning systems.

The goal is to provide a **plug-and-play monitoring layer** that integrates easily with
existing ML pipelines without requiring heavy infrastructure or vendor lock-in.

Target users:
- ML Engineers
- Backend Engineers deploying ML models
- Data Scientists moving models to production

---

## 2. Core Objectives

- Detect **data drift** between reference (training) data and production data
- Detect **model drift** using predictions and optional ground truth labels
- Provide **clear, actionable reports**
- Be usable as:
  - a Python library
  - a CLI tool
  - an optional service (future)
- Integrate seamlessly with existing tools (FastAPI, MLflow, Slack)

---

## 3. Non-Goals (V1)

- No automated retraining
- No heavy real-time streaming system
- No complex deep learning explainability
- No mandatory external services

---

## 4. Supported Drift Types

### 4.1 Data Drift

Detected when the distribution of input features changes over time.

Supported methods:

#### Numerical Features
- Kolmogorov–Smirnov Test (KS)
- Population Stability Index (PSI)
- Wasserstein Distance

#### Categorical Features
- Chi-Squared Test
- Frequency-based PSI

Drift is computed per feature and aggregated at dataset level.

---

### 4.2 Model Drift

Detected when model behavior or performance degrades.

Supported signals:
- Prediction distribution drift
- Score drift (PSI on predicted probabilities)
- Performance drop (if labels are available)
- Error distribution drift

---

## 5. Public API Design

### 5.1 Core Object

```python
from driftwatch import Monitor

monitor = Monitor(
    reference_data=train_df,
    features=[...],
    model=model,                  # optional
    thresholds={
        "psi": 0.2,
        "ks_pvalue": 0.05
    }
)
````

---

### 5.2 Drift Check

```python
report = monitor.check(prod_df)
```

Returned object:

* `DriftReport`

Capabilities:

* summary()
* feature_drift()
* has_drift()
* to_dict()
* to_json()

---

## 6. Drift Report Structure

The DriftReport object contains:

* global_status: OK / WARNING / CRITICAL
* per_feature_metrics
* drifted_features
* timestamps
* model_version (optional)
* reference_data_hash
* production_data_hash

---

## 7. Integrations

### 7.1 FastAPI / Flask

Middleware to automatically collect:

* input features
* predictions
* timestamps

Example:

```python
app.add_middleware(DriftMiddleware, monitor=monitor)
```

---

### 7.2 MLflow

Optional integration to:

* link drift reports to MLflow run IDs
* associate drift with model versions

---

### 7.3 Alerting

Supported alert backends:

* Slack Webhooks
* Email (SMTP)
* Custom webhooks

Alert triggers:

* drift score above threshold
* repeated drift events
* performance degradation

---

## 8. CLI Interface

Basic CLI for batch usage and CI/CD integration.

Examples:

```bash
driftwatch check --ref train.parquet --prod prod.parquet
driftwatch report --format json
```

---

## 9. Simulation Module

A dedicated module to simulate drift for testing and demonstration.

Supported simulations:

* mean shift
* variance increase
* noise injection
* label flipping

Example:

```python
from driftwatch.simulation import mean_shift

X_drifted = mean_shift(X, intensity=0.3)
```

---

## 10. Logging & Observability

* Structured logs (JSON)
* Drift events logged with:

  * feature name
  * drift score
  * threshold
  * model version
* Optional Prometheus metrics (future)

---

## 11. Architecture Overview

Modules:

* core: statistics, drift tests, report generation
* integrations: fastapi, mlflow, alerting
* cli: command-line interface
* simulation: drift generators
* tests: unit and integration tests

---

## 12. Quality & Testing

* Unit tests for all statistical methods
* Synthetic drift test cases
* CI via GitHub Actions
* Code formatting and linting

---

## 13. License

MIT License (open-source friendly).

---

## 14. Roadmap

### V1

* Core drift detection
* Python API
* CLI
* FastAPI middleware
* Slack alerts

### V2

* Dashboard
* Prometheus metrics
* Streaming support
* Advanced drift explainability

```

---

## ✅ Prochaine étape logique
Je peux maintenant :
- t’écrire le **README.md public**
- te générer le **squelette du repo**
- coder les **fonctions KS / PSI / Wasserstein**
- te proposer les **issues GitHub V1**

👉 Tu veux qu’on enchaîne sur **le README** ou **le code du core** ?
```
