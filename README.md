# 🔍 DriftWatch

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

> **Lightweight ML drift monitoring, built for real-world pipelines.**

DriftWatch is an open-source Python library for detecting **data drift** and **model drift** in machine learning systems. It's designed to integrate seamlessly with your existing ML infrastructure without requiring heavy dependencies or vendor lock-in.

---

## ✨ Features

- 📊 **Data Drift Detection** — KS Test, PSI, Wasserstein, Chi-Squared
- 🤖 **Model Drift Detection** — Prediction distribution, performance degradation
- 🔌 **Easy Integration** — FastAPI middleware, MLflow, Slack alerts
- 💻 **CLI Tool** — Batch processing and CI/CD integration
- 📈 **Actionable Reports** — Clear, structured drift reports
- 🧪 **Drift Simulation** — Built-in tools for testing

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "📦 DriftWatch Core"
        Monitor["🔍 Monitor"]
        Report["📊 DriftReport"]
        
        subgraph "🧪 Detectors"
            KS["KS Test"]
            PSI["PSI"]
            Wasserstein["Wasserstein"]
            Chi2["Chi-Squared"]
        end
    end
    
    subgraph "📥 Data Sources"
        RefData["📁 Reference Data<br/>(Training)"]
        ProdData["📁 Production Data"]
    end
    
    subgraph "🔌 Integrations"
        FastAPI["⚡ FastAPI<br/>Middleware"]
        CLI["💻 CLI"]
        MLflow["📈 MLflow"]
        Slack["💬 Slack<br/>Alerts"]
    end
    
    RefData --> Monitor
    ProdData --> Monitor
    Monitor --> KS & PSI & Wasserstein & Chi2
    KS & PSI & Wasserstein & Chi2 --> Report
    Report --> FastAPI & CLI & MLflow & Slack
    
    style Monitor fill:#4CAF50,color:#fff
    style Report fill:#2196F3,color:#fff
```

---

## 🔄 Drift Detection Workflow

```mermaid
sequenceDiagram
    participant User
    participant Monitor
    participant Detector
    participant Report
    participant Alert
    
    User->>Monitor: Initialize with reference_data
    Monitor->>Monitor: Setup detectors per feature type
    
    User->>Monitor: check(production_data)
    
    loop For each feature
        Monitor->>Detector: detect(ref_series, prod_series)
        Detector->>Detector: Calculate statistic
        Detector-->>Monitor: DetectionResult
    end
    
    Monitor->>Report: Create DriftReport
    Report->>Report: Compute status (OK/WARNING/CRITICAL)
    Report-->>User: Return report
    
    alt Drift Detected
        User->>Alert: send(report)
        Alert-->>User: 🚨 Notification sent
    end
```

---

## 📊 How Drift Detection Works

```mermaid
flowchart LR
    subgraph "Training Phase"
        Train["🎓 Train Model"]
        Save["💾 Save Reference<br/>Distribution"]
    end
    
    subgraph "Production Phase"
        Infer["🔮 Model Inference"]
        Collect["📥 Collect Data"]
    end
    
    subgraph "Monitoring Phase"
        Compare["⚖️ Compare<br/>Distributions"]
        Decision{Drift?}
        OK["✅ OK"]
        Alert["🚨 Alert"]
        Retrain["🔄 Retrain"]
    end
    
    Train --> Save
    Save --> Compare
    Infer --> Collect --> Compare
    Compare --> Decision
    Decision -->|No| OK
    Decision -->|Yes| Alert --> Retrain
    Retrain --> Train
    
    style Decision fill:#FF9800,color:#fff
    style Alert fill:#f44336,color:#fff
    style OK fill:#4CAF50,color:#fff
```

---

## 🧠 Decision Logic

```mermaid
graph TD
    Start["🔍 Check Production Data"]
    
    Start --> Loop["For each feature"]
    Loop --> TypeCheck{Numerical?}
    
    TypeCheck -->|Yes| NumDetector["Use PSI/KS Detector"]
    TypeCheck -->|No| CatDetector["Use Chi² Detector"]
    
    NumDetector --> CalcScore["Calculate Score"]
    CatDetector --> CalcScore
    
    CalcScore --> ThresholdCheck{Score > Threshold?}
    ThresholdCheck -->|Yes| MarkDrift["⚠️ Mark as Drift"]
    ThresholdCheck -->|No| MarkOK["✅ Mark as OK"]
    
    MarkDrift --> Aggregate
    MarkOK --> Aggregate
    
    Aggregate["Aggregate Results"]
    Aggregate --> RatioCheck{Drift Ratio}
    
    RatioCheck -->|0%| StatusOK["🟢 Status: OK"]
    RatioCheck -->|< 50%| StatusWarn["🟡 Status: WARNING"]
    RatioCheck -->|≥ 50%| StatusCrit["🔴 Status: CRITICAL"]
    
    style MarkDrift fill:#FF9800,color:#fff
    style StatusCrit fill:#f44336,color:#fff
    style StatusWarn fill:#FF9800,color:#fff
    style StatusOK fill:#4CAF50,color:#fff
```

---

## 🚀 Quick Start

### Installation

```bash
pip install driftwatch
```

### Basic Usage

```python
from driftwatch import Monitor

# Initialize with reference data
monitor = Monitor(
    reference_data=train_df,
    features=["age", "income", "category"],
    thresholds={"psi": 0.2, "ks_pvalue": 0.05}
)

# Check for drift
report = monitor.check(production_df)

# View results
print(report.summary())
print(f"Drift detected: {report.has_drift()}")
```

### CLI Usage

```bash
# Check drift between datasets
driftwatch check --ref train.parquet --prod prod.parquet

# Generate JSON report
driftwatch report --format json --output drift_report.json
```

### FastAPI Integration

```python
from fastapi import FastAPI
from driftwatch.integrations import DriftMiddleware

app = FastAPI()
app.add_middleware(DriftMiddleware, monitor=monitor)
```

---

## 📖 Documentation

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [CLI Guide](docs/cli-guide.md)
- [Integrations](docs/integrations.md)
- [Examples](examples/)

---

## 🧪 Supported Drift Tests

### Numerical Features

| Method | Description | Use Case |
|--------|-------------|----------|
| **KS Test** | Kolmogorov-Smirnov test | General distribution comparison |
| **PSI** | Population Stability Index | Production monitoring |
| **Wasserstein** | Earth Mover's Distance | Sensitive drift detection |

### Categorical Features

| Method | Description | Use Case |
|--------|-------------|----------|
| **Chi-Squared** | Chi-squared test | Category distribution |
| **Frequency PSI** | PSI on category frequencies | Production monitoring |

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/driftwatch.git
cd driftwatch

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
mypy src/
```

---

## 📊 Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development plan.

**V1 (Current Focus)**
- ✅ Core drift detection
- ✅ Python API
- ✅ CLI tool
- ✅ FastAPI middleware
- ✅ Slack alerts

**V2 (Planned)**
- 📊 Dashboard
- 📈 Prometheus metrics
- 🌊 Streaming support

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with ❤️ for the MLOps community.

Inspired by [Evidently](https://github.com/evidentlyai/evidently), [Great Expectations](https://github.com/great-expectations/great_expectations), and [WhyLabs](https://whylabs.ai/).
