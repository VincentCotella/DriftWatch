# 🏗️ DriftWatch Architecture

This document provides visual diagrams explaining DriftWatch's architecture, workflow, and decision logic.

---

## Overview

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

## Drift Detection Workflow

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

## How Drift Detection Works

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

## Decision Logic

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

## Component Structure

```mermaid
graph LR
    subgraph "src/driftwatch/"
        init["__init__.py"]
        
        subgraph "core/"
            monitor["monitor.py"]
            report["report.py"]
        end
        
        subgraph "detectors/"
            base["base.py"]
            numerical["numerical.py"]
            categorical["categorical.py"]
            registry["registry.py"]
        end
        
        subgraph "integrations/"
            fastapi_int["fastapi.py"]
            mlflow_int["mlflow.py"]
            alerting["alerting.py"]
        end
        
        subgraph "cli/"
            cli_main["main.py"]
        end
    end
    
    init --> monitor
    monitor --> registry
    registry --> numerical & categorical
    numerical & categorical --> base
```

---

## CI/CD Pipeline

```mermaid
flowchart TD
    subgraph "GitHub Actions"
        Push["📤 Push / PR"]
        
        subgraph "Lint Job"
            Ruff["Ruff Check"]
            Mypy["Mypy"]
            Format["Format Check"]
        end
        
        subgraph "Test Job"
            Test39["Python 3.9"]
            Test310["Python 3.10"]
            Test311["Python 3.11"]
            Test312["Python 3.12"]
        end
        
        subgraph "Build Job"
            Build["Build Package"]
            Check["Twine Check"]
            Artifact["Upload Artifact"]
        end
    end
    
    Push --> Ruff & Mypy & Format
    Ruff & Mypy & Format --> Test39 & Test310 & Test311 & Test312
    Test39 & Test310 & Test311 & Test312 --> Build
    Build --> Check --> Artifact
    
    style Push fill:#4CAF50,color:#fff
    style Artifact fill:#2196F3,color:#fff
```

---

## Git Workflow

```mermaid
gitGraph
    commit id: "Initial commit"
    branch develop
    checkout develop
    commit id: "feat: CLI"
    commit id: "feat: FastAPI"
    branch feature/slack-alerts
    checkout feature/slack-alerts
    commit id: "Add Slack"
    commit id: "Tests"
    checkout develop
    merge feature/slack-alerts id: "Merge PR #3"
    checkout main
    merge develop id: "Release v0.1.0" tag: "v0.1.0"
```

---

## Learn More

- [Getting Started](getting-started.md)
- [API Reference](api-reference.md)
- [Contributing](../CONTRIBUTING.md)
