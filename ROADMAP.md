# 🗺️ DriftWatch — Roadmap

> **Tagline:** *Lightweight ML drift monitoring, built for real-world pipelines.*

---

## 📅 Timeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Week 1-2          │  Week 3-4          │  Week 5-6          │  Week 7+     │
│  Foundation        │  Core Features     │  Integrations      │  Polish      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Phase 1: Foundation (Week 1-2)

### Milestone: `v0.1.0-alpha` — Project Bootstrap

| Task | Priority | Effort | Owner |
|------|----------|--------|-------|
| ✅ Setup repo structure | P0 | 2h | - |
| ✅ Configure pyproject.toml | P0 | 1h | - |
| ✅ Setup pre-commit hooks (ruff, black, mypy) | P0 | 1h | - |
| ✅ Configure GitHub Actions CI | P0 | 2h | - |
| ✅ Write README.md skeleton | P0 | 2h | - |
| ✅ Create CONTRIBUTING.md | P1 | 1h | - |
| ✅ Setup pytest + coverage | P0 | 1h | - |

**Deliverable:** Empty but fully configured Python package

---

## 🔬 Phase 2: Core Engine (Week 3-4)

### Milestone: `v0.2.0-alpha` — Statistical Tests

| Task | Priority | Effort | Status |
|------|----------|--------|--------|
| Implement `BaseDetector` abstract class | P0 | 2h | 🔲 |
| Implement KS Test (numerical) | P0 | 3h | 🔲 |
| Implement PSI (numerical + categorical) | P0 | 4h | 🔲 |
| Implement Wasserstein Distance | P0 | 2h | 🔲 |
| Implement Chi-Squared Test (categorical) | P0 | 2h | 🔲 |
| Unit tests for all detectors | P0 | 4h | 🔲 |

### Milestone: `v0.3.0-alpha` — Monitor & Report

| Task | Priority | Effort | Status |
|------|----------|--------|--------|
| Implement `Monitor` class | P0 | 4h | 🔲 |
| Implement `DriftReport` class | P0 | 3h | 🔲 |
| Add threshold configuration | P0 | 2h | 🔲 |
| Implement `to_dict()` / `to_json()` | P1 | 1h | 🔲 |
| Implement `summary()` display | P1 | 2h | 🔲 |
| Integration tests | P0 | 3h | 🔲 |

**Deliverable:** Working Python API for drift detection

---

## 🔌 Phase 3: Integrations (Week 5-6)

### Milestone: `v0.4.0-beta` — CLI

| Task | Priority | Effort | Status |
|------|----------|--------|--------|
| Setup Click/Typer CLI framework | P0 | 1h | 🔲 |
| Implement `driftwatch check` command | P0 | 3h | 🔲 |
| Implement `driftwatch report` command | P0 | 2h | 🔲 |
| Add JSON/table output formats | P1 | 2h | 🔲 |
| CLI integration tests | P0 | 2h | 🔲 |

### Milestone: `v0.5.0-beta` — FastAPI Middleware

| Task | Priority | Effort | Status |
|------|----------|--------|--------|
| Implement `DriftMiddleware` class | P0 | 4h | 🔲 |
| Auto-collect features + predictions | P0 | 3h | 🔲 |
| Background drift computation | P1 | 3h | 🔲 |
| Example FastAPI app | P1 | 2h | 🔲 |

### Milestone: `v0.6.0-beta` — Alerting

| Task | Priority | Effort | Status |
|------|----------|--------|--------|
| Implement `AlertManager` base class | P0 | 2h | 🔲 |
| Slack webhook integration | P0 | 2h | 🔲 |
| Email (SMTP) integration | P1 | 2h | 🔲 |
| Custom webhook support | P1 | 1h | 🔲 |
| Alert throttling logic | P1 | 2h | 🔲 |

**Deliverable:** CLI + FastAPI middleware + Slack alerts working

---

## ✨ Phase 4: Polish & Release (Week 7+)

### Milestone: `v1.0.0` — Production Ready

| Task | Priority | Effort | Status |
|------|----------|--------|--------|
| Simulation module (`mean_shift`, `noise`, etc.) | P1 | 4h | 🔲 |
| MLflow integration (optional) | P2 | 3h | 🔲 |
| Complete documentation (MkDocs) | P0 | 6h | 🔲 |
| API reference docs | P0 | 4h | 🔲 |
| Publish to PyPI | P0 | 2h | 🔲 |
| Example notebooks | P1 | 4h | 🔲 |
| Performance benchmarks | P2 | 3h | 🔲 |

**Deliverable:** `pip install driftwatch` works! 🎉

---

## 🔮 V2 Backlog (Future)

| Feature | Description | Priority |
|---------|-------------|----------|
| 📊 Dashboard | Streamlit/Gradio drift dashboard | P2 |
| 📈 Prometheus | Native metrics export | P2 |
| 🌊 Streaming | Kafka/Redis streaming support | P3 |
| 🧠 Explainability | SHAP-based drift explanation | P3 |
| 🔄 Auto-retrain triggers | Send retraining signals | P3 |

---

## 📦 GitHub Issues Template

### Issue Labels

| Label | Color | Description |
|-------|-------|-------------|
| `core` | `#1d76db` | Core library functionality |
| `cli` | `#5319e7` | Command-line interface |
| `integration` | `#0e8a16` | External integrations |
| `docs` | `#fbca04` | Documentation |
| `test` | `#bfd4f2` | Testing related |
| `good first issue` | `#7057ff` | Good for newcomers |
| `help wanted` | `#008672` | Extra attention needed |
| `P0` | `#b60205` | Critical priority |
| `P1` | `#d93f0b` | High priority |
| `P2` | `#fbca04` | Medium priority |

---

## 🏁 Definition of Done (DoD)

Each feature must satisfy:

- [ ] Code implemented with type hints
- [ ] Unit tests written (>80% coverage)
- [ ] Docstrings for public API
- [ ] No linting errors (ruff, mypy)
- [ ] PR reviewed
- [ ] CHANGELOG updated

---

## 📊 Progress Tracker

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Foundation | 🔲 Not started | 0% |
| Phase 2: Core Engine | 🔲 Not started | 0% |
| Phase 3: Integrations | 🔲 Not started | 0% |
| Phase 4: Polish | 🔲 Not started | 0% |

---

*Last updated: 2026-02-03*
