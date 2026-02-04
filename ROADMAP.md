# 🗺️ DriftWatch — Roadmap

> **Tagline:** *Lightweight ML drift monitoring, built for real-world pipelines.*

---

## 📦 Current Version: `v0.2.0` ✅

Released on **2024-02-04**. Available via `pip install driftwatch`.

---

## ✅ Completed

### Phase 1: Foundation
| Task | Status |
|------|--------|
| Setup repo structure (src layout) | ✅ Done |
| Configure `pyproject.toml` with extras | ✅ Done |
| Setup pre-commit hooks (ruff, mypy) | ✅ Done |
| Configure GitHub Actions CI | ✅ Done |
| Write README.md | ✅ Done |
| Create CONTRIBUTING.md | ✅ Done |
| Setup pytest + coverage (96%+) | ✅ Done |

### Phase 2: Core Engine
| Task | Status |
|------|--------|
| Implement `BaseDetector` abstract class | ✅ Done |
| Implement KS Test (numerical) | ✅ Done |
| Implement PSI (numerical) | ✅ Done |
| Implement Wasserstein Distance | ✅ Done (v0.2.0) |
| Implement Chi-Squared Test (categorical) | ✅ Done |
| Implement `Monitor` class | ✅ Done |
| Implement `DriftReport` class | ✅ Done |
| Threshold configuration | ✅ Done |
| Unit & Integration tests | ✅ Done |

### Phase 3: Integrations
| Task | Status |
|------|--------|
| CLI with Typer/Rich (`driftwatch check`, `report`) | ✅ Done (v0.2.0) |
| FastAPI `DriftMiddleware` | ✅ Done (v0.2.0) |
| Slack Alerting (`SlackAlerter`) | ✅ Done (v0.2.0) |
| JSON/Table output formats | ✅ Done |

### Phase 4: Documentation & Release
| Task | Status |
|------|--------|
| MkDocs Material documentation site | ✅ Done |
| API Reference (mkdocstrings) | ✅ Done |
| GitHub Pages deployment (CI) | ✅ Done |
| PyPI publishing workflow | ✅ Done |
| First public release (v0.2.0) | ✅ Done |

---

## 🚧 In Progress / Next Up

### v0.3.0 — Enhancements

| Task | Priority | Status |
|------|----------|--------|
| MLflow integration (log drift to experiments) | P2 | 🔲 Todo |
| Email alerting (SMTP) | P2 | 🔲 Todo |
| Example Jupyter notebooks | P1 | 🔲 Todo |
| More detectors (Jensen-Shannon, etc.) | P2 | 🔲 Todo |

---

## 🔮 Future Backlog (v1.0+)

| Feature | Description | Priority |
|---------|-------------|----------|
| 📊 Dashboard | Streamlit/Gradio drift visualization | P2 |
| 📈 Prometheus | Native metrics export (`/metrics` endpoint) | P2 |
| 🌊 Streaming | Kafka/Redis streaming support | P3 |
| 🧠 Explainability | SHAP-based drift explanation | P3 |
| 🔄 Auto-retrain | Trigger retraining pipelines on drift | P3 |
| 🐳 Docker | Official Docker image | P3 |

---

## 📊 Progress Tracker

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Foundation | ✅ Complete | 100% |
| Phase 2: Core Engine | ✅ Complete | 100% |
| Phase 3: Integrations | ✅ Complete | 100% |
| Phase 4: Docs & Release | ✅ Complete | 100% |
| Phase 5: Enhancements | 🔲 Not started | 0% |

---

## 🏁 Definition of Done (DoD)

Each feature must satisfy:

- [x] Code implemented with type hints
- [x] Unit tests written (>80% coverage)
- [x] Docstrings for public API
- [x] No linting errors (ruff, mypy)
- [x] PR reviewed
- [x] CHANGELOG updated

---

*Last updated: 2024-02-04*
