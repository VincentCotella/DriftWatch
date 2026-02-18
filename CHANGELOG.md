# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.4.0] - 2026-02-18

### Added
- 🎯 **Prediction Drift Monitoring** (`PredictionMonitor`):
  - Monitor P(Ŷ) distribution changes between reference and production
  - Supports regression (1D) and classification (per-class probabilities)
  - Uses configurable detectors (PSI, KS, Jensen-Shannon, etc.)
- 🧠 **Concept Drift Monitoring** (`ConceptMonitor`):
  - Detect P(Y|X) changes by comparing performance metrics
  - Classification: accuracy, precision, recall, F1, AUC-ROC
  - Regression: MAE, MSE, RMSE, R², MAPE
  - Absolute and relative degradation modes
  - Zero external dependency (no sklearn required)
- 🔗 **DriftSuite** — Unified multi-drift monitoring:
  - Combines Feature, Prediction, and Concept drift in one interface
  - `ComprehensiveDriftReport` with clear per-type sections
  - `summary()`, `to_dict()`, `to_json()` for analysis and serialization
- 🏷️ **DriftType Enum** (`FEATURE`, `PREDICTION`, `CONCEPT`):
  - Every drift result now carries its drift type
  - Clear separation in reports and programmatic handling
- 📊 **New Detectors** for enhanced drift detection:
  - `JensenShannonDetector` — symmetric, bounded (0-1) divergence measure
  - `AndersonDarlingDetector` — tail-sensitive hypothesis test
  - `CramerVonMisesDetector` — overall distributional shape test
- 📧 **Email Alerting** (`EmailAlerter`):
  - SMTP-based email notifications with HTML formatting
  - Throttling, custom subjects, and extra recipients support
- 📈 **MLflow Integration** (`MLflowDriftTracker`):
  - Log drift metrics and reports to MLflow experiments
- 📓 **Notebooks**: Multi-drift monitoring tutorial, complete showcase
- 📖 Updated documentation: monitoring guide, drift types, multi-drift example
- ✅ Realistic integration tests (credit scoring, house pricing scenarios)

---

## [0.3.0] - 2026-02-09

### Added
- 🔍 **Drift Explain** module for understanding drift:
  - `DriftExplainer` class with detailed statistical analysis
  - Mean shift (absolute and percentage)
  - Standard deviation change
  - Quantile differences (configurable Q25, Q50, Q75)
  - Min/Max range changes
- 📊 **Visualization** support:
  - `DriftVisualizer` class for histogram overlays
  - `plot_feature()` for single feature visualization
  - `plot_all()` for multi-feature grid
  - `save()` for exporting to PNG/PDF/SVG
- 📓 Jupyter notebook tutorial for Colab
- `[viz]` optional dependency for matplotlib

### Changed
- Export `DriftExplainer` and `DriftVisualizer` at package level

---

## [0.2.0] - 2026-02-04

### Added
- 🎉 First public release on PyPI
- CLI with Typer/Rich (`driftwatch check`, `driftwatch report`)
- FastAPI `DriftMiddleware` for API monitoring
- Slack alerting via `SlackAlerter`
- Wasserstein distance detector
- MkDocs Material documentation site
- GitHub Pages deployment
- PyPI publishing workflow
- Integration tests

---

## [0.1.0] - 2026-02-03

### Added
- 🎉 Initial release
- Core `Monitor` class for drift detection
- `DriftReport` for structured results
- Statistical tests:
  - Kolmogorov-Smirnov (KS) test
  - Population Stability Index (PSI)
  - Wasserstein distance
  - Chi-squared test
- Simulation module for testing

---

<!-- Links -->
[Unreleased]: https://github.com/VincentCotella/DriftWatch/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/VincentCotella/DriftWatch/releases/tag/v0.4.0
[0.3.0]: https://github.com/VincentCotella/DriftWatch/releases/tag/v0.3.0
[0.2.0]: https://github.com/VincentCotella/DriftWatch/releases/tag/v0.2.0
[0.1.0]: https://github.com/VincentCotella/DriftWatch/releases/tag/v0.1.0

