# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
[Unreleased]: https://github.com/VincentCotella/DriftWatch/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/VincentCotella/DriftWatch/releases/tag/v0.3.0
[0.2.0]: https://github.com/VincentCotella/DriftWatch/releases/tag/v0.2.0
[0.1.0]: https://github.com/VincentCotella/DriftWatch/releases/tag/v0.1.0
