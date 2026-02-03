# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Initial project structure
- Core documentation (README, CONTRIBUTING, ROADMAP)
- Development tooling configuration (ruff, black, mypy, pytest)

---

## [0.1.0] - TBD

### Added
- 🎉 Initial release
- Core `Monitor` class for drift detection
- `DriftReport` for structured results
- Statistical tests:
  - Kolmogorov-Smirnov (KS) test
  - Population Stability Index (PSI)
  - Wasserstein distance
  - Chi-squared test
- CLI commands (`check`, `report`)
- FastAPI middleware integration
- Slack webhook alerting
- Simulation module for testing

---

<!-- Links -->
[Unreleased]: https://github.com/YOUR_USERNAME/driftwatch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/YOUR_USERNAME/driftwatch/releases/tag/v0.1.0
