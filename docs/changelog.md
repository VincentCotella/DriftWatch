# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-02-04

### Added

- **FastAPI Integration**: Added `DriftMiddleware` for automatic drift monitoring of API endpoints.
- **CLI**: Added `driftwatch` command-line interface with `check` and `report` commands.
- **Alerting**: Added `SlackAlerter` for real-time drift notifications.
- **New Detector**: Added `WassersteinDetector` (Earth Mover's Distance) for more robust drift detection.
- **Documentation**: Complete rewrite with MkDocs Material, comprehensive guides, and API reference.
- **CI/CD**: Added GitHub Actions workflows for testing, publishing, and documentation deployment.

### Changed

- **Core**: Improved type hints across the codebase (now 100% typed).
- **Core**: Refactored detector registry to separate numerical and categorical detectors.
- **Config**: Updated default thresholds for better out-of-the-box performance.

### Fixed

- **Dependencies**: Fixed missing `typer`, `rich`, and `httpx` optional dependencies.
- **Tests**: Fixed flakey tests in registry and alerting modules.

## [0.1.0] - 2024-01-28

- Initial release.
- Basic `Monitor` class.
- PSI and KS-Test detectors.
- Simple JSON reporting.
