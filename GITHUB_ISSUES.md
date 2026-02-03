# 📋 GitHub Issues à Créer

Ces issues sont prêtes à être créées sur https://github.com/VincentCotella/DriftWatch/issues/new

---

## Issue 1: Implement CLI with Typer

**Labels:** `cli`, `P0`, `enhancement`

### Description
Implement a CLI interface using Typer for batch processing and CI/CD integration.

### Acceptance Criteria
- [ ] Setup Typer framework in `src/driftwatch/cli/`
- [ ] Implement `driftwatch check --ref <file> --prod <file>` command
- [ ] Implement `driftwatch report --format json|table` command
- [ ] Add `--threshold` options for custom thresholds
- [ ] Support parquet and CSV file formats
- [ ] Add colored output with Rich
- [ ] Write CLI integration tests

### Example Usage
```bash
driftwatch check --ref train.parquet --prod prod.parquet
driftwatch report --format json --output drift_report.json
```

### Technical Notes
- Use `typer[all]` for CLI framework
- Use `rich` for beautiful terminal output
- Entry point already configured in `pyproject.toml`

---

## Issue 2: Implement FastAPI DriftMiddleware

**Labels:** `integration`, `P0`, `enhancement`

### Description
Implement FastAPI middleware for automatic drift monitoring on ML inference endpoints.

### Acceptance Criteria
- [ ] Create `DriftMiddleware` class in `src/driftwatch/integrations/fastapi.py`
- [ ] Auto-collect input features from request body
- [ ] Auto-collect predictions from response
- [ ] Store samples in configurable buffer
- [ ] Run drift checks on configurable schedule (e.g., every N requests)
- [ ] Expose `/drift/status` endpoint for checking current drift status
- [ ] Add background task for non-blocking drift computation

### Example Usage
```python
from fastapi import FastAPI
from driftwatch.integrations import DriftMiddleware

app = FastAPI()
app.add_middleware(
    DriftMiddleware, 
    monitor=monitor,
    check_interval=100  # Check every 100 requests
)
```

### Technical Notes
- Use Starlette middleware pattern
- Consider using `BackgroundTasks` for async drift computation
- Buffer should be configurable (size, persistence)

---

## Issue 3: Add Slack webhook alerting

**Labels:** `integration`, `P1`, `enhancement`

### Description
Add Slack webhook integration for drift alerting.

### Acceptance Criteria
- [ ] Create `SlackAlerter` class in `src/driftwatch/integrations/alerting.py`
- [ ] Support Slack webhook URLs
- [ ] Format drift reports as Slack Block Kit messages
- [ ] Include feature-level details in alerts
- [ ] Add alert throttling (avoid spam)
- [ ] Support custom message templates

### Example Usage
```python
from driftwatch.integrations import SlackAlerter

alerter = SlackAlerter(
    webhook_url="https://hooks.slack.com/...",
    throttle_minutes=60  # Max 1 alert per hour
)

if report.has_drift():
    alerter.send(report)
```

### Message Format
```
🚨 Drift Detected - DriftWatch

Status: WARNING
Drifted Features: age, income (2/5)
Timestamp: 2024-01-15 14:30:00 UTC

Details:
• age: PSI=0.35 (threshold=0.20)
• income: PSI=0.28 (threshold=0.20)
```

---

## Issue 4: Add Wasserstein detector to registry

**Labels:** `core`, `P1`, `enhancement`

### Description
Add Wasserstein distance detector to the registry for automatic selection.

### Current State
- `WassersteinDetector` is implemented in `detectors/numerical.py`
- But it's not registered in `registry.py`

### Acceptance Criteria
- [ ] Add Wasserstein to `get_detector_by_name()` function
- [ ] Add option to use Wasserstein as default for numerical features
- [ ] Update `Monitor` to support detector selection per feature
- [ ] Add unit tests for Wasserstein detector
- [ ] Document when to use Wasserstein vs PSI vs KS

### Technical Notes
Wasserstein (Earth Mover's Distance) is more sensitive to subtle distributional changes and should be offered as an option for users who need higher sensitivity.

---

## Issue 5: Implement simulation module for synthetic drift

**Labels:** `core`, `P2`, `enhancement`

### Description
Implement the simulation module for generating synthetic drift for testing and demos.

### Acceptance Criteria
- [ ] Implement `mean_shift(X, intensity)` - shift distribution mean
- [ ] Implement `variance_increase(X, factor)` - increase variance
- [ ] Implement `noise_injection(X, noise_level)` - add random noise
- [ ] Implement `label_flip(y, flip_rate)` - flip labels for model drift
- [ ] Implement `category_shift(X, new_distribution)` - change category frequencies
- [ ] Add type hints and comprehensive docstrings
- [ ] Write unit tests for all simulation functions

### Example Usage
```python
from driftwatch.simulation import mean_shift, noise_injection

# Create drifted data for testing
X_drifted = mean_shift(X_train, intensity=0.5)
X_noisy = noise_injection(X_train, noise_level=0.1)
```

### Use Cases
- Testing drift detection thresholds
- Demos and tutorials
- CI/CD pipeline testing

---

## Issue 6: Setup MkDocs documentation

**Labels:** `docs`, `P1`, `enhancement`

### Description
Setup MkDocs with Material theme for beautiful documentation.

### Acceptance Criteria
- [ ] Configure `mkdocs.yml` with Material theme
- [ ] Create documentation structure:
  - Getting Started
  - API Reference (auto-generated from docstrings)
  - CLI Guide
  - Integrations (FastAPI, MLflow, Alerting)
  - Examples
- [ ] Add code syntax highlighting
- [ ] Setup GitHub Pages deployment
- [ ] Add search functionality
- [ ] Include logo and branding

### Technical Notes
- Use `mkdocstrings` for auto-generating API docs from docstrings
- Deploy to GitHub Pages via GitHub Actions
- Add docs badge to README

---

## Issue 7: Improve docstrings across codebase

**Labels:** `docs`, `good first issue`, `P2`

### Description
Improve docstrings across the codebase to follow Google style consistently.

### Scope
- [ ] `src/driftwatch/core/monitor.py` - Complete docstrings
- [ ] `src/driftwatch/core/report.py` - Add examples
- [ ] `src/driftwatch/detectors/*.py` - Add mathematical explanations
- [ ] All public methods should have docstrings

### Docstring Format
Use Google-style docstrings:
```python
def function(arg1: str, arg2: int = 10) -> bool:
    """
    Short description.
    
    Longer description with details.
    
    Args:
        arg1: Description
        arg2: Description
        
    Returns:
        Description
        
    Raises:
        ValueError: When...
        
    Example:
        >>> function("test")
        True
    """
```

### Good First Issue
This is a great issue for first-time contributors!

---

## Issue 8: Add integration tests with realistic data

**Labels:** `test`, `P1`, `enhancement`

### Description
Add integration tests using realistic datasets to validate end-to-end drift detection.

### Acceptance Criteria
- [ ] Create `tests/integration/` directory
- [ ] Add test with real-world-like synthetic data (1000+ samples)
- [ ] Test full workflow: Monitor → check → DriftReport
- [ ] Test edge cases:
  - Empty dataframes
  - Single sample
  - All NaN values
  - Extreme outliers
- [ ] Test with mixed dtypes (numerical + categorical)
- [ ] Add performance benchmarks (time to process N samples)

### Test Data
Consider using:
- Sklearn datasets (iris, boston, etc.)
- Synthetic data with known drift patterns
- Edge case generators

### Markers
Use `@pytest.mark.integration` for these tests so they can be run separately.

---

## 🏷️ Labels à Créer

Créer ces labels sur https://github.com/VincentCotella/DriftWatch/labels

| Label | Color | Description |
|-------|-------|-------------|
| `core` | `#1d76db` | Core library functionality |
| `cli` | `#5319e7` | Command-line interface |
| `integration` | `#0e8a16` | External integrations |
| `docs` | `#fbca04` | Documentation |
| `test` | `#bfd4f2` | Testing related |
| `good first issue` | `#7057ff` | Good for newcomers |
| `P0` | `#b60205` | Critical priority |
| `P1` | `#d93f0b` | High priority |
| `P2` | `#fbca04` | Medium priority |
| `enhancement` | `#a2eeef` | New feature or request |
