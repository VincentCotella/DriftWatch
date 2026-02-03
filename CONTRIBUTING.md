# Contributing to DriftWatch

First off, thank you for considering contributing to DriftWatch! 🎉

This document provides guidelines and information about contributing to this project.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

---

## 📜 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

---

## 🚀 Getting Started

### Find Something to Work On

- Check the [Issues](https://github.com/YOUR_USERNAME/driftwatch/issues) page
- Look for issues labeled `good first issue` or `help wanted`
- Check the [ROADMAP.md](ROADMAP.md) for planned features

### Before You Start

1. **Check existing issues** — Make sure no one else is already working on it
2. **Open an issue first** — For significant changes, discuss your approach
3. **Keep scope small** — Smaller PRs are easier to review and merge

---

## 💻 Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- A virtual environment tool (venv, conda, etc.)

### Setup Steps

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/driftwatch.git
cd driftwatch

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install development dependencies
pip install -e ".[dev,all]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Verify setup
pytest
ruff check .
mypy src/
```

---

## 🔧 Making Changes

### Branch Naming

Use descriptive branch names:

```
feature/add-wasserstein-distance
fix/psi-calculation-edge-case
docs/update-api-reference
test/add-ks-test-coverage
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding or updating tests
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `perf`: Performance improvement
- `chore`: Maintenance tasks

**Examples:**
```
feat(detectors): add Wasserstein distance detector
fix(monitor): handle empty dataframes gracefully
docs(readme): add FastAPI integration example
test(psi): add edge case tests for zero frequencies
```

---

## 📬 Submitting a Pull Request

### Before Submitting

1. **Run tests**: `pytest`
2. **Run linting**: `ruff check . && ruff format --check .`
3. **Run type checking**: `mypy src/`
4. **Update documentation** if needed
5. **Add tests** for new functionality

### PR Template

Your PR should include:

- [ ] Clear description of the change
- [ ] Link to related issue(s)
- [ ] Tests for new functionality
- [ ] Documentation updates
- [ ] Changelog entry (if applicable)

### PR Process

1. Create a PR against the `main` branch
2. Fill out the PR template
3. Wait for CI to pass
4. Request review from maintainers
5. Address review feedback
6. Squash and merge once approved

---

## 📏 Coding Standards

### Style Guide

- **Formatter**: Black (line length 88)
- **Linter**: Ruff
- **Type Checker**: mypy (strict mode)

### Code Principles

```python
# ✅ Good: Type hints, docstrings, clear naming
def calculate_psi(
    reference: pd.Series,
    production: pd.Series,
    buckets: int = 10,
) -> float:
    """
    Calculate Population Stability Index.
    
    Args:
        reference: Reference distribution data
        production: Production distribution data
        buckets: Number of buckets for binning
        
    Returns:
        PSI score (0 = no drift, >0.2 = significant drift)
        
    Raises:
        ValueError: If input series are empty
    """
    if reference.empty or production.empty:
        raise ValueError("Input series cannot be empty")
    ...

# ❌ Bad: No types, no docs, unclear naming
def calc(r, p, b=10):
    ...
```

### Import Order

Managed by `ruff` isort:

```python
# Standard library
from typing import Any

# Third-party
import numpy as np
import pandas as pd

# Local
from driftwatch.detectors import BaseDetector
```

---

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/
│   ├── test_detectors/
│   │   ├── test_ks.py
│   │   ├── test_psi.py
│   │   └── test_wasserstein.py
│   ├── test_monitor.py
│   └── test_report.py
├── integration/
│   ├── test_fastapi_middleware.py
│   └── test_cli.py
└── conftest.py
```

### Writing Tests

```python
import pytest
import pandas as pd
from driftwatch.detectors import KSDetector


class TestKSDetector:
    """Tests for Kolmogorov-Smirnov detector."""
    
    @pytest.fixture
    def detector(self) -> KSDetector:
        return KSDetector(threshold=0.05)
    
    @pytest.fixture
    def sample_data(self) -> tuple[pd.Series, pd.Series]:
        return (
            pd.Series([1, 2, 3, 4, 5]),
            pd.Series([1, 2, 3, 4, 5]),
        )
    
    def test_no_drift_identical_data(
        self, 
        detector: KSDetector, 
        sample_data: tuple[pd.Series, pd.Series]
    ) -> None:
        """Should not detect drift for identical distributions."""
        ref, prod = sample_data
        result = detector.detect(ref, prod)
        assert not result.has_drift
    
    def test_drift_different_distributions(
        self, 
        detector: KSDetector
    ) -> None:
        """Should detect drift for different distributions."""
        ref = pd.Series(range(100))
        prod = pd.Series(range(50, 150))
        result = detector.detect(ref, prod)
        assert result.has_drift
```

### Coverage Requirements

- Minimum 80% coverage for new code
- Critical paths (detectors, monitor) should have >90% coverage
- Run coverage: `pytest --cov=src/driftwatch --cov-report=html`

---

## 📚 Documentation

### Docstrings

Use Google-style docstrings:

```python
def function(arg1: str, arg2: int = 10) -> bool:
    """
    Short description of function.
    
    Longer description if needed, explaining behavior,
    edge cases, or implementation details.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When arg1 is empty
        
    Example:
        >>> function("test", 5)
        True
    """
```

### Updating Docs

- API changes → Update docstrings + `docs/api-reference.md`
- New features → Add to `docs/` and `README.md`
- Examples → Add to `examples/` folder

---

## ❓ Questions?

- Open a [Discussion](https://github.com/YOUR_USERNAME/driftwatch/discussions)
- Check existing issues and PRs
- Reach out to maintainers

---

Thank you for contributing! 🙏
