# Installation

## Requirements

- Python 3.9 or higher
- pip or poetry

## Basic Installation

Install DriftWatch from PyPI:

```bash
pip install driftwatch
```

This installs the core library with minimal dependencies:

- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `scipy` - Statistical functions
- `pydantic` - Data validation

## Optional Dependencies

DriftWatch provides optional extras for specific use cases:

### CLI Tools

For command-line interface with rich formatting:

```bash
pip install driftwatch[cli]
```

Includes:

- `typer` - CLI framework
- `rich` - Beautiful terminal output

### FastAPI Integration

For automatic API monitoring:

```bash
pip install driftwatch[fastapi]
```

Includes:

- `fastapi` - Web framework
- `uvicorn` - ASGI server

### Alerting

For Slack notifications and email alerts:

```bash
pip install driftwatch[alerting]
```

Includes:

- `httpx` - HTTP client for webhooks
- `aiosmtplib` - Async SMTP client

### MLflow Integration

For experiment tracking integration:

```bash
pip install driftwatch[mlflow]
```

Includes:

- `mlflow` - ML experiment tracking

### All Features

Install everything:

```bash
pip install driftwatch[all]
```

## Development Installation

For contributors:

```bash
# Clone the repository
git clone https://github.com/VincentCotella/DriftWatch.git
cd DriftWatch

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev,all]"

# Install pre-commit hooks
pre-commit install
```

## Verify Installation

Test your installation:

```python
import driftwatch
print(driftwatch.__version__)
```

Or run a quick check:

```python
from driftwatch import Monitor
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    "age": np.random.normal(35, 10, 100),
    "income": np.random.lognormal(10.5, 0.5, 100)
})

# Create monitor
monitor = Monitor(reference_data=data)
print("✓ DriftWatch installed successfully!")
```

## Next Steps

- [Quickstart →](quickstart.md) - Run your first drift check
- [Core Concepts →](concepts.md) - Understand how drift detection works
