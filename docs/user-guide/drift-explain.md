# Drift Explain

The **Drift Explain** module (introduced in v0.3.0) provides detailed insights into *why* drift was detected and *how* distributions have shifted.

## Overview

When drift is detected, you often want to understand:

- **How much** did the distribution shift?
- **Which statistics** changed the most?
- **What does it look like** visually?

DriftWatch provides two main classes for this:

| Class | Purpose |
|-------|---------|
| `DriftExplainer` | Detailed statistical analysis |
| `DriftVisualizer` | Histogram overlays and plots |

## Installation

For visualization support, install with the `viz` extra:

```bash
pip install driftwatch[viz]
```

## DriftExplainer

The `DriftExplainer` provides detailed statistics for understanding drift.

### Basic Usage

```python
from driftwatch import Monitor
from driftwatch.explain import DriftExplainer

# Setup
monitor = Monitor(reference_data=train_df)
report = monitor.check(production_df)

# Explain drift
explainer = DriftExplainer(train_df, production_df, report)
explanation = explainer.explain()

# Display summary
print(explanation.summary())
```

### Statistics Provided

For each numeric feature, you get:

| Statistic | Description |
|-----------|-------------|
| `mean_shift` | Absolute change in mean |
| `mean_shift_percent` | Relative change (%) |
| `std_change` | Absolute change in standard deviation |
| `std_change_percent` | Relative change (%) |
| `ref_min`, `prod_min` | Minimum values |
| `ref_max`, `prod_max` | Maximum values |
| `quantile_stats` | Q25, Q50, Q75 comparisons |

### Example Output

```
━━━ age ━━━
Status: 🔴 DRIFT DETECTED
Score (psi): 2.9555

📊 Central Tendency:
  Mean: 30.0234 → 40.1567 (+33.75%)

📈 Spread:
  Std: 5.0123 → 4.9876 (-0.49%)

📏 Range:
  Min: 15.2341 → 25.1234
  Max: 44.8765 → 55.3421

📐 Quantiles:
  Q25: 26.5432 → 36.4321 (+37.25%)
  Q50: 29.8765 → 40.0123 (+33.93%)
  Q75: 33.2109 → 43.7654 (+31.79%)
```

### Explain Single Feature

```python
# Get explanation for specific feature
age_exp = explainer.explain_feature("age")

print(f"Mean shifted by {age_exp.mean_shift_percent:.1f}%")
print(f"Std changed by {age_exp.std_change_percent:.1f}%")
```

### Custom Quantiles

```python
# Analyze different quantiles
explainer = DriftExplainer(
    train_df, 
    production_df, 
    report,
    quantiles=[0.1, 0.5, 0.9]  # 10th, 50th, 90th percentiles
)
```

### Export to JSON

```python
import json

# Export for logging/analysis
data = explanation.to_dict()
print(json.dumps(data, indent=2, default=str))
```

## DriftVisualizer

The `DriftVisualizer` creates histogram overlays to visualize distribution shifts.

### Basic Usage

```python
from driftwatch.explain import DriftVisualizer
import matplotlib.pyplot as plt

viz = DriftVisualizer(train_df, production_df, report)

# Plot single feature
fig = viz.plot_feature("age")
plt.show()
```

### Plot All Features

```python
# Grid of all numeric features
fig = viz.plot_all(cols=2)
plt.show()
```

### Customization

```python
# Customize appearance
fig = viz.plot_feature(
    "age",
    bins=30,              # Number of histogram bins
    figsize=(12, 8),      # Figure size
    show_stats=True,      # Show stats box
    alpha=0.7             # Transparency
)
```

### Save to File

```python
# Save single feature
viz.save("age_drift.png", feature_name="age", dpi=150)

# Save all features
viz.save("drift_report.png", dpi=150)
viz.save("drift_report.pdf")  # Vector format
```

## Complete Example

```python
import pandas as pd
import numpy as np
from driftwatch import Monitor
from driftwatch.explain import DriftExplainer, DriftVisualizer

# Generate data
np.random.seed(42)
train_df = pd.DataFrame({
    'age': np.random.normal(30, 5, 1000),
    'income': np.random.normal(50000, 10000, 1000),
})

prod_df = pd.DataFrame({
    'age': np.random.normal(40, 5, 1000),  # Drift!
    'income': np.random.normal(50000, 10000, 1000),
})

# Detect drift
monitor = Monitor(reference_data=train_df)
report = monitor.check(prod_df)

# Explain
explainer = DriftExplainer(train_df, prod_df, report)
explanation = explainer.explain()
print(explanation.summary())

# Visualize
viz = DriftVisualizer(train_df, prod_df, report)
viz.save("drift_analysis.png")
```

## API Reference

::: driftwatch.explain.DriftExplainer
    options:
      show_source: false
      heading_level: 3

::: driftwatch.explain.DriftVisualizer
    options:
      show_source: false
      heading_level: 3
