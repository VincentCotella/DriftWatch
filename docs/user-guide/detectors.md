# Drift Detectors - When to Use Each

This guide helps you choose the right drift detector for your use case.

## Numerical Features

### PSI (Population Stability Index)
**Best for**: General-purpose numerical drift detection

- **How it works**: Bins data and compares distribution of samples across bins
- **Threshold**: Typically 0.1-0.25
- **Interpretation**:
  - PSI < 0.1: No significant drift
  - 0.1 ≤ PSI < 0.2: Moderate drift (investigate)
  - PSI ≥ 0.2: Significant drift (action required)

**Use when**:
- You want a standard, industry-proven drift detector
- Your feature has a reasonable range and distribution
- You need interpretable thresholds

### KS (Kolmogorov-Smirnov Test)
**Best for**: Statistical hypothesis testing

- **How it works**: Measures maximum distance between cumulative distributions
- **Threshold**: P-value (typically 0.05)
- **Interpretation**: P-value < threshold indicates drift

**Use when**:
- You need a formal statistical test
- You want p-values for significance testing
- Sample sizes are moderate to large (> 100)

### Wasserstein Distance (Earth Mover's Distance)
**Best for**: Detecting subtle distributional shifts

- **How it works**: Calculates minimum "work" to transform one distribution into another
- **Threshold**: Typically 0.1-0.5 (normalized by reference std)
- **Interpretation**: Higher values indicate more drift

**Use when**:
- You need high sensitivity to subtle changes
- Distribution shape matters (not just mean/variance)
- You're monitoring critical features
- Small shifts are important to detect early

**Advantages over PSI/KS**:
- More sensitive to gradual shifts
- Continuous metric (no binning artifacts)
- Better for multimodal distributions

## Categorical Features

### Chi-Squared Test
**Best for**: Categorical drift detection

- **How it works**: Tests if category frequencies have changed significantly
- **Threshold**: P-value (typically 0.05)
- **Interpretation**: P-value < threshold indicates drift

**Use when**:
- Feature has discrete categories
- You need statistical significance testing
- Sample size is adequate (expected count > 5 per category)

## Choosing Between Detectors

### Quick Decision Tree

```
Is your feature numerical?
├─ Yes → Choose based on sensitivity needs:
│   ├─ Standard monitoring → PSI (threshold=0.2)
│   ├─ Need p-values → KS (threshold=0.05)
│   └─ High sensitivity needed → Wasserstein (threshold=0.1)
└─ No (categorical) → Chi-Squared (threshold=0.05)
```

### Comparison Table

| Detector | Type | Sensitivity | Interpretability | Speed | Best For |
|----------|------|-------------|------------------|-------|----------|
| PSI | Numerical | Medium | High | Fast | General use |
| KS | Numerical | Medium | Medium | Fast | Hypothesis testing |
| Wasserstein | Numerical | High | Medium | Medium | Subtle shifts |
| Chi² | Categorical | Medium | High | Fast | Categories |

## Examples

### Using Different Detectors

```python
from driftwatch import Monitor

# Default (PSI for numerical, Chi² for categorical)
monitor = Monitor(reference_data=train_df)

# Explicit detector per feature
monitor = Monitor(
    reference_data=train_df,
    thresholds={
        "psi": 0.15,
        "wasserstein": 0.2,
        "ks_pvalue": 0.05,
    }
)

# Using registry to get specific detector
from driftwatch.detectors.registry import get_detector_by_name

wasserstein = get_detector_by_name(
    "wasserstein", 
    thresholds={"wasserstein": 0.1}
)
```

## Advanced Tips

### When to Use Wasserstein

Wasserstein is particularly useful for:
- **Time series features**: Detect gradual temporal changes
- **High-stakes predictions**: Early warning for critical models
- **Multimodal distributions**: Better than PSI which bins data
- **Small sample sizes**: More robust than KS test

### Threshold Tuning

Start with defaults and adjust based on:
1. **Sensitivity needs**: Lower threshold = more sensitive
2. **False positive tolerance**: Higher threshold = fewer alerts
3. **Historical data**: Analyze typical variation in your domain

### Combining Detectors

For critical features, monitor with multiple detectors:
```python
# Monitor age with both PSI and Wasserstein
psi_result = psi_detector.detect(ref["age"], prod["age"])
wass_result = wasserstein_detector.detect(ref["age"], prod["age"])

if psi_result.has_drift or wass_result.has_drift:
    alert("Drift detected in age feature")
```
