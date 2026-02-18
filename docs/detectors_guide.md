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

### Jensen-Shannon Divergence *(New in v0.3.0)*
**Best for**: Symmetric, bounded measure of distribution similarity

- **How it works**: Computes the average KL divergence of each distribution to the mixture
- **Threshold**: Typically 0.05-0.2 (range: 0 to 1 with base-2)
- **Interpretation**:
  - JSD = 0: Identical distributions
  - JSD = 1: Maximally different distributions
  - JSD ≥ threshold: Drift detected

**Use when**:
- You need a symmetric measure (JSD(P||Q) = JSD(Q||P))
- You want a bounded score that's easy to interpret (0-1)
- You need a more robust alternative to KL divergence
- You're comparing distributions that may have non-overlapping support

**Advantages over PSI/KS**:
- Always finite (unlike KL divergence)
- Symmetric by design
- Bounded range makes thresholds more intuitive
- Robust to zero-probability bins

```python
from driftwatch.detectors.numerical import JensenShannonDetector

detector = JensenShannonDetector(threshold=0.1, buckets=50)
result = detector.detect(reference_series, production_series)
print(f"JSD: {result.score:.4f} (0=identical, 1=maximally different)")
```

### Anderson-Darling Test *(New in v0.3.0)*
**Best for**: Detecting differences in distribution tails

- **How it works**: A modified KS test that weights the tails more heavily
- **Threshold**: P-value (typically 0.05)
- **Interpretation**: P-value < threshold indicates drift

**Use when**:
- Tail behavior matters (e.g., extreme values, outliers)
- You're monitoring risk-sensitive features (credit scores, fraud signals)
- You need more power than the KS test for tail differences
- You care about the full distribution shape, especially extremes

**Advantages over KS test**:
- More sensitive to tail differences
- Better at detecting changes in extreme values
- More powerful for detecting subtle distributional changes

```python
from driftwatch.detectors.numerical import AndersonDarlingDetector

detector = AndersonDarlingDetector(threshold=0.05)
result = detector.detect(reference_series, production_series)
print(f"AD stat: {result.score:.4f}, p-value: {result.p_value:.4f}")
```

### Cramér-von Mises Test *(New in v0.3.0)*
**Best for**: Detecting overall distributional changes

- **How it works**: Integrates squared differences between CDFs (vs. KS which uses max)
- **Threshold**: P-value (typically 0.05)
- **Interpretation**: P-value < threshold indicates drift

**Use when**:
- You want sensitivity to the entire distribution shape
- You need a complementary test to KS (more sensitive to overall shifts)
- Your distributions may differ across the full range, not just at one point
- You're running multi-test drift detection pipelines

**Advantages over KS test**:
- Considers the entire distribution, not just the maximum difference
- More sensitive to uniform distributional changes
- Better power for detecting diffuse differences

```python
from driftwatch.detectors.numerical import CramerVonMisesDetector

detector = CramerVonMisesDetector(threshold=0.05)
result = detector.detect(reference_series, production_series)
print(f"CvM stat: {result.score:.4f}, p-value: {result.p_value:.4f}")
```

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
│   ├─ High sensitivity needed → Wasserstein (threshold=0.1)
│   ├─ Bounded 0-1 score → Jensen-Shannon (threshold=0.1)
│   ├─ Tail-sensitive → Anderson-Darling (threshold=0.05)
│   └─ Overall shape → Cramér-von Mises (threshold=0.05)
└─ No (categorical) → Chi-Squared (threshold=0.05)
```

### Comparison Table

| Detector | Type | Sensitivity | Interpretability | Speed | Best For |
|----------|------|-------------|------------------|-------|----------|
| PSI | Numerical | Medium | High | Fast | General use |
| KS | Numerical | Medium | Medium | Fast | Hypothesis testing |
| Wasserstein | Numerical | High | Medium | Medium | Subtle shifts |
| Jensen-Shannon | Numerical | Medium-High | High | Fast | Bounded comparison |
| Anderson-Darling | Numerical | High (tails) | Medium | Fast | Tail differences |
| Cramér-von Mises | Numerical | High (overall) | Medium | Fast | Full distribution |
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
        "jensen_shannon": 0.1,
    }
)

# Using registry to get specific detector
from driftwatch.detectors.registry import get_detector_by_name

jsd = get_detector_by_name(
    "jensen_shannon",
    thresholds={"jensen_shannon": 0.1}
)

ad = get_detector_by_name(
    "anderson_darling",
    thresholds={"anderson_darling_pvalue": 0.05}
)

cvm = get_detector_by_name(
    "cramer_von_mises",
    thresholds={"cramer_von_mises_pvalue": 0.05}
)
```

## Advanced Tips

### When to Use Each New Detector

**Jensen-Shannon** is particularly useful for:
- **Distribution comparison dashboards**: Bounded 0-1 score is easy to visualize
- **Multi-model monitoring**: Compare drift across models with a normalized score
- **Research applications**: Well-known information-theoretic measure

**Anderson-Darling** is particularly useful for:
- **Financial modeling**: Tail risk is critical
- **Anomaly detection pipelines**: Outlier distribution changes
- **Quality control**: When extreme values matter most

**Cramér-von Mises** is particularly useful for:
- **Multi-test pipelines**: Complement KS test for more robust detection
- **General monitoring**: Captures overall shape changes better than KS
- **Model validation**: Comprehensive distributional comparison

### Threshold Tuning

Start with defaults and adjust based on:
1. **Sensitivity needs**: Lower threshold = more sensitive
2. **False positive tolerance**: Higher threshold = fewer alerts
3. **Historical data**: Analyze typical variation in your domain

### Combining Detectors

For critical features, monitor with multiple detectors:
```python
# Monitor age with multiple detectors for robust detection
from driftwatch.detectors.numerical import (
    PSIDetector,
    JensenShannonDetector,
    AndersonDarlingDetector,
)

psi = PSIDetector(threshold=0.2)
jsd = JensenShannonDetector(threshold=0.1)
ad = AndersonDarlingDetector(threshold=0.05)

psi_result = psi.detect(ref["age"], prod["age"])
jsd_result = jsd.detect(ref["age"], prod["age"])
ad_result = ad.detect(ref["age"], prod["age"])

# Ensemble: drift if any detector flags it
if any([psi_result.has_drift, jsd_result.has_drift, ad_result.has_drift]):
    alert("Drift detected in age feature")
```
