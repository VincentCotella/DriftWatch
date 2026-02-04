# Reports

Learn how to work with drift reports.

## Creating Reports

Reports are created automatically by the Monitor:

```python
report = monitor.check(prod_df)
```

## Report Properties

- `status` - Overall drift status (OK, WARNING, CRITICAL)
- `has_drift()` - Boolean drift indicator  
- `drift_ratio()` - Percentage of drifted features
- `drifted_features()` - List of feature names with drift
- `feature_results` - Per-feature detailed results

## Usage Examples

See [API Reference](../api/reports.md) for full documentation.
