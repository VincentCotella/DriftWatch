# CLI Guide

Use DriftWatch from the command line for batch processing and CI/CD integration.

## Installation

```bash
pip install driftwatch[cli]
```

## Commands

### `driftwatch check`

Check for drift between reference and production datasets.

```bash
driftwatch check --ref REFERENCE --prod PRODUCTION [OPTIONS]
```

**Arguments:**

- `--ref`, `-r` - Path to reference dataset (CSV or Parquet)
- `--prod`, `-p` - Path to production dataset (CSV or Parquet)

**Options:**

- `--threshold-psi FLOAT` - PSI threshold (default: 0.2)
- `--threshold-ks FLOAT` - KS p-value threshold (default: 0.05)
- `--threshold-chi2 FLOAT` - Chi² p-value threshold (default: 0.05)
- `--output`, `-o` PATH - Save report to JSON file

**Exit Codes:**

- `0` - No drift detected (OK)
- `1` - Drift detected (WARNING)
- `2` - Critical drift (CRITICAL)

---

## Examples

### Basic Check

```bash
driftwatch check --ref train.csv --prod prod.csv
```

_<!-- Screenshot placeholder: CLI output with colored table showing drift results -->_

**Output:**

```
🔍 DriftWatch - Drift Detection

Loading reference data from train.csv...
✓ Loaded 10,000 samples with 5 features

Loading production data from prod.csv...
✓ Loaded 2,500 samples with 5 features

Initializing monitor...
Running drift detection...

Status: WARNING
Drift Detected: 2/5 features
Drift Ratio: 40.0%

Feature Analysis:

┏━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Feature      ┃ Method ┃  Score ┃ Threshold ┃ Status   ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
│ age          │ psi    │ 0.3521 │    0.2000 │ ⚠️ DRIFT │
│ income       │ psi    │ 0.1234 │    0.2000 │ ✓ OK     │
│ credit_score │ psi    │ 0.2891 │    0.2000 │ ⚠️ DRIFT │
└──────────────┴────────┴────────┴───────────┴──────────┘
```

### Custom Thresholds

```bash
driftwatch check \
  --ref train.parquet \
  --prod prod.parquet \
  --threshold-psi 0.15 \
  --threshold-ks 0.01
```

### Save Report

```bash
driftwatch check \
  --ref train.csv \
  --prod prod.csv \
  --output drift_report.json
```

---

### `driftwatch report`

Display a drift report from a JSON file.

```bash
driftwatch report REPORT_FILE [OPTIONS]
```

**Arguments:**

- `REPORT_FILE` - Path to drift report JSON

**Options:**

- `--format`, `-f` - Output format: `table` or `json` (default: `table`)
- `--output`, `-o` PATH - Save output to file

---

## CI/CD Integration

### GitHub Actions

```yaml
name: Drift Check

on:
  schedule:
    - cron: '0 0 * * *'  # Daily at midnight

jobs:
  drift-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install driftwatch[cli]
      
      - name: Run drift check
        run: |
          driftwatch check \
            --ref data/train.parquet \
            --prod data/production_latest.parquet \
            --output drift_report.json
      
      - name: Upload report
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: drift-report
          path: drift_report.json
      
      - name: Notify on drift
        if: failure()
        run: |
          # Send notification
          echo "Drift detected! Check artifacts."
```

### GitLab CI

```yaml
drift_check:
  stage: test
  image: python:3.11
  script:
    - pip install driftwatch[cli]
    - |
      driftwatch check \
        --ref data/train.parquet \
        --prod data/production.parquet \
        --threshold-psi 0.15
  artifacts:
    paths:
      - drift_report.json
    when: always
  only:
    - schedules
```

---

## Production Workflows

### Daily Drift Monitoring

```bash
#!/bin/bash
# daily_drift_check.sh

# Download latest production data
aws s3 cp s3://my-bucket/production_data.parquet .

# Run drift check
driftwatch check \
  --ref data/train.parquet \
  --prod production_data.parquet \
  --output drift_report_$(date +%Y%m%d).json

# Alert if drift detected
if [ $? -ne 0 ]; then
  # Send to Slack, PagerDuty, etc.
  curl -X POST https://hooks.slack.com/... \
    -d "{\"text\": \"Drift detected on $(date)\"}"
fi
```

### Pre-Deployment Check

```bash
#!/bin/bash
# Ensure no drift before deploying new model

driftwatch check \
  --ref data/train_v2.parquet \
  --prod data/validation.parquet \
  --threshold-psi 0.10

if [ $? -eq 0 ]; then
  echo "✓ No drift detected. Safe to deploy."
  # Deploy model
else
  echo "⚠️ Drift detected. Review before deploying."
  exit 1
fi
```

---

## Tips & Tricks

### 1. Use Parquet for Speed

Parquet is much faster than CSV for large datasets:

```bash
# Convert CSV to Parquet
python -c "
import pandas as pd
pd.read_csv('large_file.csv').to_parquet('large_file.parquet')
"

# Use in drift check
driftwatch check --ref train.parquet --prod prod.parquet
```

### 2. Pipe to jq for JSON

```bash
driftwatch check --ref train.csv --prod prod.csv --output - | jq .
```

### 3. Check Specific Features Only

Filter your data before checking:

```python
# filter_features.py
import pandas as pd
import sys

df = pd.read_parquet(sys.argv[1])
df[['age', 'income']].to_parquet(sys.argv[2])
```

```bash
python filter_features.py prod.parquet prod_filtered.parquet
driftwatch check --ref train_filtered.parquet --prod prod_filtered.parquet
```

---

## Next Steps

- [FastAPI Integration →](fastapi.md)
- [Slack Alerts →](slack.md)
- [Thresholds Guide →](../user-guide/thresholds.md)
