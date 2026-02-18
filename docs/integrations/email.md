# Email Integration

Get notified by email when drift is detected.

## Installation

```bash
pip install driftwatch[alerting]
```

## Setup

### 1. SMTP Configuration

You'll need access to an SMTP server. Common options:

| Provider | Host | Port | Notes |
|----------|------|------|-------|
| Gmail | `smtp.gmail.com` | 587 | Requires [App Password](https://support.google.com/accounts/answer/185833) |
| Outlook | `smtp.office365.com` | 587 | Use your Microsoft account |
| AWS SES | `email-smtp.{region}.amazonaws.com` | 587 | Use IAM SMTP credentials |
| SendGrid | `smtp.sendgrid.net` | 587 | Use API key as password |
| Local | `localhost` | 25 | For development/testing |

### 2. Configure Alerter

```python
from driftwatch.integrations.email import EmailAlerter

alerter = EmailAlerter(
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    username="your-email@gmail.com",
    password="your-app-password",
    sender="drift-alerts@yourcompany.com",
    recipients=["ml-team@yourcompany.com", "lead@yourcompany.com"],
    throttle_minutes=60,  # Max 1 alert per hour
)
```

## Usage

### Basic Alert

```python
from driftwatch import Monitor
from driftwatch.integrations.email import EmailAlerter

monitor = Monitor(reference_data=train_df)
report = monitor.check(prod_df)

if report.has_drift():
    alerter.send(report)
```

### Custom Subject

```python
alerter.send(
    report,
    custom_subject="🚨 URGENT: Production Model Drift Detected"
)
```

### Extra Recipients

Add recipients for a specific alert without changing the base configuration:

```python
alerter.send(
    report,
    extra_recipients=["oncall@yourcompany.com", "manager@yourcompany.com"]
)
```

### Force Send (Bypass Throttle)

```python
alerter.send(report, force=True)
```

## Email Format

The `EmailAlerter` sends **multipart emails** with both:

- **Plain text** fallback for simple email clients
- **Rich HTML** version with styled tables, status colors, and feature breakdowns

The HTML email includes:

- 🔍 **Header** with status color (green/yellow/red)
- 📊 **Summary cards** showing status, drift ratio, and affected features count
- 📋 **Feature details table** with per-feature method, score, threshold, and p-value
- 📎 **Footer** with sample size information

## Advanced Configuration

### Without Authentication

For local SMTP servers or relay configurations:

```python
alerter = EmailAlerter(
    smtp_host="localhost",
    smtp_port=25,
    sender="drift-monitor@internal.com",
    recipients=["team@internal.com"],
    use_tls=False,
)
```

### Throttle Management

```python
# Check when next alert can be sent
next_time = alerter.get_next_alert_time()
if next_time:
    print(f"Next alert available at: {next_time}")

# Reset throttle to allow immediate alert
alerter.reset_throttle()
```

### View Configuration

```python
config = alerter.get_config()
print(config)
# {
#     'smtp_host': 'smtp.gmail.com',
#     'smtp_port': 587,
#     'sender': 'alerts@example.com',
#     'recipients': ['team@example.com'],
#     'use_tls': True,
#     'throttle_seconds': 3600,
#     'subject_prefix': '[DriftWatch]'
# }
# Note: username and password are excluded for security
```

## Production Pipeline Example

```python
from driftwatch import Monitor
from driftwatch.integrations.email import EmailAlerter

# Configure once
monitor = Monitor(reference_data=training_df, features=["age", "income", "score"])
alerter = EmailAlerter(
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    username="alerts@company.com",
    password="app-password",
    sender="drift-monitor@company.com",
    recipients=["ml-team@company.com"],
    subject_prefix="[ML-Pipeline]",
    throttle_minutes=120,  # Max 1 alert every 2 hours
)

# Run in scheduled job (e.g., Airflow, cron)
def check_drift(production_batch):
    report = monitor.check(production_batch)

    if report.has_drift():
        alerter.send(report)

    return report
```

## See Also

- [Slack Integration →](slack.md)
- [CLI Integration →](cli.md)
- [FastAPI Integration →](fastapi.md)
- [MLflow Integration →](mlflow.md)
