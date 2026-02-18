# Slack Integration

Get notified in Slack when drift is detected.

## Installation

```bash
pip install driftwatch[alerting]
```

## Setup

### 1. Create Slack Webhook

1. Go to [Slack API](https://api.slack.com/apps)
2. Create new app
3. Enable "Incoming Webhooks"
4. Add webhook to workspace
5. Copy webhook URL

### 2. Configure Alerter

```python
from driftwatch.integrations.alerting import SlackAlerter

alerter = SlackAlerter(
    webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    throttle_minutes=60  # Max 1 alert per hour
)
```

## Usage

```python
from driftwatch import Monitor

monitor = Monitor(reference_data=train_df)
report = monitor.check(prod_df)

if report.has_drift():
    alerter.send(report)
```

_<!-- Screenshot placeholder: Slack notification showing drift alert with feature breakdown -->_

## Advanced Configuration

### Custom Messages

```python
alerter.send(
    report,
    custom_message="🚨 Production Model Alert - Immediate Action Required"
)
```

### Mention Users

```python
alerter = SlackAlerter(
    webhook_url="...",
    mention_user="U123ABC"  # Slack user ID
)
```

### Channel Override

```python
alerter = SlackAlerter(
    webhook_url="...",
    channel_override="#critical-alerts"
)
```

## See Also

- [Email Integration →](email.md)
- [CLI Integration →](cli.md)
- [FastAPI Integration →](fastapi.md)
- [MLflow Integration →](mlflow.md)
