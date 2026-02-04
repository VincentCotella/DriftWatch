"""Alerting integrations for DriftWatch.

Provides alerting mechanisms (Slack, Email, etc.) for drift detection.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from driftwatch.core.report import DriftReport


class SlackAlerter:
    """
    Send drift alerts to Slack via webhook.

    Formats drift reports as Slack Block Kit messages with feature-level
    details and supports alert throttling to avoid spam.

    Args:
        webhook_url: Slack webhook URL (https://hooks.slack.com/...)
        throttle_minutes: Minimum minutes between alerts (default: 60)
        mention_user: Optional Slack user ID to mention (@U123ABC)
        channel_override: Optional channel to post to (overrides webhook default)

    Example:
        ```python
        from driftwatch.integrations.alerting import SlackAlerter

        alerter = SlackAlerter(
            webhook_url="https://hooks.slack.com/services/...",
            throttle_minutes=60
        )

        if report.has_drift():
            alerter.send(report)
        ```
    """

    def __init__(
        self,
        webhook_url: str,
        throttle_minutes: int = 60,
        mention_user: str | None = None,
        channel_override: str | None = None,
    ) -> None:
        self.webhook_url = webhook_url
        self.throttle_seconds = throttle_minutes * 60
        self.mention_user = mention_user
        self.channel_override = channel_override
        self._last_alert_time: float = 0.0

    def send(
        self,
        report: DriftReport,
        force: bool = False,
        custom_message: str | None = None,
    ) -> bool:
        """
        Send drift report to Slack.

        Args:
            report: DriftReport to send
            force: Skip throttling check
            custom_message: Optional custom message prefix

        Returns:
            True if alert was sent, False if throttled

        Raises:
            httpx.HTTPError: If webhook request fails
        """
        # Check throttling
        if not force and self._is_throttled():
            return False

        # Build Slack message
        blocks = self._build_blocks(report, custom_message)
        payload: dict[str, Any] = {"blocks": blocks}

        if self.channel_override:
            payload["channel"] = self.channel_override

        # Send to Slack
        response = httpx.post(
            self.webhook_url, json=payload, timeout=10.0, follow_redirects=True
        )
        response.raise_for_status()

        # Update throttle timestamp
        self._last_alert_time = time.time()

        return True

    def _is_throttled(self) -> bool:
        """Check if alert should be throttled."""
        if self._last_alert_time == 0.0:
            return False

        elapsed = time.time() - self._last_alert_time
        return elapsed < self.throttle_seconds

    def _build_blocks(
        self, report: DriftReport, custom_message: str | None = None
    ) -> list[dict[str, Any]]:
        """Build Slack Block Kit message."""
        blocks: list[dict[str, Any]] = []

        # Status emoji and color
        emoji = {"OK": "✅", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(
            report.status.value, "📊"
        )

        # Header
        header_text = f"{emoji} *Drift Detected - DriftWatch*"
        if custom_message:
            header_text = f"{custom_message}\n{header_text}"
        if self.mention_user:
            header_text = f"<@{self.mention_user}> {header_text}"

        blocks.append(
            {"type": "header", "text": {"type": "plain_text", "text": header_text}}
        )

        # Summary section
        summary_fields = [
            {"type": "mrkdwn", "text": f"*Status:*\n{report.status.value}"},
            {
                "type": "mrkdwn",
                "text": f"*Drift Ratio:*\n{report.drift_ratio():.1%}",
            },
            {
                "type": "mrkdwn",
                "text": f"*Affected Features:*\n{len(report.drifted_features())}/{len(report.feature_results)}",
            },
            {
                "type": "mrkdwn",
                "text": f"*Timestamp:*\n{self._format_timestamp(report.timestamp)}",
            },
        ]

        blocks.append({"type": "section", "fields": summary_fields})

        # Divider
        blocks.append({"type": "divider"})

        # Feature details (only drifted features)
        if report.drifted_features():
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Drifted Features:*",
                    },
                }
            )

            feature_details = []
            for result in report.feature_results:
                if result.has_drift:
                    detail = f"• `{result.feature_name}`: {result.method.upper()}={result.score:.4f} (threshold={result.threshold:.4f})"
                    feature_details.append(detail)

            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "\n".join(feature_details),
                    },
                }
            )

        # Context footer
        context_text = "DriftWatch Monitor"
        if report.model_version:
            context_text += f" • Model: {report.model_version}"

        blocks.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": context_text}],
            }
        )

        return blocks

    def _format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp for Slack message."""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")

    def get_next_alert_time(self) -> datetime | None:
        """Get the earliest time the next alert can be sent."""
        if self._last_alert_time == 0.0:
            return None

        next_time = self._last_alert_time + self.throttle_seconds
        return datetime.fromtimestamp(next_time, tz=timezone.utc)

    def reset_throttle(self) -> None:
        """Reset throttle timer (allows immediate next alert)."""
        self._last_alert_time = 0.0
