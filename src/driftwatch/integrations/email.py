"""Email alerting integration for DriftWatch.

Provides email notifications via SMTP for drift detection alerts.
Supports both synchronous and asynchronous sending.
"""

from __future__ import annotations

import smtplib
import ssl
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from driftwatch.core.report import DriftReport


class EmailAlerter:
    """
    Send drift alerts via email using SMTP.

    Formats drift reports as HTML emails with feature-level
    details and supports alert throttling to avoid spam.

    Args:
        smtp_host: SMTP server hostname (e.g., "smtp.gmail.com")
        smtp_port: SMTP server port (default: 587 for STARTTLS)
        username: SMTP authentication username
        password: SMTP authentication password
        sender: Sender email address
        recipients: List of recipient email addresses
        use_tls: Whether to use STARTTLS (default: True)
        throttle_minutes: Minimum minutes between alerts (default: 60)
        subject_prefix: Prefix for email subject (default: "[DriftWatch]")

    Example:
        ```python
        from driftwatch.integrations.email import EmailAlerter

        alerter = EmailAlerter(
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            username="alerts@example.com",
            password="app-password",
            sender="alerts@example.com",
            recipients=["team@example.com"],
            throttle_minutes=60,
        )

        if report.has_drift():
            alerter.send(report)
        ```
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        sender: str = "",
        recipients: list[str] | None = None,
        use_tls: bool = True,
        throttle_minutes: int = 60,
        subject_prefix: str = "[DriftWatch]",
    ) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.sender = sender
        self.recipients = recipients or []
        self.use_tls = use_tls
        self.throttle_seconds = throttle_minutes * 60
        self.subject_prefix = subject_prefix
        self._last_alert_time: float = 0.0

    def send(
        self,
        report: DriftReport,
        force: bool = False,
        custom_subject: str | None = None,
        extra_recipients: list[str] | None = None,
    ) -> bool:
        """
        Send drift report via email.

        Args:
            report: DriftReport to send
            force: Skip throttling check
            custom_subject: Optional custom email subject
            extra_recipients: Additional recipients for this alert

        Returns:
            True if email was sent, False if throttled

        Raises:
            smtplib.SMTPException: If email sending fails
            ValueError: If no recipients are configured
        """
        # Check throttling
        if not force and self._is_throttled():
            return False

        # Validate recipients
        all_recipients = list(self.recipients)
        if extra_recipients:
            all_recipients.extend(extra_recipients)

        if not all_recipients:
            raise ValueError(
                "No recipients configured. "
                "Provide recipients in constructor or via extra_recipients."
            )

        # Build email
        msg = self._build_message(report, all_recipients, custom_subject)

        # Send via SMTP
        self._send_smtp(msg, all_recipients)

        # Update throttle timestamp
        self._last_alert_time = time.time()

        return True

    def _is_throttled(self) -> bool:
        """Check if alert should be throttled."""
        if self._last_alert_time == 0.0:
            return False

        elapsed = time.time() - self._last_alert_time
        return elapsed < self.throttle_seconds

    def _build_message(
        self,
        report: DriftReport,
        recipients: list[str],
        custom_subject: str | None = None,
    ) -> MIMEMultipart:
        """Build email message with HTML content."""
        msg = MIMEMultipart("alternative")

        # Subject
        status_emoji = {"OK": "✅", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(
            report.status.value, "📊"
        )
        subject = custom_subject or (
            f"{self.subject_prefix} {status_emoji} Drift {report.status.value} "
            f"— {len(report.drifted_features())}/{len(report.feature_results)} "
            f"features affected"
        )
        msg["Subject"] = subject
        msg["From"] = self.sender
        msg["To"] = ", ".join(recipients)

        # Plain text version
        plain_text = self._build_plain_text(report)
        msg.attach(MIMEText(plain_text, "plain"))

        # HTML version
        html_content = self._build_html(report)
        msg.attach(MIMEText(html_content, "html"))

        return msg

    def _build_plain_text(self, report: DriftReport) -> str:
        """Build plain text email body."""
        lines = [
            "DriftWatch — Drift Detection Alert",
            "=" * 40,
            f"Status: {report.status.value}",
            f"Timestamp: {self._format_timestamp(report.timestamp)}",
            f"Drift Ratio: {report.drift_ratio():.1%}",
            f"Affected Features: {len(report.drifted_features())}"
            f"/{len(report.feature_results)}",
            "",
        ]

        if report.model_version:
            lines.append(f"Model Version: {report.model_version}")
            lines.append("")

        if report.drifted_features():
            lines.append("Drifted Features:")
            for result in report.feature_results:
                if result.has_drift:
                    lines.append(
                        f"  • {result.feature_name}: "
                        f"{result.method.upper()}={result.score:.4f} "
                        f"(threshold={result.threshold:.4f})"
                    )

        lines.append("")
        lines.append("— Sent by DriftWatch")

        return "\n".join(lines)

    def _build_html(self, report: DriftReport) -> str:
        """Build HTML email body with styled formatting."""
        status_colors: dict[str, str] = {
            "OK": "#27ae60",
            "WARNING": "#f39c12",
            "CRITICAL": "#e74c3c",
        }
        status_color = status_colors.get(report.status.value, "#95a5a6")

        # Build feature rows
        feature_rows = ""
        for result in report.feature_results:
            drift_indicator = (
                '<span style="color: #e74c3c; font-weight: bold;">⚠ DRIFT</span>'
                if result.has_drift
                else '<span style="color: #27ae60;">✓ OK</span>'
            )
            p_value_str = f"{result.p_value:.4f}" if result.p_value else "N/A"
            feature_rows += f"""
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding: 8px 12px; font-weight: 500;">{result.feature_name}</td>
                <td style="padding: 8px 12px; text-align: center;">{drift_indicator}</td>
                <td style="padding: 8px 12px; text-align: center; font-family: monospace;">{result.method.upper()}</td>
                <td style="padding: 8px 12px; text-align: center; font-family: monospace;">{result.score:.4f}</td>
                <td style="padding: 8px 12px; text-align: center; font-family: monospace;">{result.threshold:.4f}</td>
                <td style="padding: 8px 12px; text-align: center; font-family: monospace;">{p_value_str}</td>
            </tr>"""

        model_version_html = ""
        if report.model_version:
            model_version_html = f"""
            <div style="display: inline-block; background: #f0f0f0; padding: 4px 12px;
                        border-radius: 4px; font-size: 13px; margin-top: 8px;">
                Model: <strong>{report.model_version}</strong>
            </div>"""

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="margin: 0; padding: 20px; font-family: -apple-system,
                     BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                     background: #f5f5f5; color: #333;">
            <div style="max-width: 640px; margin: 0 auto; background: #fff;
                        border-radius: 8px; overflow: hidden;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);">

                <!-- Header -->
                <div style="background: {status_color}; padding: 20px 24px; color: #fff;">
                    <h1 style="margin: 0; font-size: 20px; font-weight: 600;">
                        🔍 DriftWatch Alert
                    </h1>
                    <p style="margin: 6px 0 0; opacity: 0.9; font-size: 14px;">
                        {self._format_timestamp(report.timestamp)}
                    </p>
                </div>

                <!-- Summary -->
                <div style="padding: 20px 24px;">
                    <div style="display: flex; gap: 24px; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 120px; text-align: center;
                                    padding: 16px; background: #f9f9f9;
                                    border-radius: 6px;">
                            <div style="font-size: 11px; text-transform: uppercase;
                                        color: #999; letter-spacing: 0.5px;">Status</div>
                            <div style="font-size: 20px; font-weight: 700;
                                        color: {status_color}; margin-top: 4px;">
                                {report.status.value}
                            </div>
                        </div>
                        <div style="flex: 1; min-width: 120px; text-align: center;
                                    padding: 16px; background: #f9f9f9;
                                    border-radius: 6px;">
                            <div style="font-size: 11px; text-transform: uppercase;
                                        color: #999; letter-spacing: 0.5px;">Drift Ratio</div>
                            <div style="font-size: 20px; font-weight: 700;
                                        color: #333; margin-top: 4px;">
                                {report.drift_ratio():.1%}
                            </div>
                        </div>
                        <div style="flex: 1; min-width: 120px; text-align: center;
                                    padding: 16px; background: #f9f9f9;
                                    border-radius: 6px;">
                            <div style="font-size: 11px; text-transform: uppercase;
                                        color: #999; letter-spacing: 0.5px;">Affected</div>
                            <div style="font-size: 20px; font-weight: 700;
                                        color: #333; margin-top: 4px;">
                                {len(report.drifted_features())}/{len(report.feature_results)}
                            </div>
                        </div>
                    </div>
                    {model_version_html}
                </div>

                <!-- Feature Details -->
                <div style="padding: 0 24px 20px;">
                    <h2 style="font-size: 15px; color: #666; margin: 0 0 12px;
                              text-transform: uppercase; letter-spacing: 0.5px;">
                        Feature Details
                    </h2>
                    <table style="width: 100%; border-collapse: collapse;
                                  font-size: 13px;">
                        <thead>
                            <tr style="background: #f9f9f9;">
                                <th style="padding: 8px 12px; text-align: left;
                                          font-weight: 600; color: #666;">Feature</th>
                                <th style="padding: 8px 12px; text-align: center;
                                          font-weight: 600; color: #666;">Status</th>
                                <th style="padding: 8px 12px; text-align: center;
                                          font-weight: 600; color: #666;">Method</th>
                                <th style="padding: 8px 12px; text-align: center;
                                          font-weight: 600; color: #666;">Score</th>
                                <th style="padding: 8px 12px; text-align: center;
                                          font-weight: 600; color: #666;">Threshold</th>
                                <th style="padding: 8px 12px; text-align: center;
                                          font-weight: 600; color: #666;">P-value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {feature_rows}
                        </tbody>
                    </table>
                </div>

                <!-- Footer -->
                <div style="padding: 16px 24px; background: #f9f9f9;
                            border-top: 1px solid #eee; text-align: center;
                            font-size: 12px; color: #999;">
                    Sent by <strong>DriftWatch</strong> •
                    Reference: {report.reference_size:,} samples •
                    Production: {report.production_size:,} samples
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def _send_smtp(self, msg: MIMEMultipart, recipients: list[str]) -> None:
        """Send email via SMTP."""
        if self.use_tls:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls(context=context)
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.sendmail(self.sender, recipients, msg.as_string())
        else:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.sendmail(self.sender, recipients, msg.as_string())

    def _format_timestamp(self, timestamp: datetime) -> str:
        """Format timestamp for email display."""
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

    def get_config(self) -> dict[str, Any]:
        """
        Get alerter configuration (without sensitive data).

        Returns:
            Dictionary with non-sensitive configuration values
        """
        return {
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "sender": self.sender,
            "recipients": self.recipients,
            "use_tls": self.use_tls,
            "throttle_seconds": self.throttle_seconds,
            "subject_prefix": self.subject_prefix,
        }
