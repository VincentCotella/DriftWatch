"""Tests for Email alerting integration."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from driftwatch.core.report import DriftReport, FeatureDriftResult
from driftwatch.integrations.email import EmailAlerter


@pytest.fixture
def sample_report() -> DriftReport:
    """Create a sample drift report for testing."""
    return DriftReport(
        timestamp=datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc),
        reference_size=1000,
        production_size=500,
        feature_results=[
            FeatureDriftResult(
                feature_name="age",
                has_drift=True,
                score=0.35,
                method="psi",
                threshold=0.2,
            ),
            FeatureDriftResult(
                feature_name="income",
                has_drift=True,
                score=0.28,
                method="psi",
                threshold=0.2,
            ),
            FeatureDriftResult(
                feature_name="credit_score",
                has_drift=False,
                score=0.12,
                method="psi",
                threshold=0.2,
            ),
        ],
    )


def test_email_alerter_initialization() -> None:
    """Test EmailAlerter initialization."""
    alerter = EmailAlerter(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        username="test@example.com",
        password="secret",
        sender="alerts@example.com",
        recipients=["team@example.com", "lead@example.com"],
        throttle_minutes=30,
    )

    assert alerter.smtp_host == "smtp.gmail.com"
    assert alerter.smtp_port == 587
    assert alerter.username == "test@example.com"
    assert alerter.sender == "alerts@example.com"
    assert len(alerter.recipients) == 2
    assert alerter.throttle_seconds == 1800  # 30 * 60
    assert alerter.use_tls is True
    assert alerter._last_alert_time == 0.0


def test_email_alerter_default_values() -> None:
    """Test EmailAlerter default initialization values."""
    alerter = EmailAlerter(smtp_host="localhost")

    assert alerter.smtp_port == 587
    assert alerter.username == ""
    assert alerter.password == ""
    assert alerter.sender == ""
    assert alerter.recipients == []
    assert alerter.use_tls is True
    assert alerter.throttle_seconds == 3600  # 60 * 60
    assert alerter.subject_prefix == "[DriftWatch]"


@patch("driftwatch.integrations.email.smtplib.SMTP")
def test_email_alerter_send_success(
    mock_smtp_class: MagicMock, sample_report: DriftReport
) -> None:
    """Test successful email sending."""
    mock_server = MagicMock()
    mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
    mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

    alerter = EmailAlerter(
        smtp_host="smtp.gmail.com",
        username="test@example.com",
        password="secret",
        sender="alerts@example.com",
        recipients=["team@example.com"],
    )

    result = alerter.send(sample_report)

    assert result is True
    assert alerter._last_alert_time > 0


@patch("driftwatch.integrations.email.smtplib.SMTP")
def test_email_alerter_throttling(
    mock_smtp_class: MagicMock, sample_report: DriftReport
) -> None:
    """Test alert throttling."""
    mock_server = MagicMock()
    mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
    mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

    alerter = EmailAlerter(
        smtp_host="smtp.gmail.com",
        sender="alerts@example.com",
        recipients=["team@example.com"],
        throttle_minutes=1,
    )

    # First alert should succeed
    result1 = alerter.send(sample_report)
    assert result1 is True

    # Second alert immediately after should be throttled
    result2 = alerter.send(sample_report)
    assert result2 is False


@patch("driftwatch.integrations.email.smtplib.SMTP")
def test_email_alerter_force_send(
    mock_smtp_class: MagicMock, sample_report: DriftReport
) -> None:
    """Test forcing alert bypasses throttling."""
    mock_server = MagicMock()
    mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
    mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

    alerter = EmailAlerter(
        smtp_host="smtp.gmail.com",
        sender="alerts@example.com",
        recipients=["team@example.com"],
        throttle_minutes=1,
    )

    # First alert
    alerter.send(sample_report)

    # Force second alert (should bypass throttle)
    result = alerter.send(sample_report, force=True)
    assert result is True


def test_email_alerter_no_recipients_raises(sample_report: DriftReport) -> None:
    """Test that sending without recipients raises ValueError."""
    alerter = EmailAlerter(
        smtp_host="smtp.gmail.com",
        sender="alerts@example.com",
    )

    with pytest.raises(ValueError, match="No recipients configured"):
        alerter.send(sample_report)


@patch("driftwatch.integrations.email.smtplib.SMTP")
def test_email_alerter_extra_recipients(
    mock_smtp_class: MagicMock, sample_report: DriftReport
) -> None:
    """Test extra recipients are included."""
    mock_server = MagicMock()
    mock_smtp_class.return_value.__enter__ = MagicMock(return_value=mock_server)
    mock_smtp_class.return_value.__exit__ = MagicMock(return_value=False)

    alerter = EmailAlerter(
        smtp_host="smtp.gmail.com",
        sender="alerts@example.com",
        recipients=["team@example.com"],
    )

    result = alerter.send(
        sample_report,
        extra_recipients=["manager@example.com"],
    )
    assert result is True


def test_email_alerter_message_content(sample_report: DriftReport) -> None:
    """Test email message content is properly formatted."""
    alerter = EmailAlerter(
        smtp_host="smtp.gmail.com",
        sender="alerts@example.com",
        recipients=["team@example.com"],
        subject_prefix="[Test]",
    )

    msg = alerter._build_message(sample_report, ["team@example.com"])

    assert msg["From"] == "alerts@example.com"
    assert msg["To"] == "team@example.com"
    assert "[Test]" in msg["Subject"]
    assert "CRITICAL" in msg["Subject"]


def test_email_alerter_custom_subject(sample_report: DriftReport) -> None:
    """Test custom email subject."""
    alerter = EmailAlerter(
        smtp_host="smtp.gmail.com",
        sender="alerts@example.com",
        recipients=["team@example.com"],
    )

    msg = alerter._build_message(
        sample_report,
        ["team@example.com"],
        custom_subject="URGENT: Model Drift Alert",
    )

    assert msg["Subject"] == "URGENT: Model Drift Alert"


def test_email_alerter_plain_text(sample_report: DriftReport) -> None:
    """Test plain text email body."""
    alerter = EmailAlerter(
        smtp_host="smtp.gmail.com",
        sender="alerts@example.com",
    )

    plain = alerter._build_plain_text(sample_report)

    assert "DriftWatch" in plain
    assert "CRITICAL" in plain
    assert "age" in plain
    assert "income" in plain
    assert "PSI=0.3500" in plain
    assert "DriftWatch" in plain


def test_email_alerter_html_content(sample_report: DriftReport) -> None:
    """Test HTML email body contains expected elements."""
    alerter = EmailAlerter(
        smtp_host="smtp.gmail.com",
        sender="alerts@example.com",
    )

    html = alerter._build_html(sample_report)

    assert "<html>" in html
    assert "DriftWatch Alert" in html
    assert "age" in html
    assert "income" in html
    assert "credit_score" in html
    assert "DRIFT" in html
    assert "OK" in html


def test_email_alerter_get_next_alert_time() -> None:
    """Test getting next alert time."""
    alerter = EmailAlerter(
        smtp_host="smtp.gmail.com",
        throttle_minutes=5,
    )

    # No alerts yet
    assert alerter.get_next_alert_time() is None

    # Simulate alert
    alerter._last_alert_time = time.time()

    next_time = alerter.get_next_alert_time()
    assert next_time is not None
    assert next_time > datetime.now(timezone.utc)


def test_email_alerter_reset_throttle() -> None:
    """Test throttle reset."""
    alerter = EmailAlerter(smtp_host="smtp.gmail.com")

    alerter._last_alert_time = time.time()
    assert alerter._is_throttled() is True

    alerter.reset_throttle()
    assert alerter._is_throttled() is False
    assert alerter._last_alert_time == 0.0


def test_email_alerter_get_config() -> None:
    """Test get_config returns non-sensitive data."""
    alerter = EmailAlerter(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        username="secret_user",
        password="secret_pass",
        sender="alerts@example.com",
        recipients=["team@example.com"],
    )

    config = alerter.get_config()

    assert config["smtp_host"] == "smtp.gmail.com"
    assert config["smtp_port"] == 587
    assert config["sender"] == "alerts@example.com"
    assert "username" not in config
    assert "password" not in config
