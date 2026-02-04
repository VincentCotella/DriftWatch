"""Tests for Slack alerting integration."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from driftwatch.core.report import DriftReport, FeatureDriftResult
from driftwatch.integrations.alerting import SlackAlerter


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


def test_slack_alerter_initialization() -> None:
    """Test SlackAlerter initialization."""
    alerter = SlackAlerter(
        webhook_url="https://hooks.slack.com/test",
        throttle_minutes=30,
        mention_user="U123ABC",
    )

    assert alerter.webhook_url == "https://hooks.slack.com/test"
    assert alerter.throttle_seconds == 1800  # 30 * 60
    assert alerter.mention_user == "U123ABC"
    assert alerter._last_alert_time == 0.0


@patch("httpx.post")
def test_slack_alerter_send_success(
    mock_post: MagicMock, sample_report: DriftReport
) -> None:
    """Test successful alert sending."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    alerter = SlackAlerter(webhook_url="https://hooks.slack.com/test")

    result = alerter.send(sample_report)

    assert result is True
    assert mock_post.called
    assert alerter._last_alert_time > 0


@patch("httpx.post")
def test_slack_alerter_throttling(
    mock_post: MagicMock, sample_report: DriftReport
) -> None:
    """Test alert throttling."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    alerter = SlackAlerter(
        webhook_url="https://hooks.slack.com/test", throttle_minutes=1
    )

    # First alert should succeed
    result1 = alerter.send(sample_report)
    assert result1 is True

    # Second alert immediately after should be throttled
    result2 = alerter.send(sample_report)
    assert result2 is False

    # Only one call to httpx.post
    assert mock_post.call_count == 1


@patch("httpx.post")
def test_slack_alerter_force_send(
    mock_post: MagicMock, sample_report: DriftReport
) -> None:
    """Test forcing alert bypasses throttling."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    alerter = SlackAlerter(
        webhook_url="https://hooks.slack.com/test", throttle_minutes=1
    )

    # First alert
    alerter.send(sample_report)

    # Force second alert (should bypass throttle)
    result = alerter.send(sample_report, force=True)

    assert result is True
    assert mock_post.call_count == 2


@patch("httpx.post")
def test_slack_alerter_message_format(
    mock_post: MagicMock, sample_report: DriftReport
) -> None:
    """Test Slack message formatting."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    alerter = SlackAlerter(
        webhook_url="https://hooks.slack.com/test", mention_user="U123ABC"
    )

    alerter.send(sample_report)

    # Check the payload structure
    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]

    assert "blocks" in payload
    blocks = payload["blocks"]

    # Should have header, summary, divider, features, context
    assert len(blocks) >= 4

    # Check header contains mention
    header_block = blocks[0]
    assert header_block["type"] == "header"
    assert "<@U123ABC>" in header_block["text"]["text"]


@patch("httpx.post")
def test_slack_alerter_custom_message(
    mock_post: MagicMock, sample_report: DriftReport
) -> None:
    """Test custom message prefix."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    alerter = SlackAlerter(webhook_url="https://hooks.slack.com/test")

    alerter.send(sample_report, custom_message="Production Model Alert")

    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]
    header = payload["blocks"][0]["text"]["text"]

    assert "Production Model Alert" in header


def test_slack_alerter_get_next_alert_time() -> None:
    """Test getting next alert time."""
    alerter = SlackAlerter(
        webhook_url="https://hooks.slack.com/test", throttle_minutes=5
    )

    # No alerts yet
    assert alerter.get_next_alert_time() is None

    # Simulate alert
    alerter._last_alert_time = time.time()

    next_time = alerter.get_next_alert_time()
    assert next_time is not None
    assert next_time > datetime.now(timezone.utc)


def test_slack_alerter_reset_throttle() -> None:
    """Test throttle reset."""
    alerter = SlackAlerter(webhook_url="https://hooks.slack.com/test")

    alerter._last_alert_time = time.time()
    assert alerter._is_throttled() is True

    alerter.reset_throttle()
    assert alerter._is_throttled() is False
    assert alerter._last_alert_time == 0.0


@patch("httpx.post")
def test_slack_alerter_channel_override(
    mock_post: MagicMock, sample_report: DriftReport
) -> None:
    """Test channel override."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    alerter = SlackAlerter(
        webhook_url="https://hooks.slack.com/test", channel_override="#alerts"
    )

    alerter.send(sample_report)

    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]

    assert payload["channel"] == "#alerts"
