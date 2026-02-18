"""DriftWatch integrations for external services."""

from driftwatch.integrations.fastapi import DriftMiddleware, add_drift_routes

__all__ = ["DriftMiddleware", "EmailAlerter", "MLflowDriftTracker", "add_drift_routes"]


def __getattr__(name: str) -> object:
    """Lazy-load optional integrations to avoid hard import errors."""
    if name == "MLflowDriftTracker":
        from driftwatch.integrations.mlflow import MLflowDriftTracker

        return MLflowDriftTracker
    if name == "EmailAlerter":
        from driftwatch.integrations.email import EmailAlerter

        return EmailAlerter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
