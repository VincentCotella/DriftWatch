"""DriftWatch integrations for external services."""

from driftwatch.integrations.fastapi import DriftMiddleware, add_drift_routes

__all__ = ["DriftMiddleware", "MLflowDriftTracker", "add_drift_routes"]


def __getattr__(name: str) -> object:
    """Lazy-load optional integrations to avoid hard import errors."""
    if name == "MLflowDriftTracker":
        from driftwatch.integrations.mlflow import MLflowDriftTracker

        return MLflowDriftTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
