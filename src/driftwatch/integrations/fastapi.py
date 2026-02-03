"""FastAPI integration for DriftWatch.

Provides middleware and endpoints for automatic drift monitoring
on ML inference APIs.
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, cast

import pandas as pd
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.types import ASGIApp

    from driftwatch import Monitor
    from driftwatch.core.report import DriftReport


@dataclass
class DriftState:
    """Thread-safe state for drift monitoring."""

    samples: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=10000))
    predictions: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=10000)
    )
    last_report: DriftReport | None = None
    last_check_time: datetime | None = None
    request_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_prediction(self, prediction: dict[str, Any]) -> None:
        """Add a prediction to the buffer."""
        with self.lock:
            self.predictions.append(prediction)

    def add_sample(self, sample: dict[str, Any]) -> None:
        """Add a sample to the buffer."""
        with self.lock:
            self.samples.append(sample)
            self.request_count += 1

    def get_samples_df(self) -> pd.DataFrame:
        """Get samples as DataFrame."""
        with self.lock:
            return pd.DataFrame(list(self.samples))

    def update_report(self, report: DriftReport) -> None:
        """Update the last drift report."""
        with self.lock:
            self.last_report = report
            self.last_check_time = datetime.now(timezone.utc)


class DriftMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic drift monitoring.

    Collects input features from requests and runs drift detection
    on a configurable schedule.

    Args:
        app: The ASGI application
        monitor: DriftWatch Monitor instance with reference data
        feature_extractor: Function to extract features from request body.
            Defaults to returning the entire request body as features.
        check_interval: Number of requests between drift checks.
            Set to 0 to disable automatic checks.
        min_samples: Minimum samples required before running drift check.
        enabled: Whether drift collection is enabled.

    Example:
        ```python
        from fastapi import FastAPI
        from driftwatch import Monitor
        from driftwatch.integrations.fastapi import DriftMiddleware

        monitor = Monitor(reference_data=train_df)
        app = FastAPI()

        app.add_middleware(
            DriftMiddleware,
            monitor=monitor,
            check_interval=100,
        )
        ```
    """

    def __init__(
        self,
        app: ASGIApp,
        monitor: Monitor,
        feature_extractor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        prediction_extractor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        check_interval: int = 100,
        min_samples: int = 50,
        buffer_size: int = 10000,
        enabled: bool = True,
    ) -> None:
        super().__init__(app)
        self.monitor = monitor
        self.feature_extractor = feature_extractor or (lambda x: x)
        self.prediction_extractor = prediction_extractor
        self.check_interval = check_interval
        self.min_samples = min_samples
        self.buffer_size = buffer_size
        self.enabled = enabled
        self.state = DriftState(
            samples=deque(maxlen=buffer_size),
            predictions=deque(maxlen=buffer_size),
        )
        self._background_tasks: set[asyncio.Task[None]] = set()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect features for drift monitoring."""
        if not self.enabled:
            return cast("Response", await call_next(request))

        # Skip non-POST requests and internal endpoints
        if request.method != "POST" or request.url.path.startswith("/drift"):
            return cast("Response", await call_next(request))

        # Try to extract features from request body
        try:
            body = await request.json()
            features = self.feature_extractor(body)

            if features and isinstance(features, dict):
                # Filter to only monitored features
                monitored = {
                    k: v
                    for k, v in features.items()
                    if k in self.monitor.monitored_features
                }
                if monitored:
                    self.state.add_sample(monitored)

        except Exception:
            # Don't fail the request if feature extraction fails
            pass

        # Process the request
        response = cast("Response", await call_next(request))

        # Try to extract predictions from response
        if self.prediction_extractor is not None:
            try:
                # For JSONResponse, we can access the body
                if hasattr(response, "body"):
                    import json

                    response_body = json.loads(response.body)
                    prediction = self.prediction_extractor(response_body)
                    if prediction and isinstance(prediction, dict):
                        self.state.add_prediction(prediction)
            except Exception:
                pass

        # Check if we should run drift detection
        if self._should_check_drift():
            task = asyncio.create_task(self._run_drift_check())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        return response

    def _should_check_drift(self) -> bool:
        """Determine if drift check should run."""
        if self.check_interval <= 0:
            return False
        if len(self.state.samples) < self.min_samples:
            return False
        return self.state.request_count % self.check_interval == 0

    async def _run_drift_check(self) -> None:
        """Run drift detection in background."""
        try:
            production_df = self.state.get_samples_df()
            if production_df.empty:
                return

            # Run check in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            report = await loop.run_in_executor(None, self.monitor.check, production_df)
            self.state.update_report(report)

        except Exception:
            # Log error in production, but don't crash
            pass


def add_drift_routes(app: Any, middleware: DriftMiddleware) -> None:
    """
    Add drift monitoring endpoints to a FastAPI app.

    Endpoints:
        GET /drift/status - Current drift status
        GET /drift/report - Full drift report
        GET /drift/health - Health check

    Args:
        app: FastAPI application instance
        middleware: DriftMiddleware instance
    """
    from fastapi import FastAPI

    if not isinstance(app, FastAPI):
        raise TypeError("app must be a FastAPI instance")

    @app.get("/drift/status")
    async def drift_status() -> dict[str, Any]:
        """Get current drift status."""
        state = middleware.state

        if state.last_report is None:
            return {
                "status": "NO_DATA",
                "message": "No drift check has been performed yet",
                "samples_collected": len(state.samples),
                "min_samples_required": middleware.min_samples,
            }

        return {
            "status": state.last_report.status.value,
            "has_drift": state.last_report.has_drift(),
            "drift_ratio": state.last_report.drift_ratio(),
            "drifted_features": state.last_report.drifted_features(),
            "last_check": (
                state.last_check_time.isoformat() if state.last_check_time else None
            ),
            "samples_collected": len(state.samples),
            "total_requests": state.request_count,
        }

    @app.get("/drift/report")
    async def drift_report() -> dict[str, Any]:
        """Get full drift report."""
        state = middleware.state

        if state.last_report is None:
            return {
                "error": "No drift report available",
                "samples_collected": len(state.samples),
            }

        return state.last_report.to_dict()

    @app.get("/drift/health")
    async def drift_health() -> dict[str, Any]:
        """Health check endpoint."""
        state = middleware.state

        return {
            "status": "healthy",
            "monitoring_enabled": middleware.enabled,
            "features_monitored": middleware.monitor.monitored_features,
            "samples_in_buffer": len(state.samples),
            "check_interval": middleware.check_interval,
        }

    @app.post("/drift/check")
    async def trigger_drift_check() -> dict[str, Any]:
        """Manually trigger a drift check."""
        production_df = middleware.state.get_samples_df()

        if len(production_df) < middleware.min_samples:
            return {
                "error": f"Not enough samples. Need {middleware.min_samples}, have {len(production_df)}",
            }

        report = middleware.monitor.check(production_df)
        middleware.state.update_report(report)

        return {
            "status": report.status.value,
            "has_drift": report.has_drift(),
            "drifted_features": report.drifted_features(),
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    @app.post("/drift/reset")
    async def reset_samples() -> dict[str, Any]:
        """Reset collected samples."""
        with middleware.state.lock:
            middleware.state.samples.clear()
            middleware.state.request_count = 0

        return {"message": "Samples reset successfully"}
