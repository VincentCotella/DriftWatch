"""
DriftWatch - Lightweight ML drift monitoring, built for real-world pipelines.

DriftWatch is an open-source Python library for detecting data drift and
model drift in machine learning systems.

Basic Usage:
    >>> from driftwatch import Monitor
    >>> monitor = Monitor(reference_data=train_df, features=["age", "income"])
    >>> report = monitor.check(production_df)
    >>> print(report.summary())
"""

from driftwatch.core.monitor import Monitor
from driftwatch.core.report import DriftReport

__version__ = "0.2.0"
__all__ = [
    "DriftReport",
    "Monitor",
    "__version__",
]
