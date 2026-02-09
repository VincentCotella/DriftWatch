"""
DriftWatch - Lightweight ML drift monitoring, built for real-world pipelines.

DriftWatch is an open-source Python library for detecting data drift and
model drift in machine learning systems.

Basic Usage:
    >>> from driftwatch import Monitor
    >>> monitor = Monitor(reference_data=train_df, features=["age", "income"])
    >>> report = monitor.check(production_df)
    >>> print(report.summary())

Drift Explanation (v0.3.0+):
    >>> from driftwatch.explain import DriftExplainer, DriftVisualizer
    >>> explainer = DriftExplainer(train_df, prod_df, report)
    >>> print(explainer.explain().summary())
    >>> viz = DriftVisualizer(train_df, prod_df, report)
    >>> viz.plot_all()
"""

from driftwatch.core.monitor import Monitor
from driftwatch.core.report import DriftReport
from driftwatch.explain import DriftExplainer, DriftVisualizer

__version__ = "0.3.0"
__all__ = [
    "DriftExplainer",
    "DriftReport",
    "DriftVisualizer",
    "Monitor",
    "__version__",
]
