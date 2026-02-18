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

Multi-Drift Monitoring:
    >>> from driftwatch import DriftSuite
    >>> suite = DriftSuite(reference_data=X_train, reference_predictions=y_val_pred)
    >>> report = suite.check(production_data=X_prod, production_predictions=y_prod_pred)
    >>> print(report.drift_types_detected())
"""

from driftwatch.core.concept_monitor import ConceptMonitor
from driftwatch.core.drift_suite import DriftSuite
from driftwatch.core.monitor import Monitor
from driftwatch.core.prediction_monitor import PredictionMonitor
from driftwatch.core.report import ComprehensiveDriftReport, DriftReport, DriftType
from driftwatch.explain import DriftExplainer, DriftVisualizer

__version__ = "0.4.0"
__all__ = [
    "ComprehensiveDriftReport",
    "ConceptMonitor",
    "DriftExplainer",
    "DriftReport",
    "DriftSuite",
    "DriftType",
    "DriftVisualizer",
    "Monitor",
    "PredictionMonitor",
    "__version__",
]
