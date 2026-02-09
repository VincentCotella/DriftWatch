"""
Visualization tools for drift explanation.

Provides histogram overlays and distribution comparisons
to visually understand how data has shifted.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from driftwatch.core.report import DriftReport


class DriftVisualizer:
    """
    Visualizes drift between reference and production distributions.

    Creates matplotlib figures showing distribution overlays,
    helping users understand exactly how data has shifted.

    Example:
        >>> from driftwatch import Monitor
        >>> from driftwatch.explain import DriftVisualizer
        >>>
        >>> monitor = Monitor(reference_data=train_df)
        >>> report = monitor.check(prod_df)
        >>>
        >>> viz = DriftVisualizer(train_df, prod_df, report)
        >>> fig = viz.plot_feature("age")
        >>> plt.show()
        >>>
        >>> # Or plot all features
        >>> fig = viz.plot_all()
        >>> plt.savefig("drift_report.png")
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        production_data: pd.DataFrame,
        report: DriftReport,
        style: str = "seaborn-v0_8-whitegrid",
    ) -> None:
        """
        Initialize the DriftVisualizer.

        Args:
            reference_data: Reference DataFrame (training data)
            production_data: Production DataFrame
            report: DriftReport from Monitor.check()
            style: Matplotlib style to use (default: seaborn-v0_8-whitegrid)
        """
        self.reference_data = reference_data
        self.production_data = production_data
        self.report = report
        self.style = style

        # Color scheme
        self.colors = {
            "reference": "#3498db",  # Blue
            "production": "#e74c3c",  # Red
            "ok": "#27ae60",  # Green
            "drift": "#e74c3c",  # Red
        }

    def plot_feature(
        self,
        feature_name: str,
        bins: int = 50,
        figsize: tuple[int, int] = (10, 6),
        show_stats: bool = True,
        alpha: float = 0.6,
    ) -> Any:
        """
        Plot histogram overlay for a single feature.

        Args:
            feature_name: Name of the feature to plot
            bins: Number of histogram bins
            figsize: Figure size (width, height)
            show_stats: Whether to show statistical annotations
            alpha: Transparency of histograms

        Returns:
            matplotlib Figure object

        Raises:
            ImportError: If matplotlib is not installed
            ValueError: If feature not found in data
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install it with: pip install driftwatch[viz]"
            ) from e

        import numpy as np

        if feature_name not in self.reference_data.columns:
            raise ValueError(f"Feature '{feature_name}' not found in reference data")

        if feature_name not in self.production_data.columns:
            raise ValueError(f"Feature '{feature_name}' not found in production data")

        ref_data = self.reference_data[feature_name].dropna()
        prod_data = self.production_data[feature_name].dropna()

        # Check if numeric
        if not np.issubdtype(ref_data.dtype, np.number):
            raise ValueError(
                f"Feature '{feature_name}' is not numeric. "
                "Visualization only supports numeric features."
            )

        # Get drift status
        feature_result = self.report.feature_drift(feature_name)
        has_drift = feature_result.has_drift if feature_result else False
        drift_score = feature_result.score if feature_result else 0.0
        drift_method = feature_result.method if feature_result else "unknown"

        # Create figure
        with contextlib.suppress(OSError):
            plt.style.use(self.style)

        fig, ax = plt.subplots(figsize=figsize)

        # Compute common bin edges
        all_data = np.concatenate([ref_data.values, prod_data.values])
        bin_edges = np.histogram_bin_edges(all_data, bins=bins)

        # Plot histograms
        ax.hist(
            ref_data,
            bins=bin_edges,
            alpha=alpha,
            label=f"Reference (n={len(ref_data):,})",
            color=self.colors["reference"],
            density=True,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.hist(
            prod_data,
            bins=bin_edges,
            alpha=alpha,
            label=f"Production (n={len(prod_data):,})",
            color=self.colors["production"],
            density=True,
            edgecolor="white",
            linewidth=0.5,
        )

        # Add vertical lines for means
        ref_mean = ref_data.mean()
        prod_mean = prod_data.mean()

        ax.axvline(
            ref_mean,
            color=self.colors["reference"],
            linestyle="--",
            linewidth=2,
            label=f"Ref mean: {ref_mean:.2f}",
        )
        ax.axvline(
            prod_mean,
            color=self.colors["production"],
            linestyle="--",
            linewidth=2,
            label=f"Prod mean: {prod_mean:.2f}",
        )

        # Title with drift status
        status_emoji = "🔴" if has_drift else "✅"
        status_text = "DRIFT DETECTED" if has_drift else "NO DRIFT"
        ax.set_title(
            f"{status_emoji} {feature_name}: {status_text}\n"
            f"Score ({drift_method}): {drift_score:.4f}",
            fontsize=14,
            fontweight="bold",
        )

        # Labels
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)

        # Legend
        ax.legend(loc="upper right", fontsize=10)

        # Add stats box if requested
        if show_stats:
            mean_shift = prod_mean - ref_mean
            mean_shift_pct = (mean_shift / abs(ref_mean) * 100) if ref_mean != 0 else 0
            ref_std = ref_data.std()
            prod_std = prod_data.std()

            stats_text = (
                f"Mean shift: {mean_shift:+.3f} ({mean_shift_pct:+.1f}%)\n"
                f"Ref std: {ref_std:.3f}\n"
                f"Prod std: {prod_std:.3f}"
            )

            # Position the text box
            props = {"boxstyle": "round,pad=0.5", "facecolor": "wheat", "alpha": 0.8}
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=props,
                family="monospace",
            )

        plt.tight_layout()
        return fig

    def plot_all(
        self,
        cols: int = 2,
        figsize: tuple[int, int] | None = None,
        bins: int = 50,
        alpha: float = 0.6,
    ) -> Any:
        """
        Plot histogram overlays for all numeric features.

        Args:
            cols: Number of columns in the grid
            figsize: Figure size (auto-calculated if None)
            bins: Number of histogram bins
            alpha: Transparency of histograms

        Returns:
            matplotlib Figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install it with: pip install driftwatch[viz]"
            ) from e

        import numpy as np

        # Get numeric features
        numeric_features = [
            r.feature_name
            for r in self.report.feature_results
            if r.feature_name in self.reference_data.columns
            and np.issubdtype(self.reference_data[r.feature_name].dtype, np.number)
        ]

        if not numeric_features:
            raise ValueError("No numeric features found to visualize")

        n_features = len(numeric_features)
        rows = (n_features + cols - 1) // cols

        if figsize is None:
            figsize = (6 * cols, 4 * rows)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, feature_name in enumerate(numeric_features):
            ax = axes[idx]
            self._plot_feature_on_ax(
                ax=ax,
                feature_name=feature_name,
                bins=bins,
                alpha=alpha,
            )

        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            "Drift Analysis - Distribution Comparison",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        return fig

    def _plot_feature_on_ax(
        self,
        ax: Any,
        feature_name: str,
        bins: int,
        alpha: float,
    ) -> None:
        """Plot a single feature on the given axes."""
        import numpy as np

        ref_data = self.reference_data[feature_name].dropna()
        prod_data = self.production_data[feature_name].dropna()

        feature_result = self.report.feature_drift(feature_name)
        has_drift = feature_result.has_drift if feature_result else False
        drift_score = feature_result.score if feature_result else 0.0

        # Compute common bin edges
        all_data = np.concatenate([ref_data.values, prod_data.values])
        bin_edges = np.histogram_bin_edges(all_data, bins=bins)

        # Plot histograms
        ax.hist(
            ref_data,
            bins=bin_edges,
            alpha=alpha,
            label="Reference",
            color=self.colors["reference"],
            density=True,
            edgecolor="white",
            linewidth=0.5,
        )

        ax.hist(
            prod_data,
            bins=bin_edges,
            alpha=alpha,
            label="Production",
            color=self.colors["production"],
            density=True,
            edgecolor="white",
            linewidth=0.5,
        )

        # Title
        status_emoji = "🔴" if has_drift else "✅"
        ax.set_title(
            f"{status_emoji} {feature_name} (score: {drift_score:.3f})",
            fontsize=11,
            fontweight="bold",
        )

        ax.set_xlabel(feature_name, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8)

    def save(
        self,
        filename: str,
        feature_name: str | None = None,
        dpi: int = 150,
        **kwargs: Any,
    ) -> str:
        """
        Save visualization to file.

        Args:
            filename: Output filename (supports png, pdf, svg)
            feature_name: Specific feature to plot (or all if None)
            dpi: Resolution for raster formats
            **kwargs: Additional arguments passed to savefig

        Returns:
            The filename that was saved
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install it with: pip install driftwatch[viz]"
            ) from e

        fig = self.plot_feature(feature_name) if feature_name else self.plot_all()

        fig.savefig(filename, dpi=dpi, bbox_inches="tight", **kwargs)
        plt.close(fig)

        return filename
