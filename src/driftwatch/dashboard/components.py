"""
Reusable Streamlit UI components for the DriftWatch dashboard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from driftwatch.core.report import DriftReport


def render_status_badge(status: str) -> None:
    """Render a colored status badge using Streamlit markdown.

    Args:
        status: One of "OK", "WARNING", or "CRITICAL".
    """
    import streamlit as st

    colors = {
        "OK": ("#1a9e5c", "🟢"),
        "WARNING": ("#d4a017", "🟡"),
        "CRITICAL": ("#c0392b", "🔴"),
    }
    color, emoji = colors.get(status, ("#888888", "⚪"))
    st.markdown(
        f"""
        <div style="
            display: inline-block;
            background-color: {color};
            color: white;
            font-size: 1.4rem;
            font-weight: 700;
            padding: 0.4rem 1.2rem;
            border-radius: 2rem;
            margin-bottom: 1rem;
        ">
            {emoji} {status}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(report: DriftReport) -> None:
    """Render the three top-level KPI metric cards.

    Args:
        report: The DriftReport to visualise.
    """
    import streamlit as st

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Status", report.status.value)
    with col2:
        st.metric(
            "Features Analysed",
            len(report.feature_results),
        )
    with col3:
        drifted = len(report.drifted_features())
        st.metric(
            "Features with Drift",
            drifted,
            delta=f"{drifted} detected" if drifted else None,
            delta_color="inverse",
        )
    with col4:
        st.metric("Drift Ratio", f"{report.drift_ratio():.1%}")


def render_feature_table(report: DriftReport) -> None:
    """Render the per-feature drift results as a styled dataframe.

    Args:
        report: The DriftReport to visualise.
    """
    import pandas as pd
    import streamlit as st

    rows = []
    for r in report.feature_results:
        rows.append(
            {
                "Feature": r.feature_name,
                "Method": r.method,
                "Score": round(r.score, 6),
                "Threshold": r.threshold,
                "P-value": round(r.p_value, 4) if r.p_value is not None else "—",
                "Status": "⚠️ DRIFT" if r.has_drift else "✅ OK",
            }
        )

    df = pd.DataFrame(rows)

    def _highlight(row: Any) -> list[str]:
        if row["Status"].startswith("⚠️"):
            return ["background-color: #ffeaea"] * len(row)
        return ["background-color: #eafff2"] * len(row)

    styled = df.style.apply(_highlight, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_distribution_chart(
    feature: str,
    ref_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    report: DriftReport,
) -> None:
    """Render a histogram overlay for a single feature using DriftVisualizer.

    Args:
        feature: Feature column name to plot.
        ref_df: Reference DataFrame.
        prod_df: Production DataFrame.
        report: DriftReport containing drift results.
    """
    import streamlit as st

    try:
        from driftwatch.explain.visualize import DriftVisualizer
    except ImportError:
        st.warning(
            "matplotlib is required for distribution charts. "
            "Install it with: `pip install driftwatch[viz]`"
        )
        return

    viz = DriftVisualizer(ref_df, prod_df, report)
    try:
        fig = viz.plot_feature(feature, figsize=(10, 5))
        st.pyplot(fig)
    except ValueError as exc:
        st.info(f"Distribution chart not available for this feature: {exc}")
