"""
DriftWatch Dashboard — Main Streamlit application.

Interactive UI for detecting and visualizing feature drift between
a reference dataset and a production dataset.

Usage:
    # Via CLI (recommended)
    driftwatch dashboard

    # Or directly
    streamlit run src/driftwatch/dashboard/app.py
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
import streamlit as st

if TYPE_CHECKING:
    import pandas as pd

# ── Page configuration ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="DriftWatch Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Global ── */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    /* ── Metric cards ── */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        backdrop-filter: blur(10px);
    }
    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.2rem;
        background: rgba(255,255,255,0.05);
        color: #a0a0a0;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(100, 149, 237, 0.25) !important;
        color: #6495ed !important;
        font-weight: 600;
    }
    /* ── Section headers ── */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #c8d6f0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 0.4rem;
        margin-bottom: 0.8rem;
    }
    /* ── Upload area ── */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 0.5rem;
    }
    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Helper functions ──────────────────────────────────────────────────────────


def _load_df(uploaded_file: object) -> pd.DataFrame:
    """Load a DataFrame from an uploaded CSV or Parquet file.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Parsed pandas DataFrame.

    Raises:
        ValueError: If the file format is not supported.
    """
    import pandas as pd

    name: str = getattr(uploaded_file, "name", "")
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".parquet", ".pq")):
        return pd.read_parquet(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: {name}. Use .csv or .parquet")


def _generate_sample_csv(drifted: bool = False) -> bytes:
    """Generate a sample CSV for quick demo.

    Args:
        drifted: If True, shift the distributions to simulate drift.

    Returns:
        CSV content as bytes.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42 if not drifted else 99)
    n = 500
    shift = 2.5 if drifted else 0.0
    df = pd.DataFrame(
        {
            "age": rng.normal(35 + shift, 10, n).clip(18, 80).astype(int),
            "income": rng.normal(50000 + shift * 5000, 15000, n).clip(10000),
            "score": rng.uniform(0.3 + shift * 0.1, 0.9, n),
            "category": rng.choice(
                ["A", "B", "C"] if not drifted else ["B", "C", "D"], n
            ),
        }
    )
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


class _NamedBytesIO(io.BytesIO):
    """BytesIO with a `name` attribute so `_load_df` can detect the format."""

    def __init__(self, data: bytes, name: str) -> None:
        super().__init__(data)
        self.name = name


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "## 🔍 DriftWatch",
        help="Upload reference and production datasets to check for feature drift.",
    )
    st.caption("v0.4.0 · Lightweight ML Drift Monitoring")
    st.divider()

    # ── Quick demo ──
    st.markdown('<p class="section-title">⚡ Quick Demo</p>', unsafe_allow_html=True)
    if st.button("Load sample data", use_container_width=True):
        st.session_state["demo_ref"] = _generate_sample_csv(drifted=False)
        st.session_state["demo_prod"] = _generate_sample_csv(drifted=True)
        st.success("Sample data loaded! Scroll down and click **Run Drift Check**.")

    st.divider()

    # ── Data upload ──
    st.markdown('<p class="section-title">📂 Data Upload</p>', unsafe_allow_html=True)

    ref_file = st.file_uploader(
        "Reference dataset (train)",
        type=["csv", "parquet"],
        key="ref_upload",
        help="Your baseline / training data",
    )
    prod_file = st.file_uploader(
        "Production dataset",
        type=["csv", "parquet"],
        key="prod_upload",
        help="Current production data to compare",
    )

    # Allow demo data to be used if no file uploaded
    ref_bytes = st.session_state.get("demo_ref")
    prod_bytes = st.session_state.get("demo_prod")

    using_demo = False
    if ref_file is None and ref_bytes:
        ref_file = _NamedBytesIO(ref_bytes, "reference_sample.csv")
        using_demo = True
    if prod_file is None and prod_bytes:
        prod_file = _NamedBytesIO(prod_bytes, "production_sample.csv")
        using_demo = True

    st.divider()

    # ── Threshold configuration ──
    st.markdown('<p class="section-title">⚙️ Thresholds</p>', unsafe_allow_html=True)
    threshold_psi = st.slider("PSI threshold", 0.05, 0.5, 0.2, 0.01)
    threshold_ks = st.slider("KS p-value threshold", 0.01, 0.2, 0.05, 0.005)
    threshold_chi2 = st.slider("Chi² p-value threshold", 0.01, 0.2, 0.05, 0.005)
    threshold_wasserstein = st.slider("Wasserstein threshold", 0.01, 1.0, 0.1, 0.01)

    st.divider()

    # ── Feature selection (filled later) ──
    st.markdown(
        '<p class="section-title">🎯 Feature Selection</p>', unsafe_allow_html=True
    )
    feature_filter = st.text_input(
        "Filter features (comma-separated, empty = all)",
        placeholder="age, income, score",
        help="Leave empty to monitor all columns",
    )

# ── Main content ─────────────────────────────────────────────────────────────

st.markdown(
    "# 🔍 DriftWatch Dashboard",
)
st.caption(
    "Compare your reference dataset against production to detect distribution shifts."
)

# ── Data preview tab ────────────────────────────────────────────────────

if ref_file is None or prod_file is None:
    st.info(
        "👈 Upload a **reference** and a **production** dataset in the sidebar, "
        "or click **Load sample data** for a quick demo."
    )
    st.stop()

# Load dataframes
try:
    ref_df = _load_df(ref_file)
    prod_df = _load_df(prod_file)
except ValueError as exc:
    st.error(f"❌ Error loading data: {exc}")
    st.stop()

if using_demo:
    st.warning(
        "You are using the **demo datasets**. "
        "Upload your own CSV/Parquet files via the sidebar to analyse real data."
    )

# ── Parse feature filter ──
features_input = [f.strip() for f in feature_filter.split(",") if f.strip()]
selected_features = features_input if features_input else None

# ── Preview tabs ──────────────────────────────────────────────────────────
tab_preview, tab_results, tab_charts, tab_json = st.tabs(
    ["📋 Data Preview", "📊 Drift Results", "📈 Distribution Charts", "📄 Raw JSON"]
)

with tab_preview:
    col_ref, col_prod = st.columns(2)
    with col_ref:
        st.subheader("Reference dataset")
        st.caption(f"{len(ref_df):,} rows · {len(ref_df.columns)} columns")
        st.dataframe(ref_df.head(20), use_container_width=True)
    with col_prod:
        st.subheader("Production dataset")
        st.caption(f"{len(prod_df):,} rows · {len(prod_df.columns)} columns")
        st.dataframe(prod_df.head(20), use_container_width=True)

# ── Run drift check ─────────────────────────────────────────────────────────

st.divider()
run_clicked = st.button(
    "🚀 Run Drift Check",
    type="primary",
    use_container_width=True,
    key="run_btn",
)

if not run_clicked and "last_report" not in st.session_state:
    st.info("Configure your thresholds and click **Run Drift Check** to start.")
    st.stop()

if run_clicked:
    with st.spinner("Running drift detection…"):
        try:
            from driftwatch import Monitor

            monitor = Monitor(
                reference_data=ref_df,
                features=selected_features,
                thresholds={
                    "psi": threshold_psi,
                    "ks_pvalue": threshold_ks,
                    "chi2_pvalue": threshold_chi2,
                    "wasserstein": threshold_wasserstein,
                },
            )
            report = monitor.check(prod_df)
            st.session_state["last_report"] = report
            st.session_state["last_ref_df"] = ref_df
            st.session_state["last_prod_df"] = prod_df
        except Exception as exc:
            st.error(f"❌ Drift check failed: {exc}")
            st.stop()

report = st.session_state["last_report"]
cached_ref = st.session_state["last_ref_df"]
cached_prod = st.session_state["last_prod_df"]

# ── Results tab ──────────────────────────────────────────────────────────────

with tab_results:
    from driftwatch.dashboard.components import (
        render_feature_table,
        render_kpi_cards,
        render_status_badge,
    )

    st.subheader("Overall Status")
    render_status_badge(report.status.value)

    st.subheader("Key Metrics")
    render_kpi_cards(report)

    st.divider()

    st.subheader("Feature-level Results")
    render_feature_table(report)

    # Download report
    st.download_button(
        label="⬇️ Download report (JSON)",
        data=report.to_json(),
        file_name="drift_report.json",
        mime="application/json",
        use_container_width=True,
    )

# ── Distribution charts tab ───────────────────────────────────────────────────

with tab_charts:
    from driftwatch.dashboard.components import render_distribution_chart

    # Only numeric features can be plotted
    numeric_features = [
        r.feature_name
        for r in report.feature_results
        if r.feature_name in cached_ref.columns
        and np.issubdtype(cached_ref[r.feature_name].dtype, np.number)
    ]

    if not numeric_features:
        st.info("No numeric features found to visualise.")
    else:
        selected = st.selectbox(
            "Select a feature to visualise",
            options=numeric_features,
            index=0,
        )
        if selected:
            render_distribution_chart(selected, cached_ref, cached_prod, report)

        st.divider()
        if st.checkbox("Show all features (one chart per feature)"):
            for feat in numeric_features:
                with st.expander(feat, expanded=False):
                    render_distribution_chart(feat, cached_ref, cached_prod, report)

# ── Raw JSON tab ─────────────────────────────────────────────────────────────

with tab_json:
    import json

    st.subheader("Raw report JSON")
    st.json(json.loads(report.to_json()))
