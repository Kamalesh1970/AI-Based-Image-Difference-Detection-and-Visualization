"""
streamlit_app.py
================
Feature-rich, styled Streamlit frontend for the AI-Powered CAD Revision
Analyzer. Upload a BEFORE and AFTER engineering drawing PDF, run the full
10-stage hybrid pipeline, and interactively explore the detected changes,
annotated comparison image, statistics dashboard, and downloadable
JSON/Markdown/PNG reports.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.pipeline import CADRevisionPipeline  # noqa: E402
from config.config import settings  # noqa: E402
from modules.classification.classify import ChangeCategory  # noqa: E402

# --------------------------------------------------------------------------- #
# Page configuration & global styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="CAD Revision Analyzer",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --brand-navy: #0f2540;
    --brand-blue: #1c5d99;
    --brand-cyan: #2fb5c4;
    --brand-amber: #f2a71b;
    --brand-green: #23a55a;
    --brand-red: #e2493d;
    --surface: #f7f9fb;
}

html, body, [class*="css"] { font-family: 'Segoe UI', 'Inter', sans-serif; }

.main .block-container { padding-top: 1.6rem; max-width: 1400px; }

.hero {
    background: linear-gradient(120deg, var(--brand-navy) 0%, var(--brand-blue) 55%, var(--brand-cyan) 100%);
    border-radius: 18px;
    padding: 2.1rem 2.4rem;
    color: #ffffff;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 30px rgba(15, 37, 64, 0.25);
}
.hero h1 { font-size: 2.05rem; margin-bottom: 0.35rem; font-weight: 750; letter-spacing: -0.01em; }
.hero p { font-size: 1.02rem; opacity: 0.92; margin: 0; max-width: 760px; }

.metric-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    border: 1px solid #e7ebef;
    box-shadow: 0 2px 10px rgba(15, 37, 64, 0.05);
    text-align: center;
}
.metric-card .metric-value { font-size: 1.9rem; font-weight: 750; color: var(--brand-navy); }
.metric-card .metric-label { font-size: 0.82rem; color: #66738a; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 2px;}

.badge {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
    letter-spacing: 0.02em;
}
.badge-added { background: var(--brand-green); }
.badge-removed { background: var(--brand-red); }
.badge-modified { background: #d69b00; }
.badge-geometry { background: var(--brand-blue); }

.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--brand-navy);
    margin: 1.4rem 0 0.6rem 0;
    border-left: 5px solid var(--brand-cyan);
    padding-left: 10px;
}

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 8px 18px;
    background-color: #eef2f6;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: var(--brand-navy) !important;
    color: white !important;
}

.warning-box {
    background: #fff6e5;
    border-left: 4px solid var(--brand-amber);
    padding: 10px 16px;
    border-radius: 8px;
    margin-bottom: 8px;
    font-size: 0.9rem;
    color: #6a4c00;
}

footer {visibility: hidden;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

CATEGORY_BADGE_CLASS = {
    "ADDED": "badge-added",
    "REMOVED": "badge-removed",
    "MODIFIED": "badge-modified",
    "DIMENSION_CHANGE": "badge-modified",
    "TEXT_CHANGE": "badge-modified",
    "GEOMETRY_CHANGE": "badge-geometry",
}

# --------------------------------------------------------------------------- #
# Hero header
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
        <h1>📐 AI-Powered CAD Revision Analyzer</h1>
        <p>Upload a BEFORE and AFTER engineering drawing to automatically detect added, removed,
        modified, dimension, text, and geometry changes — with a hybrid CV + OCR + YOLO + Hungarian
        matching pipeline, numbered color-coded annotations, and exportable JSON / Markdown reports.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Sidebar — configuration
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚙️ Run Configuration")
    page_index = st.number_input("Page number to compare", min_value=0, value=0, step=1)

    st.markdown("### 🎨 Legend")
    st.markdown(
        """
        - 🟢 **Added**
        - 🔴 **Removed**
        - 🟡 **Modified / Dimension / Text**
        - 🔵 **Geometry Changed**
        """
    )

    st.markdown("### 🧠 Pipeline Stages")
    st.caption(
        "PDF Render → Preprocess → ECC Alignment → SSIM+Canny+Morphology "
        "Diffing → OCR → YOLOv8 Detection → Hungarian Matching → "
        "Classification → Visualization → Reporting"
    )

    st.markdown("---")
    st.caption(f"Render DPI: `{settings.pdf.render_dpi}`")
    st.caption(f"Alignment motion model: `{settings.alignment.motion_type}`")
    st.caption(f"YOLO classes: `{len(settings.yolo.class_names)}`")


# --------------------------------------------------------------------------- #
# Upload section
# --------------------------------------------------------------------------- #
st.markdown('<div class="section-title">1. Upload Drawings</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    before_file = st.file_uploader("BEFORE drawing (PDF)", type=["pdf"], key="before_pdf")
with col2:
    after_file = st.file_uploader("AFTER drawing (PDF)", type=["pdf"], key="after_pdf")

run_clicked = st.button("🚀 Run Revision Analysis", type="primary", use_container_width=True, disabled=not (before_file and after_file))

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None


@st.cache_resource(show_spinner=False)
def get_pipeline() -> CADRevisionPipeline:
    return CADRevisionPipeline()


if run_clicked and before_file and after_file:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        before_path = tmp_path / "before.pdf"
        after_path = tmp_path / "after.pdf"
        before_path.write_bytes(before_file.getvalue())
        after_path.write_bytes(after_file.getvalue())

        output_dir = tmp_path / "output"

        progress = st.progress(0, text="Initializing pipeline...")
        try:
            pipeline = get_pipeline()
            progress.progress(15, text="Rendering PDFs & running hybrid CV/OCR/YOLO pipeline...")
            start = time.time()
            result = pipeline.run(before_path, after_path, output_dir=output_dir, page_index=int(page_index))
            elapsed = time.time() - start
            progress.progress(100, text=f"Done in {elapsed:.1f}s")

            # Persist bytes (not paths, since the tempdir is about to be deleted)
            st.session_state.pipeline_result = {
                "changes": result.changes,
                "report": result.report,
                "json_bytes": Path(result.json_report_path).read_bytes(),
                "md_bytes": Path(result.markdown_report_path).read_bytes(),
                "png_bytes": Path(result.annotated_image_path).read_bytes(),
                "ssim_score": result.ssim_score,
                "alignment_quality": result.alignment_quality,
                "warnings": result.warnings,
                "timings": result.timings,
            }
        except Exception as exc:  # noqa: BLE001
            progress.empty()
            st.error(f"Pipeline failed: {exc}")
            st.exception(exc)
            st.session_state.pipeline_result = None


# --------------------------------------------------------------------------- #
# Results section
# --------------------------------------------------------------------------- #
data = st.session_state.pipeline_result

if data:
    changes = data["changes"]
    report = data["report"]

    for w in data["warnings"]:
        st.markdown(f'<div class="warning-box">⚠️ {w}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">2. Summary Dashboard</div>', unsafe_allow_html=True)
    counts = report["summary"]["by_category"]
    m1, m2, m3, m4, m5 = st.columns(5)
    metrics = [
        ("Total Changes", report["summary"]["total_changes"], m1),
        ("Added", counts.get("ADDED", 0), m2),
        ("Removed", counts.get("REMOVED", 0), m3),
        ("Modified", counts.get("MODIFIED", 0) + counts.get("DIMENSION_CHANGE", 0) + counts.get("TEXT_CHANGE", 0), m4),
        ("Geometry Changed", counts.get("GEOMETRY_CHANGE", 0), m5),
    ]
    for label, value, col in metrics:
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{value}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    q1, q2 = st.columns(2)
    with q1:
        st.metric("Global SSIM Score", f"{data['ssim_score']:.3f}")
    with q2:
        st.metric("Alignment Quality", f"{data['alignment_quality']:.3f}")

    st.markdown('<div class="section-title">3. Annotated Comparison</div>', unsafe_allow_html=True)
    st.image(data["png_bytes"], use_container_width=True, caption="Numbered, color-coded change annotations with legend")

    st.markdown('<div class="section-title">4. Detailed Change Log</div>', unsafe_allow_html=True)

    tab_table, tab_cards, tab_charts = st.tabs(["📋 Table View", "🗂️ Card View", "📊 Charts"])

    df = pd.DataFrame(report["changes"])
    if not df.empty:
        df_display = df[["id", "category", "confidence", "object_class", "before_text", "after_text", "reasons"]].copy()
        df_display["confidence"] = (df_display["confidence"] * 100).round(1).astype(str) + "%"
        df_display["reasons"] = df_display["reasons"].apply(lambda r: "; ".join(r) if isinstance(r, list) else r)

    with tab_table:
        if df.empty:
            st.info("No changes detected between the two drawings.")
        else:
            category_filter = st.multiselect(
                "Filter by category",
                options=sorted(df["category"].unique()),
                default=sorted(df["category"].unique()),
            )
            filtered = df_display[df["category"].isin(category_filter)]
            st.dataframe(filtered, use_container_width=True, hide_index=True)

    with tab_cards:
        if df.empty:
            st.info("No changes detected between the two drawings.")
        else:
            for change in report["changes"]:
                badge_class = CATEGORY_BADGE_CLASS.get(change["category"], "badge-geometry")
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(
                            f"**#{change['id']}** &nbsp; "
                            f'<span class="badge {badge_class}">{change["category"]}</span>',
                            unsafe_allow_html=True,
                        )
                        if change["reasons"]:
                            st.caption(" • ".join(change["reasons"]))
                        bt = change["before_text"] or (change["before_value"] if change["before_value"] is not None else None)
                        at = change["after_text"] or (change["after_value"] if change["after_value"] is not None else None)
                        if bt or at:
                            st.write(f"Before: `{bt or '—'}`  →  After: `{at or '—'}`")
                    with c2:
                        st.metric("Confidence", f"{change['confidence'] * 100:.0f}%")

    with tab_charts:
        if df.empty:
            st.info("No changes detected between the two drawings.")
        else:
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown("**Changes by Category**")
                st.bar_chart(df["category"].value_counts())
            with chart_col2:
                st.markdown("**Confidence Distribution**")
                st.bar_chart(df["confidence"])

    st.markdown('<div class="section-title">5. Download Reports</div>', unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "⬇️ Download JSON Report", data=data["json_bytes"],
            file_name="report.json", mime="application/json", use_container_width=True,
        )
    with d2:
        st.download_button(
            "⬇️ Download Markdown Report", data=data["md_bytes"],
            file_name="report.md", mime="text/markdown", use_container_width=True,
        )
    with d3:
        st.download_button(
            "⬇️ Download Annotated Image", data=data["png_bytes"],
            file_name="annotated_comparison.png", mime="image/png", use_container_width=True,
        )

    with st.expander("⏱️ Stage Timing Breakdown"):
        timing_df = pd.DataFrame(
            [{"Stage": t.stage, "Seconds": round(t.seconds, 3)} for t in data["timings"]]
        )
        st.bar_chart(timing_df.set_index("Stage"))

else:
    st.info("Upload BEFORE and AFTER drawing PDFs above, then click **Run Revision Analysis** to begin.")
