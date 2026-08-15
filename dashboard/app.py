"""
dashboard/app.py

BMW iFACTORY MLOps Dashboard

Tabs:
  1. Production Health  — station anomaly rates, live KPIs
  2. Experiment History — MLflow run comparison
  3. Drift Monitor      — PSI/KS drift status, HTML report embed
  4. Live Prediction    — submit a cycle reading for instant classification
  5. Pipeline Control   — one-click retraining
"""

import os
import sys
import json
import subprocess
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="BMW iFACTORY MLOps",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
DRIFT_DIR     = os.path.join(PROJECT_ROOT, "data", "drift_reports")
FEATURE_STORE = os.path.join(PROJECT_ROOT, "data", "feature_store")
MLFLOW_URI    = f"sqlite:///{os.path.join(MODELS_DIR, 'mlflow.db')}"

STATIONS = ["biw_welding", "paint_curing", "powertrain_torque",
            "closure_assembly", "eol_functional"]
SENSOR_COLS = ["torque_nm", "vibration_g", "temperature_c",
               "cycle_time_s", "current_a", "pressure_bar"]


# ── Loaders ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=20)
def load_registry_info():
    path = os.path.join(MODELS_DIR, "registry_info.json")
    return json.load(open(path)) if os.path.exists(path) else None


@st.cache_data(ttl=20)
def load_drift():
    path = os.path.join(DRIFT_DIR, "latest.json")
    return json.load(open(path)) if os.path.exists(path) else None


@st.cache_data(ttl=20)
def load_training_summary():
    path = os.path.join(MODELS_DIR, "training_summary.json")
    return json.load(open(path)) if os.path.exists(path) else []


@st.cache_data(ttl=20)
def load_mlflow_runs():
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient(tracking_uri=MLFLOW_URI)
        exp    = client.get_experiment_by_name("bmw_ifactory_anomaly_detection")
        if exp is None:
            return pd.DataFrame()
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.cv_f1 DESC"], max_results=50,
        )
        return pd.DataFrame([{
            "run_id":    r.info.run_id[:8],
            "model":     r.data.tags.get("mlflow.runName", "unknown"),
            "f1":        round(r.data.metrics.get("cv_f1", 0), 4),
            "precision": round(r.data.metrics.get("precision", 0), 4),
            "recall":    round(r.data.metrics.get("recall", 0), 4),
            "roc_auc":   round(r.data.metrics.get("roc_auc", 0), 4),
            "start":     pd.to_datetime(r.info.start_time, unit="ms").strftime("%Y-%m-%d %H:%M"),
        } for r in runs])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=20)
def load_latest_raw():
    raw_dir  = os.path.join(PROJECT_ROOT, "data", "raw")
    batches  = sorted([f for f in os.listdir(raw_dir) if f.startswith("batch_")])
    if not batches:
        return pd.DataFrame()
    return pd.read_parquet(os.path.join(raw_dir, batches[-1]))


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    st.sidebar.markdown("## BMW iFACTORY MLOps")
    st.sidebar.markdown(
        "Production anomaly detection pipeline for BMW assembly line stations. "
        "Mirrors the iFACTORY AI platform across Regensburg, Leipzig, and Debrecen."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Services")
    st.sidebar.markdown(
        "- [MLflow UI](http://localhost:5000)\n"
        "- [Prediction API](http://localhost:8000/docs)"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Pipeline stages")
    for s in ["1 — Ingestion + Validation", "2 — Feature Engineering",
              "3 — Model Training (MLflow)", "4 — Registry Promotion",
              "5 — FastAPI Serving", "6 — Drift Monitoring (Azure ML)"]:
        st.sidebar.markdown(f"- {s}")

    if st.sidebar.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ── Tab 1: Production health ──────────────────────────────────────────────────

def render_production_health(raw_df: pd.DataFrame, registry):
    st.subheader("Production Health — Assembly Line Stations")

    if raw_df.empty:
        st.info("No production data. Run python run_pipeline.py first.")
        return

    total      = len(raw_df)
    anomalies  = int(raw_df["anomaly"].sum())
    anom_rate  = anomalies / total

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total cycles",     f"{total:,}")
    c2.metric("Anomalies detected", f"{anomalies:,}",
              delta=f"{anom_rate:.1%}", delta_color="inverse")
    c3.metric("Production model",
              registry.get("model_name", "—") if registry else "—")
    c4.metric("Model F1",
              f"{registry.get('cv_f1', 0):.4f}" if registry else "—")

    st.markdown("")

    # Per-station anomaly rates
    station_stats = raw_df.groupby("station").agg(
        total=("anomaly", "count"),
        anomalies=("anomaly", "sum"),
    ).reset_index()
    station_stats["anomaly_rate"] = station_stats["anomalies"] / station_stats["total"]

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(go.Bar(
            x=station_stats["station"],
            y=station_stats["anomaly_rate"] * 100,
            marker_color=[
                "#e74c3c" if r > 0.08 else "#f39c12" if r > 0.04 else "#2ecc71"
                for r in station_stats["anomaly_rate"]
            ],
            text=[f"{r:.1%}" for r in station_stats["anomaly_rate"]],
            textposition="outside",
        ))
        fig.update_layout(
            title="Anomaly rate per station",
            yaxis_title="Anomaly rate (%)",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        anomaly_counts = raw_df[raw_df["anomaly"] == 1]["anomaly_type"].value_counts()
        fig = px.pie(
            values=anomaly_counts.values,
            names=anomaly_counts.index,
            title="Anomaly type distribution",
            hole=0.5,
        )
        fig.update_layout(
            height=300, margin=dict(l=0, r=0, t=40, b=0),
            paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Sensor box plots
    st.markdown("### Sensor distribution by station")
    sensor_sel = st.selectbox("Sensor", SENSOR_COLS)
    fig = px.box(
        raw_df, x="station", y=sensor_sel,
        color="station",
        title=f"{sensor_sel} distribution across stations",
        points="outliers",
    )
    fig.update_layout(
        height=340, showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Experiment history ─────────────────────────────────────────────────

def render_experiments(runs_df: pd.DataFrame):
    st.subheader("MLflow Experiment History")

    if runs_df.empty:
        st.info("No MLflow runs. Run python run_pipeline.py first.")
        return

    # Metrics comparison bar chart
    melted = runs_df.melt(id_vars=["model", "run_id"],
                          value_vars=["f1", "precision", "recall", "roc_auc"],
                          var_name="metric", value_name="value")
    fig = px.bar(
        melted, x="model", y="value", color="metric",
        barmode="group", title="Model metrics comparison",
        range_y=[0, 1.1],
    )
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        runs_df.style.highlight_max(subset=["f1", "recall", "roc_auc"], color="#EAF3DE"),
        use_container_width=True, hide_index=True,
    )


# ── Tab 3: Drift monitor ──────────────────────────────────────────────────────

def render_drift(drift: dict):
    st.subheader("Data Drift Monitor")

    if drift is None:
        st.info("No drift reports. Run python run_pipeline.py.")
        return

    retrain_color = "#FCEBEB" if drift.get("retrain_recommended") else "#EAF3DE"
    status_text   = "RETRAINING RECOMMENDED" if drift.get("retrain_recommended") else "Data Distribution Stable"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""<div style="background:{retrain_color}; border-radius:8px;
                            padding:1rem 1.25rem; margin-bottom:1rem;">
                <h3 style="margin:0;">{status_text}</h3>
                <p style="margin:0.4rem 0 0; font-size:13px; color:#555;">
                    {drift.get('n_drifted', 0)}/{drift.get('n_features', 0)} features drifted
                    ({drift.get('drift_rate', 0):.0%}) |
                    Max PSI: {drift.get('max_psi', 0):.3f}
                </p>
                <p style="margin:0.2rem 0 0; font-size:11px; color:#777;">
                    Azure ML: {"Connected" if drift.get("azure_ml_enabled") else "Local mode"} |
                    Checked: {drift.get('timestamp', '')[:19]}
                </p>
            </div>""",
            unsafe_allow_html=True,
        )
        if drift.get("drifted_features"):
            st.markdown("**Drifted features:**")
            for f in drift["drifted_features"][:10]:
                st.markdown(f"- `{f}`")

    with col2:
        rate = drift.get("drift_rate", 0)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rate * 100,
            title={"text": "Drift Rate (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#e74c3c" if rate > 0.30 else "#f39c12" if rate > 0.10 else "#2ecc71"},
                "steps": [
                    {"range": [0, 10],  "color": "#EAF3DE"},
                    {"range": [10, 30], "color": "#FAEEDA"},
                    {"range": [30, 100],"color": "#FCEBEB"},
                ],
                "threshold": {"line": {"color": "#e74c3c", "width": 3},
                              "thickness": 0.8, "value": 30},
            },
        ))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    reports = sorted([f for f in os.listdir(DRIFT_DIR) if f.endswith(".html")], reverse=True)
    if reports:
        with st.expander("View full drift report"):
            with open(os.path.join(DRIFT_DIR, reports[0])) as f:
                st.components.v1.html(f.read(), height=600, scrolling=True)


# ── Tab 4: Live prediction ────────────────────────────────────────────────────

def render_live_prediction():
    st.subheader("Live Cycle Prediction")
    st.markdown(
        "Submit a single assembly cycle reading to the production model "
        "for instant anomaly classification."
    )

    station = st.selectbox("Station", STATIONS)
    st.markdown("**Sensor readings:**")
    cols = st.columns(3)
    torque    = cols[0].number_input("torque_nm",     0.0,  500.0, 180.0, step=1.0)
    vibration = cols[1].number_input("vibration_g",   0.0,   15.0,   1.2, step=0.1)
    temp      = cols[2].number_input("temperature_c", 10.0, 220.0,  95.0, step=1.0)
    cycle_t   = cols[0].number_input("cycle_time_s",  5.0,  300.0,  42.0, step=1.0)
    current   = cols[1].number_input("current_a",     0.0,   80.0,  35.0, step=0.5)
    pressure  = cols[2].number_input("pressure_bar",  0.0,   12.0,   6.5, step=0.1)

    if st.button("Predict", type="primary"):
        try:
            import requests
            from pipeline.features import extract_features
            import pandas as pd

            raw = pd.DataFrame([{
                "cycle_id": 0, "station": station,
                "timestamp": datetime.now().isoformat(),
                "torque_nm": torque, "vibration_g": vibration,
                "temperature_c": temp, "cycle_time_s": cycle_t,
                "current_a": current, "pressure_bar": pressure,
                "anomaly": 0, "anomaly_type": "none",
            }])
            feat = extract_features(raw)
            feat_cols = [c for c in feat.columns
                         if c not in ["cycle_id", "station", "anomaly", "anomaly_type"]]
            features = feat[feat_cols].values[0].tolist()

            resp = requests.post(
                "http://localhost:8000/predict",
                json={"features": features, "station": station, "cycle_id": "dashboard_test"},
                timeout=5,
            )
            if resp.status_code == 200:
                r = resp.json()
                color = "#e74c3c" if r["anomaly"] == 1 else "#2ecc71"
                st.markdown(
                    f"""<div style="background:{color}22; border-left:5px solid {color};
                                    border-radius:8px; padding:1rem 1.5rem; margin-top:1rem;">
                        <h3 style="margin:0; color:{color};">{r['anomaly_label']}</h3>
                        <p style="margin:0.3rem 0 0; font-size:13px; color:#555;">
                            Confidence: {r['confidence']:.1%} |
                            Station: {station} |
                            Model v{r['model_version']}
                        </p>
                        <p style="margin:0.3rem 0 0; font-size:12px; color:#777;">
                            {r['recommended_action']}
                        </p>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.error(f"API error: {resp.status_code}")
        except Exception as e:
            st.warning(f"API not reachable ({e}). Run: uvicorn serving.serve:app --port 8000")


# ── Tab 5: Pipeline control ───────────────────────────────────────────────────

def render_pipeline_control():
    st.subheader("Pipeline Control")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Run Full Pipeline", type="primary", use_container_width=True):
            with st.spinner("Running pipeline..."):
                try:
                    result = subprocess.run(
                        [sys.executable, os.path.join(PROJECT_ROOT, "run_pipeline.py")],
                        capture_output=True, text=True, timeout=300,
                    )
                    if result.returncode == 0:
                        st.success("Pipeline completed.")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error("Pipeline failed.")
                        st.code(result.stderr[-3000:], language="bash")
                except subprocess.TimeoutExpired:
                    st.warning("Pipeline timed out after 5 minutes.")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        st.code("""# Individual stage commands
python pipeline/ingest.py       # Stage 1 — ingest + validate
python pipeline/features.py     # Stage 2 — feature store
python pipeline/train.py        # Stage 3 — MLflow training
python pipeline/evaluate.py     # Stage 4 — registry promotion
python monitoring/monitor.py    # Stage 6 — drift check

# Services
uvicorn serving.serve:app --port 8000       # Prediction API
mlflow ui --backend-store-uri sqlite:///models/mlflow.db --port 5000
""", language="bash")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    st.markdown("# BMW iFACTORY — Production Anomaly Detection MLOps")
    st.markdown(
        "End-to-end ML pipeline for real-time assembly line anomaly detection. "
        "Mirrors BMW's iFACTORY AI platform across Regensburg, Leipzig, and Debrecen plants."
    )
    st.markdown("---")

    registry = load_registry_info()
    drift    = load_drift()
    runs_df  = load_mlflow_runs()
    raw_df   = load_latest_raw()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Production Health",
        "Experiment History",
        "Drift Monitor",
        "Live Prediction",
        "Pipeline Control",
    ])

    with tab1: render_production_health(raw_df, registry)
    with tab2: render_experiments(runs_df)
    with tab3: render_drift(drift)
    with tab4: render_live_prediction()
    with tab5: render_pipeline_control()


if __name__ == "__main__":
    main()
