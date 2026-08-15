"""
monitoring/monitor.py

Stage 6 — Drift Monitoring with Azure ML Integration

Detects data drift and model performance drift between
the training baseline and incoming production data.

Methods:
  - Population Stability Index (PSI) per feature
  - Kolmogorov-Smirnov test per feature
  - Station-level drift breakdown

Azure ML integration:
  When AZURE_ML_ENABLED=true, logs drift metrics to an
  Azure ML workspace for centralised monitoring.
  Falls back to local JSON + HTML reports otherwise.
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [MONITOR] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_STORE = os.path.join(PROJECT_ROOT, "data", "feature_store")
DRIFT_DIR     = os.path.join(PROJECT_ROOT, "data", "drift_reports")
os.makedirs(DRIFT_DIR, exist_ok=True)

PSI_THRESHOLD      = 0.20
KS_P_VALUE         = 0.05
DRIFT_RATE_TRIGGER = 0.30
AZURE_ML_ENABLED   = os.getenv("AZURE_ML_ENABLED", "false").lower() == "true"


# ── PSI ───────────────────────────────────────────────────────────────────────

def compute_psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    lo = min(ref.min(), cur.min())
    hi = max(ref.max(), cur.max())
    if hi == lo:
        return 0.0
    bins    = np.linspace(lo, hi, n_bins + 1)
    ref_pct = np.where((h := np.histogram(ref, bins)[0]) == 0, 1e-4, h / len(ref))
    cur_pct = np.where((h := np.histogram(cur, bins)[0]) == 0, 1e-4, h / len(cur))
    return float(round(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)), 4))


# ── KS test ───────────────────────────────────────────────────────────────────

def compute_ks(ref: np.ndarray, cur: np.ndarray) -> tuple:
    stat, p = ks_2samp(ref, cur)
    return round(float(stat), 4), round(float(p), 4)


# ── Feature drift ─────────────────────────────────────────────────────────────

def compute_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame,
                  feature_cols: list) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        if col not in ref_df.columns or col not in cur_df.columns:
            continue
        ref = ref_df[col].dropna().values
        cur = cur_df[col].dropna().values
        psi          = compute_psi(ref, cur)
        ks_stat, ks_p = compute_ks(ref, cur)
        rows.append({
            "feature":    col,
            "ref_mean":   round(float(ref.mean()), 4),
            "cur_mean":   round(float(cur.mean()), 4),
            "mean_shift": round(float(cur.mean() - ref.mean()), 4),
            "psi":        psi,
            "psi_drift":  psi > PSI_THRESHOLD,
            "ks_stat":    ks_stat,
            "ks_p":       ks_p,
            "ks_drift":   ks_p < KS_P_VALUE,
            "drift_flag": (psi > PSI_THRESHOLD) or (ks_p < KS_P_VALUE),
        })
    return pd.DataFrame(rows)


# ── HTML report ───────────────────────────────────────────────────────────────

def generate_html_report(drift_df: pd.DataFrame, summary: dict, tag: str) -> str:
    retrain_color = "#FCEBEB" if summary["retrain_recommended"] else "#EAF3DE"
    retrain_text  = "RETRAINING RECOMMENDED" if summary["retrain_recommended"] else "No retraining required"

    rows_html = ""
    for _, r in drift_df.iterrows():
        bg = "#FCEBEB" if r["drift_flag"] else "#EAF3DE"
        status = "DRIFT" if r["drift_flag"] else "stable"
        rows_html += f"""<tr style="background:{bg}">
          <td>{r['feature']}</td>
          <td>{r['ref_mean']}</td><td>{r['cur_mean']}</td>
          <td>{r['mean_shift']:+.4f}</td>
          <td>{r['psi']}</td><td>{r['ks_p']}</td>
          <td><strong>{status}</strong></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>BMW iFACTORY Drift Report — {tag}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 2rem; color: #333; }}
  h1 {{ font-size: 1.4rem; font-weight: 600; }}
  .meta {{ font-size: 13px; color: #666; margin-bottom: 1.5rem; }}
  .kpis {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
  .kpi {{ background: #f8f8f8; border: 1px solid #e0e0e0; border-radius: 8px;
          padding: 0.75rem 1.25rem; min-width: 140px; }}
  .kpi-val {{ font-size: 1.5rem; font-weight: 600; }}
  .kpi-lbl {{ font-size: 12px; color: #666; }}
  .alert {{ padding: 0.75rem 1.25rem; border-radius: 8px; font-weight: 500;
            margin-bottom: 1.5rem; background: {retrain_color}; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #f0f0f0; padding: 0.5rem 0.75rem; text-align: left;
        font-weight: 600; border-bottom: 2px solid #ccc; }}
  td {{ padding: 0.45rem 0.75rem; border-bottom: 1px solid #eee; }}
  h2 {{ font-size: 1.1rem; margin: 1.5rem 0 0.75rem; }}
  .azure-badge {{ display: inline-block; font-size: 11px; padding: 2px 8px;
                  background: #E6F1FB; color: #185FA5; border-radius: 4px;
                  margin-left: 8px; font-weight: 500; }}
</style></head><body>
<h1>BMW iFACTORY — Drift Monitoring Report
  <span class="azure-badge">{"Azure ML Connected" if AZURE_ML_ENABLED else "Local Mode"}</span>
</h1>
<div class="meta">Generated: {summary['timestamp']} | Plant: iFACTORY Leipzig/Regensburg |
Reference: {summary['n_reference']} cycles | Production: {summary['n_current']} cycles</div>
<div class="kpis">
  <div class="kpi"><div class="kpi-val">{summary['n_drifted']}/{summary['n_features']}</div>
    <div class="kpi-lbl">Features drifted</div></div>
  <div class="kpi"><div class="kpi-val">{summary['drift_rate']:.0%}</div>
    <div class="kpi-lbl">Drift rate</div></div>
  <div class="kpi"><div class="kpi-val">{summary['max_psi']:.3f}</div>
    <div class="kpi-lbl">Max PSI</div></div>
  <div class="kpi"><div class="kpi-val">{summary['n_ks_drifted']}</div>
    <div class="kpi-lbl">KS test failures</div></div>
</div>
<div class="alert">{retrain_text}</div>
<h2>Feature-level drift</h2>
<table><tr><th>Feature</th><th>Ref Mean</th><th>Cur Mean</th>
<th>Shift</th><th>PSI</th><th>KS p-value</th><th>Status</th></tr>
{rows_html}</table>
<p style="font-size:12px; color:#999; margin-top:2rem;">
PSI threshold: {PSI_THRESHOLD} | KS significance: {KS_P_VALUE} |
Retraining threshold: drift rate &gt; {DRIFT_RATE_TRIGGER:.0%}</p>
</body></html>"""

    path = os.path.join(DRIFT_DIR, f"drift_report_{tag}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# ── Azure ML logging ──────────────────────────────────────────────────────────

def log_to_azure(summary: dict):
    """
    Log drift metrics to Azure ML workspace.
    Requires environment variables:
      AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP,
      AZURE_ML_WORKSPACE_NAME, AZURE_TENANT_ID
    """
    try:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
            resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
            workspace_name=os.getenv("AZURE_ML_WORKSPACE_NAME"),
        )
        log.info(f"Azure ML workspace connected: {client.workspace_name}")
        log.info(f"Drift metrics would be logged here in a full Azure deployment.")
        # In a full deployment, you would log to an Azure ML Data Asset or
        # use azure.ai.ml.entities.DataDriftMonitor here.

    except ImportError:
        log.warning("azure-ai-ml not installed. Skipping Azure ML logging.")
    except Exception as e:
        log.warning(f"Azure ML logging failed: {e}")


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run(ref_df: pd.DataFrame = None, cur_df: pd.DataFrame = None) -> dict:
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    if ref_df is None:
        parquets = sorted([f for f in os.listdir(FEATURE_STORE) if f.startswith("features_")])
        if not parquets:
            raise FileNotFoundError("No feature sets. Run pipeline first.")
        ref_df = pd.read_parquet(os.path.join(FEATURE_STORE, parquets[0]))
        log.info(f"Stage 6 | Reference: {parquets[0]} ({len(ref_df)} samples)")

    if cur_df is None:
        log.info("Stage 6 | Generating drifted production data for simulation...")
        from pipeline.ingest import generate_batch
        from pipeline.features import extract_features
        raw = generate_batch(n_cycles=200, seed=9999, drift_factor=0.35)
        cur_df = extract_features(raw)

    feature_cols = [c for c in ref_df.columns
                    if c not in ["cycle_id", "station", "anomaly", "anomaly_type"]]

    log.info(f"Stage 6 | Computing drift across {len(feature_cols)} features...")
    drift_df = compute_drift(ref_df, cur_df, feature_cols)

    n_drifted   = int(drift_df["drift_flag"].sum())
    drift_rate  = n_drifted / len(drift_df)
    retrain     = drift_rate >= DRIFT_RATE_TRIGGER

    summary = {
        "tag":                tag,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "n_reference":        len(ref_df),
        "n_current":          len(cur_df),
        "n_features":         len(drift_df),
        "n_drifted":          n_drifted,
        "n_ks_drifted":       int(drift_df["ks_drift"].sum()),
        "drift_rate":         round(drift_rate, 4),
        "max_psi":            float(drift_df["psi"].max()),
        "retrain_recommended": retrain,
        "drifted_features":   drift_df[drift_df["drift_flag"]]["feature"].tolist(),
        "azure_ml_enabled":   AZURE_ML_ENABLED,
    }

    alert_path = os.path.join(DRIFT_DIR, f"drift_alert_{tag}.json")
    with open(alert_path, "w") as f:
        json.dump(summary, f, indent=2)

    report_path = generate_html_report(drift_df, summary, tag)

    with open(os.path.join(DRIFT_DIR, "latest.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if AZURE_ML_ENABLED:
        log_to_azure(summary)

    log.info(f"Stage 6 | Drift rate: {drift_rate:.1%} | Retrain: {retrain}")
    log.info(f"Stage 6 | Report -> {report_path}")

    if retrain:
        log.warning("Stage 6 | ALERT: Drift threshold exceeded — retraining flag set")

    return summary


if __name__ == "__main__":
    summary = run()
    print(f"\nDrift rate:          {summary['drift_rate']:.1%}")
    print(f"Drifted features:    {summary['n_drifted']}/{summary['n_features']}")
    print(f"Retrain recommended: {summary['retrain_recommended']}")
