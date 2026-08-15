"""
pipeline/features.py

Stage 2 — Feature Engineering

Extracts per-cycle features from raw assembly sensor readings.
Groups by station so station-specific baselines are captured.

Feature tiers:
  1. Per-sensor statistics  — mean, std, max, min, RMS, IQR per sensor
  2. Ratio features         — torque/current (efficiency proxy),
                              temp/cycle_time (thermal load rate)
  3. Station encoding       — one-hot encoding of station
  4. Cross-cycle rolling    — rolling mean and std over last 10 cycles
                              per station (drift proxy)
"""

import os
import json
import hashlib
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [FEATURES] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
FEATURE_STORE = os.path.join(PROJECT_ROOT, "data", "feature_store")
os.makedirs(FEATURE_STORE, exist_ok=True)

SENSOR_COLS = [
    "torque_nm", "vibration_g", "temperature_c",
    "cycle_time_s", "current_a", "pressure_bar",
]

STATIONS = [
    "biw_welding", "paint_curing", "powertrain_torque",
    "closure_assembly", "eol_functional",
]


# ── Per-cycle feature extraction ──────────────────────────────────────────────

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each row in df is already one cycle reading (not a time-series).
    We extract ratio features, station encoding, and rolling features.
    """
    feat = df[["cycle_id", "station", "anomaly", "anomaly_type"]].copy()

    # Raw sensor values as features
    for col in SENSOR_COLS:
        feat[col] = df[col].astype(float)

    # Ratio features
    feat["torque_current_ratio"] = (
        df["torque_nm"] / (df["current_a"].replace(0, np.nan))
    ).fillna(0).round(4)

    feat["thermal_load_rate"] = (
        df["temperature_c"] / (df["cycle_time_s"].replace(0, np.nan))
    ).fillna(0).round(4)

    feat["pressure_torque_ratio"] = (
        df["pressure_bar"] / (df["torque_nm"].replace(0, np.nan))
    ).fillna(0).round(4)

    # Deviation from station baseline
    station_means = df.groupby("station")[SENSOR_COLS].transform("mean")
    station_stds  = df.groupby("station")[SENSOR_COLS].transform("std").replace(0, 1)
    for col in SENSOR_COLS:
        feat[f"{col}_zscore"] = (
            (df[col] - station_means[col]) / station_stds[col]
        ).round(4)

    # Station one-hot encoding
    for s in STATIONS:
        feat[f"station_{s}"] = (df["station"] == s).astype(int)

    # Cross-cycle rolling features (per station, last 10 cycles)
    feat = feat.sort_values("cycle_id").reset_index(drop=True)
    for col in ["torque_nm", "vibration_g", "temperature_c", "current_a"]:
        feat[f"{col}_roll_mean"] = (
            feat.groupby("station")[col]
            .transform(lambda x: x.rolling(10, min_periods=1).mean())
            .round(4)
        )
        feat[f"{col}_roll_std"] = (
            feat.groupby("station")[col]
            .transform(lambda x: x.rolling(10, min_periods=1).std().fillna(0))
            .round(4)
        )

    return feat


# ── Feature store ─────────────────────────────────────────────────────────────

def save_features(feat_df: pd.DataFrame, tag: str = None) -> str:
    tag   = tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    fhash = hashlib.md5(
        ",".join(sorted([c for c in feat_df.columns if c != "anomaly"])).encode()
    ).hexdigest()[:8]
    version = f"{tag}_{fhash}"

    path = os.path.join(FEATURE_STORE, f"features_{version}.parquet")
    feat_df.to_parquet(path, index=False)

    feature_cols = [c for c in feat_df.columns
                    if c not in ["cycle_id", "station", "anomaly", "anomaly_type"]]
    manifest = {
        "version":      version,
        "tag":          tag,
        "feature_hash": fhash,
        "n_samples":    len(feat_df),
        "n_features":   len(feature_cols),
        "feature_cols": feature_cols,
        "anomaly_rate": round(float(feat_df["anomaly"].mean()), 4),
        "stations":     feat_df["station"].value_counts().to_dict(),
    }
    with open(os.path.join(FEATURE_STORE, f"manifest_{version}.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Stage 2 | Saved features v{version} — {len(feat_df)} x {len(feature_cols)}")
    return path


def load_latest_features() -> tuple:
    parquets = sorted([f for f in os.listdir(FEATURE_STORE) if f.startswith("features_")])
    if not parquets:
        raise FileNotFoundError("No feature sets found. Run pipeline first.")
    latest  = parquets[-1]
    version = latest.replace("features_", "").replace(".parquet", "")
    feat_df = pd.read_parquet(os.path.join(FEATURE_STORE, latest))
    with open(os.path.join(FEATURE_STORE, f"manifest_{version}.json")) as f:
        manifest = json.load(f)
    log.info(f"Stage 2 | Loaded features v{version} ({len(feat_df)} samples)")
    return feat_df, manifest


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run(raw_df: pd.DataFrame = None, tag: str = None) -> tuple:
    if raw_df is None:
        batches = sorted([f for f in os.listdir(RAW_DIR) if f.startswith("batch_")])
        if not batches:
            raise FileNotFoundError("No raw batches found. Run ingest.py first.")
        raw_df = pd.read_parquet(os.path.join(RAW_DIR, batches[-1]))
        log.info(f"Stage 2 | Loaded: {batches[-1]}")

    log.info(f"Stage 2 | Extracting features from {len(raw_df)} cycles...")
    feat_df = extract_features(raw_df)
    path    = save_features(feat_df, tag=tag)
    return feat_df, path


if __name__ == "__main__":
    feat_df, path = run()
    fc = [c for c in feat_df.columns if c not in ["cycle_id", "station", "anomaly", "anomaly_type"]]
    print(f"\nFeature matrix: {len(feat_df)} samples x {len(fc)} features")
    print(f"Saved: {path}")
