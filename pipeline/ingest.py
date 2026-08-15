"""
pipeline/ingest.py

Stage 1 — Data Ingestion and Validation

Simulates BMW iFACTORY assembly line sensor telemetry across
5 production stations. Validates every batch before it enters
the feature store or training pipeline.

Stations modelled:
  - Body-in-White (BiW) welding station
  - Paint shop curing oven
  - Powertrain assembly torque station
  - Final assembly closure panel station
  - End-of-line (EoL) functional test station

Sensor channels per station:
  torque_nm, vibration_g, temperature_c, cycle_time_s,
  current_a, pressure_bar
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [INGEST] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

STATIONS = [
    "biw_welding",
    "paint_curing",
    "powertrain_torque",
    "closure_assembly",
    "eol_functional",
]

SENSOR_COLS = [
    "torque_nm", "vibration_g", "temperature_c",
    "cycle_time_s", "current_a", "pressure_bar",
]

# Normal operating ranges per sensor
SCHEMA = {
    "torque_nm":      {"min": 0.0,   "max": 500.0},
    "vibration_g":    {"min": 0.0,   "max": 15.0},
    "temperature_c":  {"min": 10.0,  "max": 220.0},
    "cycle_time_s":   {"min": 5.0,   "max": 300.0},
    "current_a":      {"min": 0.0,   "max": 80.0},
    "pressure_bar":   {"min": 0.0,   "max": 12.0},
}

ANOMALY_TYPES = ["none", "torque_spike", "thermal_runaway",
                 "vibration_fault", "cycle_overrun", "pressure_drop"]

# Station-specific baseline sensor profiles
STATION_PROFILES = {
    "biw_welding":        {"torque_nm": 180, "vibration_g": 1.2, "temperature_c": 95,
                           "cycle_time_s": 42, "current_a": 35, "pressure_bar": 6.5},
    "paint_curing":       {"torque_nm": 20,  "vibration_g": 0.4, "temperature_c": 185,
                           "cycle_time_s": 180, "current_a": 55, "pressure_bar": 2.0},
    "powertrain_torque":  {"torque_nm": 320, "vibration_g": 1.8, "temperature_c": 75,
                           "cycle_time_s": 65, "current_a": 28, "pressure_bar": 8.0},
    "closure_assembly":   {"torque_nm": 45,  "vibration_g": 0.8, "temperature_c": 35,
                           "cycle_time_s": 38, "current_a": 12, "pressure_bar": 4.5},
    "eol_functional":     {"torque_nm": 95,  "vibration_g": 2.1, "temperature_c": 55,
                           "cycle_time_s": 95, "current_a": 22, "pressure_bar": 7.2},
}


# ── Synthetic data generator ──────────────────────────────────────────────────

def generate_batch(
    n_cycles: int = 500,
    anomaly_rate: float = 0.05,
    seed: int = 42,
    drift_factor: float = 0.0,
) -> pd.DataFrame:
    """
    Generate a batch of assembly cycle readings.

    drift_factor > 0 simulates gradual sensor drift (used for monitoring tests).
    anomaly_rate controls fraction of cycles with injected faults.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for cycle_id in range(n_cycles):
        station = STATIONS[cycle_id % len(STATIONS)]
        profile = STATION_PROFILES[station]
        noise   = 0.04  # 4% noise factor

        # Inject anomaly
        is_anomaly  = rng.random() < anomaly_rate
        anomaly_type = "none"
        if is_anomaly:
            anomaly_type = rng.choice(ANOMALY_TYPES[1:])

        def val(key, scale=1.0):
            base = profile[key] * scale
            return float(np.clip(
                base + base * noise * rng.standard_normal(),
                SCHEMA[key]["min"], SCHEMA[key]["max"]
            ))

        torque    = val("torque_nm")
        vibration = val("vibration_g")
        temp      = val("temperature_c")
        cycle     = val("cycle_time_s")
        current   = val("current_a")
        pressure  = val("pressure_bar")

        # Apply drift
        if drift_factor > 0:
            temp     *= (1.0 + drift_factor * 0.4)
            vibration *= (1.0 + drift_factor * 0.3)
            current  *= (1.0 + drift_factor * 0.2)

        # Apply anomaly perturbation
        if anomaly_type == "torque_spike":
            torque *= float(rng.uniform(1.8, 2.5))
        elif anomaly_type == "thermal_runaway":
            temp   *= float(rng.uniform(1.25, 1.5))
        elif anomaly_type == "vibration_fault":
            vibration *= float(rng.uniform(3.0, 5.0))
        elif anomaly_type == "cycle_overrun":
            cycle  *= float(rng.uniform(1.6, 2.2))
        elif anomaly_type == "pressure_drop":
            pressure *= float(rng.uniform(0.2, 0.4))

        rows.append({
            "cycle_id":     cycle_id,
            "station":      station,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "torque_nm":    round(min(torque,   SCHEMA["torque_nm"]["max"]),    2),
            "vibration_g":  round(min(vibration, SCHEMA["vibration_g"]["max"]), 4),
            "temperature_c":round(min(temp,      SCHEMA["temperature_c"]["max"]),2),
            "cycle_time_s": round(min(cycle,     SCHEMA["cycle_time_s"]["max"]), 2),
            "current_a":    round(min(current,   SCHEMA["current_a"]["max"]),   2),
            "pressure_bar": round(min(pressure,  SCHEMA["pressure_bar"]["max"]), 3),
            "anomaly":      int(is_anomaly),
            "anomaly_type": anomaly_type,
        })

    return pd.DataFrame(rows)


# ── Validation ────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def fail(self, msg: str):
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)


def validate(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult()

    if len(df) < 50:
        result.fail(f"Batch too small: {len(df)} rows (min 50)")

    missing = set(SENSOR_COLS + ["station", "anomaly"]) - set(df.columns)
    if missing:
        result.fail(f"Missing columns: {missing}")
        return result

    for col in SENSOR_COLS:
        null_rate = df[col].isnull().mean()
        if null_rate > 0.02:
            result.fail(f"Column '{col}' null rate {null_rate:.1%} exceeds 2%")

    for col, spec in SCHEMA.items():
        if col not in df.columns:
            continue
        out = ((df[col] < spec["min"]) | (df[col] > spec["max"])).sum()
        if out / len(df) > 0.05:
            result.fail(f"Column '{col}': {out} values out of valid range")

    missing_stations = set(STATIONS) - set(df["station"].unique())
    if missing_stations:
        result.warn(f"Stations missing from batch: {missing_stations}")

    result.stats = {
        "n_rows":           len(df),
        "n_cycles":         df["cycle_id"].nunique(),
        "anomaly_rate":     round(df["anomaly"].mean(), 4),
        "stations":         df["station"].value_counts().to_dict(),
        "col_means":        df[SENSOR_COLS].mean().round(3).to_dict(),
    }
    return result


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run(
    n_cycles: int = 500,
    seed: int = 42,
    tag: str = None,
    drift_factor: float = 0.0,
) -> Tuple[pd.DataFrame, ValidationResult]:
    log.info(f"Stage 1 | Generating BMW assembly line batch ({n_cycles} cycles)...")
    df = generate_batch(n_cycles=n_cycles, seed=seed, drift_factor=drift_factor)

    log.info("Stage 1 | Validating batch...")
    result = validate(df)

    for err in result.errors:
        log.error(f"         FAIL: {err}")
    for w in result.warnings:
        log.warning(f"         WARN: {w}")

    if not result.passed:
        log.error("Stage 1 | Validation FAILED — batch rejected")
        return df, result

    tag      = tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RAW_DIR, f"batch_{tag}.parquet")
    df.to_parquet(out_path, index=False)

    report = os.path.join(RAW_DIR, f"validation_{tag}.json")
    with open(report, "w") as f:
        json.dump({"passed": result.passed, "errors": result.errors,
                   "warnings": result.warnings, "stats": result.stats, "tag": tag}, f, indent=2)

    log.info(f"Stage 1 | Passed — {len(df)} rows saved -> {out_path}")
    log.info(f"         Anomaly rate: {result.stats['anomaly_rate']:.1%}")
    return df, result


if __name__ == "__main__":
    df, result = run()
    print(f"\nValidation passed: {result.passed}")
    print(f"Rows: {result.stats['n_rows']} | Anomaly rate: {result.stats['anomaly_rate']:.1%}")
