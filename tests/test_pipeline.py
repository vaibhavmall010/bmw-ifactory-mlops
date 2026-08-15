"""
tests/test_pipeline.py — BMW iFACTORY MLOps Pipeline Test Suite
Runs without live infrastructure (no MLflow server, no Azure, no API).
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

STATIONS = ["biw_welding", "paint_curing", "powertrain_torque",
            "closure_assembly", "eol_functional"]
SENSOR_COLS = ["torque_nm", "vibration_g", "temperature_c",
               "cycle_time_s", "current_a", "pressure_bar"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def raw_df():
    from pipeline.ingest import generate_batch
    return generate_batch(n_cycles=100, seed=0)


@pytest.fixture(scope="session")
def feat_df(raw_df):
    from pipeline.features import extract_features
    return extract_features(raw_df)


# ── Stage 1: Ingestion ────────────────────────────────────────────────────────

class TestIngestion:
    def test_generates_correct_row_count(self, raw_df):
        assert len(raw_df) == 100

    def test_all_sensor_columns_present(self, raw_df):
        for col in SENSOR_COLS:
            assert col in raw_df.columns

    def test_all_stations_represented(self, raw_df):
        assert set(raw_df["station"].unique()) == set(STATIONS)

    def test_torque_within_range(self, raw_df):
        assert raw_df["torque_nm"].between(0, 500).all()

    def test_vibration_within_range(self, raw_df):
        assert raw_df["vibration_g"].between(0, 15).all()

    def test_temperature_within_range(self, raw_df):
        assert raw_df["temperature_c"].between(10, 220).all()

    def test_anomaly_column_binary(self, raw_df):
        assert set(raw_df["anomaly"].unique()).issubset({0, 1})

    def test_anomaly_type_valid(self, raw_df):
        valid = {"none", "torque_spike", "thermal_runaway",
                 "vibration_fault", "cycle_overrun", "pressure_drop"}
        assert set(raw_df["anomaly_type"].unique()).issubset(valid)

    def test_anomaly_type_none_when_flag_zero(self, raw_df):
        normal = raw_df[raw_df["anomaly"] == 0]
        assert (normal["anomaly_type"] == "none").all()

    def test_validation_passes_good_data(self, raw_df):
        from pipeline.ingest import validate
        result = validate(raw_df)
        assert result.passed, f"Validation failed: {result.errors}"

    def test_validation_fails_missing_columns(self):
        from pipeline.ingest import validate
        bad = pd.DataFrame({"x": [1, 2, 3]})
        result = validate(bad)
        assert not result.passed

    def test_validation_fails_small_batch(self):
        from pipeline.ingest import validate, generate_batch
        tiny = generate_batch(n_cycles=5, seed=0)
        result = validate(tiny)
        assert not result.passed

    def test_drift_factor_increases_temperature(self):
        from pipeline.ingest import generate_batch
        df_normal = generate_batch(n_cycles=50, seed=0, drift_factor=0.0)
        df_drift  = generate_batch(n_cycles=50, seed=0, drift_factor=0.5)
        assert df_drift["temperature_c"].mean() > df_normal["temperature_c"].mean()


# ── Stage 2: Feature engineering ─────────────────────────────────────────────

class TestFeatures:
    def test_feature_row_count_matches_raw(self, raw_df, feat_df):
        assert len(feat_df) == len(raw_df)

    def test_raw_sensors_preserved(self, feat_df):
        for col in SENSOR_COLS:
            assert col in feat_df.columns

    def test_ratio_features_present(self, feat_df):
        assert "torque_current_ratio"  in feat_df.columns
        assert "thermal_load_rate"     in feat_df.columns
        assert "pressure_torque_ratio" in feat_df.columns

    def test_zscore_features_present(self, feat_df):
        for col in SENSOR_COLS:
            assert f"{col}_zscore" in feat_df.columns

    def test_station_encoding_present(self, feat_df):
        for s in STATIONS:
            assert f"station_{s}" in feat_df.columns

    def test_rolling_features_present(self, feat_df):
        for col in ["torque_nm", "vibration_g", "temperature_c", "current_a"]:
            assert f"{col}_roll_mean" in feat_df.columns
            assert f"{col}_roll_std"  in feat_df.columns

    def test_no_nan_in_features(self, feat_df):
        fc = [c for c in feat_df.columns
              if c not in ["cycle_id", "station", "anomaly", "anomaly_type"]]
        nan_cols = feat_df[fc].isnull().sum()
        assert nan_cols.sum() == 0, f"NaN in: {nan_cols[nan_cols > 0].index.tolist()}"

    def test_station_encoding_mutually_exclusive(self, feat_df):
        station_cols = [f"station_{s}" for s in STATIONS]
        row_sums = feat_df[station_cols].sum(axis=1)
        assert (row_sums == 1).all(), "Each row must have exactly one station encoded"

    def test_feature_count_reasonable(self, feat_df):
        fc = [c for c in feat_df.columns
              if c not in ["cycle_id", "station", "anomaly", "anomaly_type"]]
        assert len(fc) >= 25, f"Expected at least 25 features, got {len(fc)}"


# ── Anomaly detection model tests ─────────────────────────────────────────────

class TestModelTraining:
    def test_isolation_forest_trains(self, feat_df):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        fc = [c for c in feat_df.columns
              if c not in ["cycle_id", "station", "anomaly", "anomaly_type"]]
        X = feat_df[fc].values
        y = feat_df["anomaly"].values

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", IsolationForest(n_estimators=50, contamination=y.mean(), random_state=0)),
        ])
        pipe.fit(X)
        preds_raw = pipe.named_steps["model"].predict(
            pipe.named_steps["scaler"].transform(X)
        )
        preds = np.where(preds_raw == -1, 1, 0)
        assert len(preds) == len(y)

    def test_random_forest_trains_and_predicts(self, feat_df):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from pipeline.ingest import generate_batch
        from pipeline.features import extract_features

        raw = generate_batch(n_cycles=300, anomaly_rate=0.30, seed=7)
        fd  = extract_features(raw)
        fc  = [c for c in fd.columns
               if c not in ["cycle_id", "station", "anomaly", "anomaly_type"]]
        X = fd[fc].values
        y = fd["anomaly"].values

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=50, class_weight="balanced",
                                              random_state=0)),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({0, 1})

class TestDriftMonitoring:
    def test_psi_zero_for_identical_data(self):
        from monitoring.monitor import compute_psi
        data = np.random.default_rng(0).normal(50, 10, 500)
        assert compute_psi(data, data.copy()) < 0.01

    def test_psi_high_for_large_shift(self):
        from monitoring.monitor import compute_psi
        ref = np.random.default_rng(0).normal(50, 5, 500)
        cur = np.random.default_rng(1).normal(90, 5, 500)
        assert compute_psi(ref, cur) > 0.20

    def test_ks_detects_drift(self):
        from monitoring.monitor import compute_ks
        ref = np.random.default_rng(0).normal(0, 1, 500)
        cur = np.random.default_rng(1).normal(3, 1, 500)
        _, p = compute_ks(ref, cur)
        assert p < 0.05

    def test_ks_no_drift_same_distribution(self):
        from monitoring.monitor import compute_ks
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 500)
        cur = rng.normal(0, 1, 500)
        _, p = compute_ks(ref, cur)
        assert p > 0.01

    def test_drift_report_has_required_columns(self, feat_df):
        from monitoring.monitor import compute_drift
        fc = [c for c in feat_df.columns
              if c not in ["cycle_id", "station", "anomaly", "anomaly_type"]]
        drift_df = compute_drift(feat_df, feat_df, fc)
        for col in ["feature", "psi", "ks_p", "drift_flag"]:
            assert col in drift_df.columns

    def test_no_drift_on_identical_data(self, feat_df):
        from monitoring.monitor import compute_drift
        fc = [c for c in feat_df.columns
              if c not in ["cycle_id", "station", "anomaly", "anomaly_type"]]
        drift_df = compute_drift(feat_df, feat_df, fc)
        assert drift_df["drift_flag"].sum() == 0

    def test_drift_detected_on_shifted_data(self, raw_df, feat_df):
        from pipeline.ingest import generate_batch
        from pipeline.features import extract_features
        from monitoring.monitor import compute_drift
        drifted_raw = generate_batch(n_cycles=100, seed=1, drift_factor=0.6)
        drifted_feat = extract_features(drifted_raw)
        fc = [c for c in feat_df.columns
              if c not in ["cycle_id", "station", "anomaly", "anomaly_type"]
              and c in drifted_feat.columns]
        drift_df = compute_drift(feat_df, drifted_feat, fc)
        assert drift_df["drift_flag"].sum() > 0


# ── API schema tests ──────────────────────────────────────────────────────────

class TestAPISchemas:
    def test_cycle_features_valid(self):
        from serving.serve import CycleFeatures
        req = CycleFeatures(features=[1.0] * 40, cycle_id="test", station="biw_welding")
        assert req.station == "biw_welding"

    def test_batch_cycles_valid(self):
        from serving.serve import BatchCycles, CycleFeatures
        batch = BatchCycles(cycles=[
            CycleFeatures(features=[1.0] * 10, cycle_id="a"),
            CycleFeatures(features=[2.0] * 10, cycle_id="b"),
        ])
        assert len(batch.cycles) == 2
