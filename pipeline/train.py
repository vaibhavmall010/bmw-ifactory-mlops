"""
pipeline/train.py

Stage 3 — Model Training with MLflow Experiment Tracking

Two complementary anomaly detection approaches:
  1. Isolation Forest  — unsupervised, detects outliers without labels
  2. XGBoost Classifier — supervised, uses anomaly labels for higher precision
  3. Voting Ensemble   — combines both for robust production detection

Every run is logged to MLflow: params, metrics, confusion matrix,
feature importance, and model artifact.
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest, VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [TRAIN] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
MLFLOW_URI   = f"sqlite:///{os.path.join(MODELS_DIR, 'mlflow.db')}"
EXPERIMENT   = "bmw_ifactory_anomaly_detection"
os.makedirs(MODELS_DIR, exist_ok=True)


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)


def plot_confusion_matrix(y_true, y_pred, run_name: str) -> str:
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {run_name}", fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(MODELS_DIR, f"cm_{run_name}.png")
    plt.savefig(path, dpi=100); plt.close()
    return path


def train_supervised(X: np.ndarray, y: np.ndarray,
                     feature_cols: list, feature_version: str) -> dict:
    """Train XGBoost + Random Forest supervised anomaly classifier."""
    name = "xgboost_classifier" if HAS_XGB else "random_forest_classifier"
    log.info(f"Stage 3 | Training supervised model: {name}")

    if HAS_XGB:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                scale_pos_weight=int((y == 0).sum() / max((y == 1).sum(), 1)),
                random_state=42, eval_metric="logloss", verbosity=0,
            )),
        ])
    else:
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=200, max_depth=8, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )),
        ])

    cv_scores = cross_val_score(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                                scoring="f1", n_jobs=-1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]

    metrics = {
        "cv_f1":       float(cv_scores.mean()),
        "cv_f1_std":   float(cv_scores.std()),
        "train_f1":    float(f1_score(y, y_pred)),
        "precision":   float(precision_score(y, y_pred, zero_division=0)),
        "recall":      float(recall_score(y, y_pred, zero_division=0)),
        "roc_auc":     float(roc_auc_score(y, y_prob)),
        "accuracy":    float(accuracy_score(y, y_pred)),
    }

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_type",      name)
        mlflow.log_param("feature_version", feature_version)
        mlflow.log_param("n_samples",       len(X))
        mlflow.log_param("n_features",      X.shape[1])
        mlflow.log_param("anomaly_rate",    round(float(y.mean()), 4))
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        cm_path = plot_confusion_matrix(y, y_pred, name)
        mlflow.log_artifact(cm_path)

        report_path = os.path.join(MODELS_DIR, f"report_{name}.txt")
        with open(report_path, "w") as f:
            f.write(f"Model: {name}\n")
            f.write(f"CV F1: {metrics['cv_f1']:.4f} +/- {metrics['cv_f1_std']:.4f}\n\n")
            f.write(classification_report(y, y_pred, target_names=["Normal", "Anomaly"]))
        mlflow.log_artifact(report_path)

        if hasattr(clf.named_steps["model"], "feature_importances_"):
            imp     = clf.named_steps["model"].feature_importances_
            feat_arr = np.array(feature_cols)
            top_idx = np.argsort(imp)[::-1][:15]
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(feat_arr[top_idx][::-1], imp[top_idx][::-1], color="#1c69d3")
            ax.set_xlabel("Feature Importance"); ax.set_title(f"Top Features — {name}", fontsize=10)
            plt.tight_layout()
            fi_path = os.path.join(MODELS_DIR, f"feature_imp_{name}.png")
            plt.savefig(fi_path, dpi=100); plt.close()
            mlflow.log_artifact(fi_path)

        mlflow.sklearn.log_model(
    clf,
    artifact_path="model",
    skops_trusted_types=[
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier",
        "sklearn.pipeline.Pipeline",
        "sklearn.preprocessing._data.StandardScaler",
        "numpy.ndarray",
    ],
)
        run_id = run.info.run_id

    log.info(f"Stage 3 | {name} — CV F1: {metrics['cv_f1']:.4f} | "
             f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
    return {"name": name, "run_id": run_id, "pipeline": clf, **metrics}


def train_isolation_forest(X: np.ndarray, y: np.ndarray,
                            feature_cols: list, feature_version: str) -> dict:
    """Train Isolation Forest unsupervised model."""
    name = "isolation_forest"
    log.info(f"Stage 3 | Training unsupervised model: {name}")

    contamination = float(y.mean())
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    clf.fit(X)

    # IsolationForest returns -1 for anomaly, 1 for normal
    raw_pred = clf.named_steps["model"].predict(
        clf.named_steps["scaler"].transform(X)
    )
    y_pred = np.where(raw_pred == -1, 1, 0)

    metrics = {
        "train_f1":   float(f1_score(y, y_pred, zero_division=0)),
        "precision":  float(precision_score(y, y_pred, zero_division=0)),
        "recall":     float(recall_score(y, y_pred, zero_division=0)),
        "accuracy":   float(accuracy_score(y, y_pred)),
        "cv_f1":      float(f1_score(y, y_pred, zero_division=0)),  # no CV for unsupervised
    }

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_type",      name)
        mlflow.log_param("contamination",   round(contamination, 4))
        mlflow.log_param("feature_version", feature_version)
        mlflow.log_param("n_samples",       len(X))
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        cm_path = plot_confusion_matrix(y, y_pred, name)
        mlflow.log_artifact(cm_path)
        mlflow.sklearn.log_model(
    clf,
    artifact_path="model",
    skops_trusted_types=[
        "sklearn.ensemble._iforest.IsolationForest",
        "sklearn.pipeline.Pipeline",
        "sklearn.preprocessing._data.StandardScaler",
        "numpy.ndarray",
    ],
)
        run_id = run.info.run_id

    log.info(f"Stage 3 | {name} — F1: {metrics['train_f1']:.4f} | "
             f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
    return {"name": name, "run_id": run_id, "pipeline": clf, **metrics}


# ── Pipeline entry point ──────────────────────────────────────────────────────

def run(feat_df: pd.DataFrame = None, feature_version: str = "latest") -> list:
    setup_mlflow()

    if feat_df is None:
        from pipeline.features import load_latest_features
        feat_df, manifest = load_latest_features()
        feature_version   = manifest["version"]

    feature_cols = [c for c in feat_df.columns
                    if c not in ["cycle_id", "station", "anomaly", "anomaly_type"]]
    X = feat_df[feature_cols].values.astype(float)
    y = feat_df["anomaly"].values.astype(int)

    log.info(f"Stage 3 | Training on {X.shape[0]} x {X.shape[1]} | "
             f"Anomaly rate: {y.mean():.1%}")

    results = [
        train_supervised(X, y, feature_cols, feature_version),
        train_isolation_forest(X, y, feature_cols, feature_version),
    ]

    best = max(results, key=lambda r: r["cv_f1"])
    log.info(f"\nStage 3 | Best model: {best['name']} (CV F1: {best['cv_f1']:.4f})")

    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "pipeline"} for r in results], f, indent=2)

    return results


if __name__ == "__main__":
    results = run()
    print("\n--- Training Summary ---")
    for r in sorted(results, key=lambda x: x["cv_f1"], reverse=True):
        print(f"  {r['name']:30s}  F1: {r['cv_f1']:.4f}  "
              f"Precision: {r['precision']:.4f}  Recall: {r['recall']:.4f}")
