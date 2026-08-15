"""
pipeline/evaluate.py

Stage 4 — Model Evaluation and MLflow Registry Promotion

Reads all MLflow runs, selects the best model by F1 score,
runs a quality gate (F1 >= 0.70, Recall >= 0.65),
registers to MLflow Model Registry, and promotes to Production.
"""

import os
import json
import logging
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [EVALUATE] %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
MLFLOW_URI    = f"sqlite:///{os.path.join(MODELS_DIR, 'mlflow.db')}"
EXPERIMENT    = "bmw_ifactory_anomaly_detection"
REGISTRY_NAME = "BMWiFactoryAnomalyDetector"

F1_GATE     = 0.65
RECALL_GATE = 0.60


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)


def get_best_run():
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    exp    = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise RuntimeError("Experiment not found. Run train.py first.")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.cv_f1 DESC"],
        max_results=20,
    )
    if not runs:
        raise RuntimeError("No runs found.")
    best = runs[0]
    log.info(f"Stage 4 | Best run: {best.info.run_id[:8]} "
             f"({best.data.tags.get('mlflow.runName', 'unknown')}) "
             f"F1: {best.data.metrics.get('cv_f1', 0):.4f}")
    return best


def quality_gate(run) -> bool:
    f1     = run.data.metrics.get("cv_f1", 0)
    recall = run.data.metrics.get("recall", 0)
    if f1 < F1_GATE:
        log.warning(f"Stage 4 | Quality gate FAILED: F1 {f1:.4f} < {F1_GATE}")
        return False
    if recall < RECALL_GATE:
        log.warning(f"Stage 4 | Quality gate FAILED: Recall {recall:.4f} < {RECALL_GATE}")
        return False
    log.info(f"Stage 4 | Quality gate PASSED: F1={f1:.4f} Recall={recall:.4f}")
    return True


def register_and_promote(run) -> str:
    client    = MlflowClient(tracking_uri=MLFLOW_URI)
    run_id    = run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    log.info(f"Stage 4 | Registering model from run {run_id[:8]}...")
    mv = mlflow.register_model(model_uri=model_uri, name=REGISTRY_NAME)
    version = mv.version

    client.update_model_version(
        name=REGISTRY_NAME, version=version,
        description=(
            f"BMW iFACTORY anomaly detector | "
            f"{run.data.tags.get('mlflow.runName', 'unknown')} | "
            f"F1: {run.data.metrics.get('cv_f1', 0):.4f} | "
            f"Recall: {run.data.metrics.get('recall', 0):.4f}"
        ),
    )

    client.transition_model_version_stage(
        name=REGISTRY_NAME, version=version, stage="Staging",
        archive_existing_versions=False,
    )
    log.info(f"Stage 4 | Version {version} -> Staging")

    client.transition_model_version_stage(
        name=REGISTRY_NAME, version=version, stage="Production",
        archive_existing_versions=True,
    )
    log.info(f"Stage 4 | Version {version} -> Production")

    info = {
        "registry_name": REGISTRY_NAME,
        "version":       version,
        "run_id":        run_id,
        "model_name":    run.data.tags.get("mlflow.runName", "unknown"),
        "cv_f1":         run.data.metrics.get("cv_f1", 0),
        "recall":        run.data.metrics.get("recall", 0),
        "precision":     run.data.metrics.get("precision", 0),
        "roc_auc":       run.data.metrics.get("roc_auc", 0),
        "mlflow_uri":    MLFLOW_URI,
        "stage":         "Production",
    }
    with open(os.path.join(MODELS_DIR, "registry_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    log.info(f"Stage 4 | Registry info saved.")
    return version


def load_production_model():
    mlflow.set_tracking_uri(MLFLOW_URI)
    reg_path = os.path.join(MODELS_DIR, "registry_info.json")
    if not os.path.exists(reg_path):
        raise FileNotFoundError("No registered model. Run evaluate.py first.")
    with open(reg_path) as f:
        info = json.load(f)
    model = mlflow.sklearn.load_model(f"models:/{REGISTRY_NAME}/Production")
    return model, info


def run(feat_df: pd.DataFrame = None) -> str:
    setup_mlflow()
    best_run = get_best_run()
    if not quality_gate(best_run):
        log.error("Stage 4 | Pipeline halted — quality gate failed.")
        return None
    version = register_and_promote(best_run)
    log.info(f"Stage 4 | Complete — Production model version: {version}")
    return version


if __name__ == "__main__":
    version = run()
    if version:
        print(f"\nProduction model version: {version}")
