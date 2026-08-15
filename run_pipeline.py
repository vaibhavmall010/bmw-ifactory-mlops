"""
run_pipeline.py — One-command full pipeline execution

Stages:
  1. Data ingestion and validation
  2. Feature engineering and feature store
  3. Model training (MLflow)
  4. Model evaluation and registry
  6. Drift monitoring (Azure ML optional)
"""

import os
import sys
import time
import logging

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_ROOT, "models", "pipeline.log"), mode="a"),
    ],
)
log = logging.getLogger(__name__)


def run():
    log.info("=" * 65)
    log.info("BMW iFACTORY — Anomaly Detection MLOps Pipeline")
    log.info("=" * 65)
    t_total = time.time()

    log.info("\n>>> STAGE 1: Data Ingestion and Validation")
    t = time.time()
    from pipeline.ingest import run as ingest_run
    raw_df, validation = ingest_run(n_cycles=500, seed=42, tag="pipeline_run")
    if not validation.passed:
        log.error("Pipeline aborted — data validation failed.")
        sys.exit(1)
    log.info(f"    Complete ({time.time()-t:.1f}s)")

    log.info("\n>>> STAGE 2: Feature Engineering")
    t = time.time()
    from pipeline.features import run as features_run
    feat_df, feat_path = features_run(raw_df=raw_df, tag="pipeline_run")
    log.info(f"    Complete ({time.time()-t:.1f}s)")

    log.info("\n>>> STAGE 3: Model Training")
    t = time.time()
    from pipeline.train import run as train_run
    results = train_run(feat_df=feat_df)
    best = max(results, key=lambda r: r["cv_f1"])
    log.info(f"    Best model: {best['name']} (F1: {best['cv_f1']:.4f})")
    log.info(f"    Complete ({time.time()-t:.1f}s)")

    log.info("\n>>> STAGE 4: Model Evaluation and Registry")
    t = time.time()
    from pipeline.evaluate import run as evaluate_run
    version = evaluate_run(feat_df=feat_df)
    if version is None:
        log.error("Pipeline aborted — quality gate failed.")
        sys.exit(1)
    log.info(f"    Production model version: {version}")
    log.info(f"    Complete ({time.time()-t:.1f}s)")

    log.info("\n>>> STAGE 6: Drift Monitoring")
    t = time.time()
    from monitoring.monitor import run as monitor_run
    drift_summary = monitor_run(ref_df=feat_df)
    log.info(f"    Drift rate: {drift_summary['drift_rate']:.1%}")
    log.info(f"    Retrain: {drift_summary['retrain_recommended']}")
    log.info(f"    Complete ({time.time()-t:.1f}s)")

    elapsed = time.time() - t_total
    log.info("\n" + "=" * 65)
    log.info(f"Pipeline complete in {elapsed:.1f}s")
    log.info(f"  Production model : v{version} ({best['name']})")
    log.info(f"  F1               : {best['cv_f1']:.4f}")
    log.info(f"  Recall           : {best.get('recall', 0):.4f}")
    log.info(f"  Drift status     : {'ALERT' if drift_summary['retrain_recommended'] else 'Stable'}")
    log.info("=" * 65)
    log.info("\nNext steps:")
    log.info("  Dashboard  : streamlit run dashboard/app.py")
    log.info("  API        : uvicorn serving.serve:app --port 8000")
    log.info("  MLflow UI  : mlflow ui --backend-store-uri "
             f"sqlite:///{os.path.join(PROJECT_ROOT, 'models', 'mlflow.db')}")


if __name__ == "__main__":
    run()
