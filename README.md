# BMW iFACTORY — Production Anomaly Detection MLOps Pipeline

End-to-end MLOps pipeline for real-time assembly line anomaly detection.
Mirrors BMW's iFACTORY AI platform across Regensburg, Leipzig, and Debrecen plants.
Covers the full ML lifecycle: data ingestion, validation, feature engineering,
experiment tracking, model registry, REST serving, drift monitoring with
Azure ML integration, and a CI/CD pipeline.

---

## Industry context — BMW iFACTORY

BMW's iFACTORY strategy defines three pillars for next-generation production:
LEAN, GREEN, and DIGITAL. The DIGITAL pillar is built on AI systems that monitor
every assembly station in real time, detect process deviations before they produce
defects, and trigger corrective actions automatically.

This project mirrors the architecture of BMW's production AI platform:

| BMW Initiative | What this project mirrors |
|----------------|--------------------------|
| iFACTORY AIQX | Real-time sensor anomaly detection per station |
| iFACTORY Digital | MLflow experiment tracking and model registry |
| Azure cloud platform | Azure ML integration for drift monitoring |
| GenAI4Q Regensburg | Data-driven quality prediction pipeline |

This is directly relevant to BMW's AI Engineer roles in the
Production AI, Quality Engineering, and Data Platform teams.

---

## Architecture

```
[Stage 1] Data Ingestion + Validation
          5 BMW assembly stations simulated:
          BiW welding, paint curing, powertrain torque,
          closure assembly, end-of-line functional test
          6 sensor channels: torque, vibration, temperature,
          cycle time, current, pressure
          Schema validation, range checks, null rate checks
          |
          v
[Stage 2] Feature Engineering + Feature Store
          Per-sensor statistics, ratio features (torque/current),
          station z-score normalisation, station one-hot encoding,
          cross-cycle rolling mean and std (drift proxy)
          Versioned parquet storage with MD5 manifest
          |
          v
[Stage 3] Model Training + MLflow Tracking
          Isolation Forest — unsupervised, no labels needed
          XGBoost Classifier — supervised with anomaly labels
          Both logged to MLflow: params, metrics, confusion matrix,
          feature importance, classification report
          |
          v
[Stage 4] Model Evaluation + Registry Promotion
          Quality gate: F1 >= 0.65, Recall >= 0.60
          MLflow Model Registry: Staging -> Production
          Previous Production version archived automatically
          |
          v
[Stage 5] FastAPI Prediction Service
          /predict — single cycle anomaly classification
          /predict/batch — batch inference
          /model/info — current production model metadata
          Loads champion model from MLflow Registry at startup
          |
          v
[Stage 6] Drift Monitoring + Azure ML Integration
          PSI per feature — distribution shift detection
          KS-test per feature — statistical significance
          Retraining trigger when drift rate > 30% of features
          Azure ML workspace logging when AZURE_ML_ENABLED=true
          HTML drift report per run
          |
          v
[Stage 7] Streamlit MLOps Dashboard
          Production Health tab — station anomaly rates, sensor box plots
          Experiment History tab — MLflow run comparison
          Drift Monitor tab — gauge, report embed, Azure ML status
          Live Prediction tab — submit a cycle reading via API
          Pipeline Control tab — one-click full retraining
```

---

## Quickstart

### Local

```bash
# 1. Clone and install
git clone https://github.com/vaibhavmall010/bmw-ifactory-mlops
cd bmw-ifactory-mlops
pip install -r requirements.txt

# 2. Run full pipeline (all 6 stages, approx 2-3 minutes)
python run_pipeline.py

# 3. Launch dashboard
streamlit run dashboard/app.py

# 4. Start prediction API (separate terminal)
uvicorn serving.serve:app --host 0.0.0.0 --port 8000

# 5. Launch MLflow UI (separate terminal)
mlflow ui --backend-store-uri sqlite:///models/mlflow.db --port 5000
```

### Docker Compose (all services)

```bash
docker-compose up --build
```

| Service | URL |
|---------|-----|
| MLOps Dashboard | http://localhost:8501 |
| Prediction API + Swagger docs | http://localhost:8000/docs |
| MLflow Experiment Tracking | http://localhost:5000 |

---

## Azure ML integration

Set the following environment variables to enable Azure ML drift logging:

```bash
export AZURE_ML_ENABLED=true
export AZURE_SUBSCRIPTION_ID=your-subscription-id
export AZURE_RESOURCE_GROUP=your-resource-group
export AZURE_ML_WORKSPACE_NAME=your-workspace-name
export AZURE_TENANT_ID=your-tenant-id
```

When enabled, drift metrics are logged to the Azure ML workspace
alongside the local HTML report. This mirrors how BMW's iFACTORY
platform centralises model monitoring across all 31 production facilities.

---

## Project structure

```
bmw_ifactory_mlops/
├── pipeline/
│   ├── ingest.py        # Stage 1 — ingestion, 5-station simulator, validation
│   ├── features.py      # Stage 2 — feature engineering, feature store
│   ├── train.py         # Stage 3 — Isolation Forest + XGBoost, MLflow logging
│   └── evaluate.py      # Stage 4 — quality gate, MLflow registry promotion
├── serving/
│   └── serve.py         # Stage 5 — FastAPI prediction API
├── monitoring/
│   └── monitor.py       # Stage 6 — PSI + KS drift detection, Azure ML integration
├── dashboard/
│   └── app.py           # Stage 7 — 5-tab Streamlit MLOps dashboard
├── tests/
│   └── test_pipeline.py # pytest suite (30+ tests, no live infrastructure needed)
├── data/
│   ├── raw/             # validated sensor batches (parquet)
│   ├── feature_store/   # versioned feature matrices + manifests
│   └── drift_reports/   # HTML drift reports + JSON alerts
├── models/
│   ├── mlflow.db        # MLflow SQLite backend
│   └── registry_info.json
├── .github/workflows/
│   └── ci.yml           # GitHub Actions CI/CD (test + Docker build)
├── run_pipeline.py      # one-command full pipeline runner
├── Dockerfile
└── docker-compose.yml
```

---

## Assembly stations modelled

| Station | Key sensors | Primary anomaly risk |
|---------|-------------|----------------------|
| BiW Welding | torque_nm, current_a | torque_spike, vibration_fault |
| Paint Curing | temperature_c, current_a | thermal_runaway |
| Powertrain Torque | torque_nm, pressure_bar | torque_spike, pressure_drop |
| Closure Assembly | torque_nm, vibration_g | vibration_fault |
| EoL Functional | cycle_time_s, vibration_g | cycle_overrun |

---

## Model design

### Why both Isolation Forest and XGBoost?

Isolation Forest is unsupervised — it can detect anomalies even without
labelled fault data, which is the reality in early production ramp-up.
XGBoost is supervised — when labels are available it delivers higher
precision and recall. The pipeline trains both and selects the winner
by F1 score. In production, both run in parallel as a defence-in-depth strategy.

### Why these features?

The torque/current ratio is a direct proxy for drive efficiency — a rising
ratio under constant torque indicates increasing mechanical resistance, an
early warning of bearing wear or lubrication issues. The z-score normalisation
per station ensures that station-specific baselines do not create false positives
when a single model covers all five stations.

---

## Running tests

```bash
pytest tests/ -v
# Expected: 30+ passed, 0 failed
```

---

## Author

Vaibhav Mall
M.Sc. Digital Engineering, RWTH Aachen
Master Thesis Researcher at Fraunhofer IPT
[linkedin.com/in/mallvaibhav](https://linkedin.com/in/mallvaibhav) | [github.com/vaibhavmall010](https://github.com/vaibhavmall010)

---

## License

MIT
