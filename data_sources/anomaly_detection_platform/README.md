# Industrial Anomaly Detection and Alerting Platform

![CI](https://github.com/zachhersick/anomaly-detection/actions/workflows/ci.yml/badge.svg)

End-to-end ML systems project for simulated industrial anomaly detection.

The system generates synthetic machine sensor data, engineers time-series features, trains an anomaly model, creates alerts, groups alerts into events, stores results in SQLite, exposes results through FastAPI, serves saved model artifacts through `/predict`, and displays run results in Streamlit.

```text
Data generation -> Features -> Model -> Alerts -> Events -> SQLite -> FastAPI -> Streamlit
```

---

## Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the ML pipeline

```bash
python run_pipeline.py
```

### 3. Load outputs into SQLite

```bash
python load_to_db.py
```

### 4. Start the API

```bash
python -m uvicorn api:app --reload
```

API docs:

```text
http://127.0.0.1:8000/docs
```

### 5. Start the dashboard

In a second terminal:

```bash
streamlit run dashboard.py
```

### 6. Run tests

```bash
python -m pytest
```

---

## What This Project Demonstrates

```text
synthetic data generation
time-series feature engineering
model training and thresholding
model artifact saving
basic model serving
alert generation
alert event grouping
SQLite persistence
FastAPI API design
Streamlit dashboard
testing and CI
```

---

## Tech Stack

```text
Python
pandas
NumPy
scikit-learn
joblib
SQLite
FastAPI
Pydantic
Uvicorn
Streamlit
pytest
GitHub Actions
```

---

## Repository Structure

```text
generator.py              Generate synthetic machine sensor data
features.py               Build temporal/statistical features
model.py                  Train model, write predictions, save artifacts
model_ablation.py         Compare feature-group ablation runs
evaluate.py               Print model evaluation report
alerts.py                 Create row-level alerts
alert_events.py           Group row alerts into events
run_pipeline.py           Run the full pipeline

config.py                 Shared project configuration
db.py                     SQLite schema, connection, and indexes
load_to_db.py             Load generated outputs into SQLite
db_queries.py             SQLite query helpers

api.py                    FastAPI backend
schemas.py                Pydantic request/response models
model_serving.py          Load model artifacts and run inference
dashboard.py              Streamlit dashboard

tests/                    Pytest suite
.github/workflows/ci.yml  GitHub Actions CI
requirements.txt          Dependencies
```

---

## Data and Features

The synthetic dataset simulates multiple industrial machines with these sensors:

```text
temperature
pressure
vibration
flow_rate
voltage
current
```

Supported anomaly types:

```text
spike
drop
drift
oscillation
stuck_sensor
impossible_value
```

Feature engineering creates time-series features such as:

```text
rolling mean/std
rolling min/max/range
delta and absolute delta
z-scores
rolling slope
lag autocorrelation
zero-crossing counts
short-vs-long rolling differences
```

Main generated feature output:

```text
outputs/sensor_data_features.csv
```

---

## Model Training

`model.py` trains a Random Forest classifier and writes:

```text
outputs/predictions.csv
outputs/feature_importance.csv
outputs/threshold_results.csv
```

The default anomaly threshold is configured in `config.py`:

```text
0.35
```

---

## Model Artifacts

`model.py` saves deployable model artifacts:

```text
artifacts/model.joblib
artifacts/feature_columns.json
artifacts/model_metadata.json
```

These are generated locally and ignored by Git.

```text
model.joblib              trained Random Forest model
feature_columns.json      exact training feature order
model_metadata.json       threshold, model type, feature count, training rows, timestamp
```

Regenerate artifacts:

```bash
python model.py
```

---

## Model Ablation

`model_ablation.py` compares the final model against versions with selected feature groups removed.

Example runs:

```text
final_model
without_lag_autocorr
without_zero_cross
without_center_balance
without_trend_ratio
```

Run:

```bash
python model_ablation.py
```

Main output:

```text
outputs/ablation_results.csv
```

---

## Alerting and Events

`alerts.py` converts predictions into row-level alerts with:

```text
machine_id
step
sensor
sensor_value
prediction
anomaly_score
severity
alert_type
reason
status
anomaly_type
```

Severity levels:

```text
INFO
WARNING
CRITICAL
```

`alert_events.py` groups consecutive row alerts into higher-level events based on:

```text
machine_id
sensor
anomaly_type
step gap
```

Main outputs:

```text
outputs/alerts.csv
outputs/alert_events.csv
```

---

## SQLite Storage

SQLite tables:

```text
pipeline_runs
sensor_readings
model_predictions
row_alerts
alert_events
```

Database file:

```text
anomaly_detection.db
```

Load generated outputs:

```bash
python load_to_db.py
```

Each load creates a new `run_id`, allowing multiple runs to be stored and compared.

---

## FastAPI Backend

Start API:

```bash
python -m uvicorn api:app --reload
```

Main endpoints:

```text
GET  /health
GET  /runs
GET  /runs/latest
GET  /runs/{run_id}/summary
GET  /runs/{run_id}/events
GET  /runs/{run_id}/events/critical
GET  /runs/{run_id}/events/{event_id}/alerts
GET  /runs/{run_id}/events/anomaly-type-distribution
GET  /runs/{run_id}/events/sensor-distribution
GET  /runs/{run_id}/events/severity-distribution
GET  /runs/{run_id}/predictions
GET  /runs/{run_id}/machines/{machine_id}/readings
GET  /dashboard/runs/{run_id}
POST /predict
```

The API supports:

```text
run validation
event validation
filtering
pagination
Pydantic response models
dashboard-ready aggregate responses
basic model inference
```

---

## Model Inference

`POST /predict` serves the saved model artifact.

Request:

```json
{
  "features": {
    "feature_a": 1.0,
    "feature_b": 0.5
  }
}
```

Response:

```json
{
  "prediction": 0,
  "anomaly_score": 0.0033333333333333335,
  "threshold": 0.35,
  "is_anomaly": false
}
```

The endpoint expects engineered features, not raw sensor readings. Missing expected features are filled with `0`, then inputs are aligned to the saved feature order.

---

## Streamlit Dashboard

Run after starting the API:

```bash
streamlit run dashboard.py
```

The dashboard displays:

```text
run summary metrics
anomaly type distribution
sensor distribution
severity distribution
top critical events
raw API response
```

Architecture:

```text
Streamlit -> FastAPI -> SQLite
```

The dashboard does not read CSV files or SQLite directly.

---

## Testing

Run all tests:

```bash
python -m pytest
```

Run selected tests:

```bash
python -m pytest tests/test_api.py
python -m pytest tests/test_db.py
```

The test suite covers:

```text
data generation
feature engineering
model helpers
alert creation
event grouping
database schema/loading/indexes
query helpers
API routes
filtering
pagination
validation
summary/distribution/dashboard endpoints
model inference endpoint
pipeline orchestration
```

---

## Generated Files

Generated files should not be committed.

Ignored/generated files include:

```text
outputs/
artifacts/
*.csv
*.db
*.sqlite
*.sqlite3
metrics.txt
.pytest_cache/
__pycache__/
```

Common outputs:

```text
outputs/sensor_data_raw.csv
outputs/sensor_data_features.csv
outputs/predictions.csv
outputs/feature_importance.csv
outputs/threshold_results.csv
outputs/alerts.csv
outputs/alert_events.csv
outputs/ablation_results.csv
anomaly_detection.db
artifacts/model.joblib
artifacts/feature_columns.json
artifacts/model_metadata.json
```

---

## Current Status

Completed:

```text
synthetic data generator
feature engineering
Random Forest model
threshold sweep
model ablation
alert generation
alert event grouping
SQLite persistence
FastAPI backend
Streamlit dashboard
model artifact saving
POST /predict endpoint
pytest suite
GitHub Actions CI
```

Remaining:

```text
Dockerize API and dashboard
add Docker deployment docs
add screenshots
final polish
```