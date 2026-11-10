# Restaurant Analytics and MLOps Demo

A Python project for restaurant rating regression, restaurant recommendations,
cuisine classification, and geospatial clustering. It includes a FastAPI
prediction service, MLflow tracking and registry, Prometheus/Grafana monitoring,
automated tests, CI, and a local Docker Compose stack.

> **Data note:** source CSVs are intentionally excluded from Git and Docker.
> Place the datasets listed below in `data/` before running data-dependent jobs.
> The geospatial demo can generate synthetic coordinates; those points are not
> real restaurant locations.

## Requirements

- Python 3.11
- The project CSV files in `data/`
- Docker (optional)

Install project dependencies from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The Docker image installs only the smaller API runtime dependency set from
`requirements-api.txt`; training, mapping, and test dependencies stay out of
the production image.

Required local datasets:

```text
data/
├── bangalore-restaurants.csv
└── zomato_astana_dirty_full.csv
```

The Bangalore dataset is used by training, recommendation, and cuisine
classification scripts. The geospatial script processes both datasets when
present. Their exact development-copy SHA-256 values are tracked in
`data/checksums.sha256`; training verifies the Bangalore file before reading it.
The source and license for these local copies remain unverified, so this does
not make the datasets redistributable or independently downloadable.

## Project Layout

```text
api/app.py                         FastAPI inference service
docker-compose.yml                 Local MLflow/API/Prometheus/Grafana stack
artifacts/
├── metrics/metrics.json           Latest regression evaluation
└── models/model.pkl               Trained model artifact
data/                              Local datasets (not committed)
data/README.md                     Dataset naming, schema, and provenance notes
data/checksums.sha256              SHA-256 for the exact development CSV files
monitoring/                        Prometheus rules and Grafana dashboard config
reports/maps/                      Locally generated city maps and heatmaps
src/train.py                       Rating training, tuning, and evaluation
src/data_validation.py             Dataset checksum verification
scripts/model_registry.py          List, promote, and roll back model versions
src/recommender.py                 Content recommendations and K-Means
src/cuisine_classifier.py          Cuisine classification experiments
src/geo_clustering.py              City maps and geospatial clustering
src/eda.py                         Dataset inspection helper
tests/                             API and ML-module tests
Dockerfile                         Non-root API container
Dockerfile.training                Training and MLflow server image
```

## Run the Project

### Train rating models

```powershell
python -m src.train
```

The script first splits the raw rows into training and holdout partitions, then
parses and validates each partition independently. No IQR filtering or
target-based feature selection is performed before cross-validation: every
schema-available feature is used, and mutual information is recorded only as a
training-set diagnostic. Hyperparameters and the winning model are chosen by
3-fold cross-validation on training rows, never by holdout performance. The
holdout is reserved for final RMSE, MAE, R², and permutation-importance
reporting. Metrics record the local dataset SHA-256, feature schema, split
seed, validation counts, and CV configuration; the hash identifies the exact
file but does not establish its source or license. The model and input schema
are written to `artifacts/models/model.pkl`; metrics and selected parameters
are written to `artifacts/metrics/metrics.json`.

Each successful training run is also tracked in a local MLflow experiment and
registered as a new version of `restaurant-rating`. The default SQLite tracking
database and run artifacts are stored under `artifacts/` and ignored by Git.
View runs with:

```powershell
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
```

Set `MLFLOW_TRACKING_URI` to use a shared tracking server instead.

Check the local dataset before training:

```powershell
python -m scripts.verify_data
```

Current checked-in/local results are dataset-dependent. Regression should be
reported with RMSE, MAE, and R²; an “82% accuracy” claim is not established by
these regression metrics.

### Start the API

```powershell
uvicorn api.app:app --reload
```

- Swagger UI: <http://127.0.0.1:8000/docs>
- Liveness: <http://127.0.0.1:8000/livez>
- Readiness: <http://127.0.0.1:8000/readyz>
- Compatibility health check: <http://127.0.0.1:8000/healthz>
- Prediction: `POST /predict`

Example request:

```json
{
  "features": [1200, 1500]
}
```

The two values are votes and approximate cost for two people. Any additional
binary model inputs use their saved training defaults when omitted.

### Recommendations and cuisine classification

```powershell
python src\recommender.py
python src\cuisine_classifier.py
```

The recommender combines numeric and categorical restaurant attributes with
cosine similarity and can group restaurants with K-Means. Collaborative
filtering needs a separate user-rating table (`user_id`, `restaurant_id`,
`rating`); aggregate restaurant data alone is insufficient.

The cuisine classifier compares HistGradientBoosting, XGBoost, and an MLP, and
reports accuracy, macro-F1, and per-class metrics. Its observed quality is
limited by class imbalance and the available input features.

### Generate city maps

```powershell
python src\geo_clustering.py
```

Outputs:

- `reports/maps/bangalore-restaurants.html`
- `reports/maps/bangalore-restaurants-correlations.html`
- `reports/maps/astana-restaurants.html`
- `reports/maps/astana-restaurants-correlations.html`

Map HTML files are generated locally and ignored by Git; run the command above
to recreate them. Open the map files in a browser. Real latitude/longitude
columns are used when available. Otherwise, reproducible demonstration
coordinates are spread over a 12 km radius around the configured city center.
Maps are visibly labeled as synthetic and must not be used as evidence of
actual restaurant locations or density.

### Tests and CI

```powershell
python -m pytest -q
python -m compileall -q api src scripts tests
```

GitHub Actions runs compilation and unit/API tests, validates the Compose and
Prometheus configuration, builds the stack, then smoke-tests the API,
Prometheus, Grafana, and MLflow. The workflow does not train a model because
the large data files are intentionally not stored in Git.

### Run the local MLOps stack

```powershell
docker compose up --build -d
```

Open the local services:

- API and Swagger: <http://127.0.0.1:8000/docs>
- MLflow experiments and Registry: <http://127.0.0.1:5000>
- Prometheus targets and alerts: <http://127.0.0.1:9090>
- Grafana dashboard: <http://127.0.0.1:3000> (`admin` / `local-demo-only` by
  default; override with `GRAFANA_ADMIN_PASSWORD` in a local `.env` file)

The stack binds ports to localhost, persists MLflow/Prometheus/Grafana state in
Docker named volumes, and starts the inference container as a non-root user.
Change the demo Grafana password before sharing the stack. Do not expose this
unauthenticated MLflow demo server directly to the public internet.

To train using your local data and register a new version:

```powershell
docker compose --profile tools run --build --rm trainer
docker compose --profile tools run --rm trainer python scripts/model_registry.py list
```

Training places the new Registry version under the `candidate` alias. To deploy
or roll back to a version number shown by `list`:

```powershell
docker compose --profile tools run --rm trainer python scripts/model_registry.py promote 1
docker compose restart api
```

Promotion validates that the model artifact matches the runtime before
replacing the active file and moving the Registry `champion` alias. Restarting
the API loads that artifact. Repeating the command with an older version number
rolls back the local demo. This is an explicit single-host promotion workflow,
not an automated cloud deployment or zero-downtime rollout.

Prometheus scrapes prediction request, latency, and out-of-training-range
counters. The Grafana dashboard visualizes them; Prometheus also defines alerts
for API unavailability and out-of-range input. The current drift signal is a
per-feature 1st-to-99th percentile range heuristic, not a statistical
distribution-drift test or an automatic retraining trigger.

The API fails startup if the model artifact is missing or invalid rather than
starting in a degraded state that only fails on prediction requests. It exposes
separate `/livez` and `/readyz` checks and validates finite, non-negative input
values. Its process-local counters reset on restart; Prometheus persists scraped
samples in the Compose demo. Model artifacts store their training versions of
scikit-learn and NumPy; API startup rejects incompatible artifacts rather than
silently loading an unsupported pickle.

Stop the stack without deleting experiment or dashboard history:

```powershell
docker compose down
```

`docker compose down -v` also deletes the local named volumes and their
persisted MLflow, Prometheus, and Grafana state.
