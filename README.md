# Restaurant Analytics and MLOps Demo

A Python project for restaurant rating regression, restaurant recommendations,
cuisine classification, and geospatial clustering. It includes a FastAPI
prediction service, reproducible evaluation scripts, automated tests, CI, and a
Docker image.

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
present.

## Project Layout

```text
api/app.py                         FastAPI inference service
artifacts/
├── metrics/metrics.json           Latest regression evaluation
└── models/model.pkl               Trained model artifact
data/                              Local datasets (not committed)
reports/maps/                      Generated city maps and heatmaps
src/train.py                       Rating training, tuning, and evaluation
src/recommender.py                 Content recommendations and K-Means
src/cuisine_classifier.py          Cuisine classification experiments
src/geo_clustering.py              City maps and geospatial clustering
src/eda.py                         Dataset inspection helper
tests/                             API and ML-module tests
Dockerfile                         Non-root API container
```

## Run the Project

### Train rating models

```powershell
python src\train.py
```

The script cleans rating/cost fields, encodes available order options, removes
IQR outliers, compares XGBoost and HistGradientBoosting with randomized
cross-validation, and evaluates the selected model on a holdout set. It writes
the selected model and input schema to `artifacts/models/model.pkl`, and
metrics/parameters to `artifacts/metrics/metrics.json`.

Current checked-in/local results are dataset-dependent. Regression should be
reported with RMSE, MAE, and R²; an “82% accuracy” claim is not established by
these regression metrics.

### Start the API

```powershell
uvicorn api.app:app --reload
```

- Swagger UI: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/healthz>
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

Open the map files in a browser. Real latitude/longitude columns are used when
available. Otherwise, reproducible demonstration coordinates are spread over a
12 km radius around the configured city center. Maps are visibly labeled as
synthetic and must not be used as evidence of actual restaurant locations or
density.

### Tests and CI

```powershell
python -m pytest -q
python -m compileall -q api src tests
```

GitHub Actions runs compilation and tests without requiring the large local
datasets, and builds the Docker image separately.

### Docker

```powershell
docker build -t restaurant-ml-api .
docker run --rm -p 8000:8000 restaurant-ml-api
```

The container includes the API and trained model, runs as a non-root user, and
does not copy local datasets or generated HTML maps.
