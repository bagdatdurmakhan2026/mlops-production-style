import json
import os
import pickle
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor, __version__ as xgboost_version

from src.data_validation import verify_dataset_checksum

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CSV_NAME = "bangalore-restaurants.csv"
CHECKSUMS_PATH = DATA_DIR / "checksums.sha256"
MODEL_PATH = ROOT_DIR / "artifacts" / "models" / "model.pkl"
METRICS_PATH = ROOT_DIR / "artifacts" / "metrics" / "metrics.json"
RANDOM_STATE = 42
HOLDOUT_FRACTION = 0.2
CV_FOLDS = 3
SEARCH_ITERATIONS = 4


def load_data() -> tuple[pd.DataFrame, str]:
    csv_path = DATA_DIR / CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Датасет не найден по пути: {csv_path}")

    dataset_checksum = verify_dataset_checksum(csv_path, CHECKSUMS_PATH)
    df = pd.read_csv(csv_path)
    print("Доступные колонки в датасете:", df.columns.tolist())
    return df, dataset_checksum


def _clean_dataset(df):
    required = {"rate", "votes", "approx_cost(for two people)"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"В датасете отсутствуют обязательные колонки: {sorted(missing)}")

    target = pd.to_numeric(
        df["rate"].astype(str).str.replace("/5", "", regex=False).str.strip(),
        errors="coerce",
    )
    features = pd.DataFrame(
        {
            "votes": pd.to_numeric(df["votes"], errors="coerce"),
            "approx_cost": pd.to_numeric(
                df["approx_cost(for two people)"]
                .astype(str)
                .str.replace(",", "", regex=False),
                errors="coerce",
            ),
        },
        index=df.index,
    )

    for column in ("online_order", "book_table"):
        if column in df.columns:
            binary = df[column].astype(str).str.strip().str.lower().map(
                {"yes": 1.0, "no": 0.0}
            )
            features[column] = binary

    parsed = (
        features.assign(rate=target)
        .replace([np.inf, -np.inf], np.nan)
    )
    complete = parsed.notna().all(axis=1)
    valid_domain = (
        parsed["rate"].between(0, 5)
        & parsed["votes"].ge(0)
        & parsed["approx_cost"].ge(0)
    )
    clean = parsed.loc[complete & valid_domain]
    if clean.empty:
        raise ValueError("После очистки данных не осталось строк для обучения.")

    X = clean.drop(columns="rate")
    y = clean["rate"]
    validation_report = {
        "input_rows": int(len(df)),
        "valid_rows": int(len(clean)),
        "dropped_incomplete_rows": int((~complete).sum()),
        "dropped_domain_invalid_rows": int((complete & ~valid_domain).sum()),
        "rating_min": float(y.min()),
        "rating_max": float(y.max()),
        "negative_vote_rows": int((parsed["votes"] < 0).sum()),
        "negative_cost_rows": int((parsed["approx_cost"] < 0).sum()),
    }
    return X, y, validation_report


def _feature_diagnostics(X_train, y_train):
    scores = mutual_info_regression(X_train, y_train, random_state=42)
    importance = pd.Series(scores, index=X_train.columns).sort_values(ascending=False)

    print("Корреляции признаков:")
    print(X_train.corr().round(3).to_string())
    print("Взаимная информация признаков:")
    print(importance.round(4).to_string())
    return importance


def _split_and_clean_dataset(df):
    train_rows, holdout_rows = train_test_split(
        df, test_size=HOLDOUT_FRACTION, random_state=RANDOM_STATE
    )
    X_train, y_train, train_report = _clean_dataset(train_rows)
    X_holdout, y_holdout, holdout_report = _clean_dataset(holdout_rows)
    return (
        X_train,
        X_holdout,
        y_train,
        y_holdout,
        {"train": train_report, "holdout": holdout_report},
    )


def _tune_models(X_train, y_train):
    candidates = {
        "HistGradientBoostingRegressor": (
            HistGradientBoostingRegressor(random_state=42),
            {
                "max_iter": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_leaf_nodes": [15, 31],
                "l2_regularization": [0.0, 1.0],
            },
        ),
        "XGBRegressor": (
            XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                tree_method="hist",
                n_jobs=1,
                random_state=42,
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        ),
    }

    results = {}
    for name, (estimator, params) in candidates.items():
        print(f"Подбор параметров: {name}...")
        search = RandomizedSearchCV(
            estimator,
            params,
            n_iter=SEARCH_ITERATIONS,
            scoring="neg_root_mean_squared_error",
            cv=CV_FOLDS,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        results[name] = search
        print(f"Лучший CV RMSE ({name}): {-search.best_score_:.4f}")
    return results


def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Загрузка датасета и фиксация holdout...")
    raw_data, dataset_checksum = load_data()
    (
        X_train,
        X_test,
        y_train,
        y_test,
        validation_report,
    ) = _split_and_clean_dataset(raw_data)
    feature_schema = list(X_train.columns)
    mutual_info = _feature_diagnostics(X_train, y_train)
    X_train = X_train[feature_schema]
    X_test = X_test[feature_schema]

    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    baseline_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, baseline_predictions))),
        "mae": float(mean_absolute_error(y_test, baseline_predictions)),
        "r2_score": float(r2_score(y_test, baseline_predictions)),
    }

    searches = _tune_models(X_train, y_train)
    candidate_metrics = {
        name: {
            "cv_rmse": float(-search.best_score_),
            "best_params": search.best_params_,
        }
        for name, search in searches.items()
    }

    best_name = min(
        candidate_metrics,
        key=lambda name: candidate_metrics[name]["cv_rmse"],
    )
    model = searches[best_name].best_estimator_
    holdout_predictions = model.predict(X_test)
    holdout_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, holdout_predictions))),
        "mae": float(mean_absolute_error(y_test, holdout_predictions)),
        "r2_score": float(r2_score(y_test, holdout_predictions)),
    }
    training_environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost_version,
        "mlflow": mlflow.__version__,
    }
    permutation = permutation_importance(
        model, X_test, y_test, scoring="neg_root_mean_squared_error",
        n_repeats=5, random_state=42, n_jobs=-1,
    )
    permutation_scores = pd.Series(
        permutation.importances_mean, index=feature_schema
    ).sort_values(ascending=False)

    print(f"Выбрана модель: {best_name}")
    print("Важность признаков (permutation):")
    print(permutation_scores.round(4).to_string())
    print(f"Holdout RMSE: {holdout_metrics['rmse']:.4f}")
    print(f"Holdout MAE: {holdout_metrics['mae']:.4f}")
    print(f"Holdout R2: {holdout_metrics['r2_score']:.4f}")

    metrics = {
        "selected_model": best_name,
        "features": feature_schema,
        "selection_metric": "3-fold CV RMSE on training split",
        "feature_selection": "none; all schema-available features are used",
        "split": {
            "strategy": "random row split before parsing and validation",
            "holdout_fraction": HOLDOUT_FRACTION,
            "random_state": RANDOM_STATE,
            "cv_strategy": "non-shuffled KFold",
            "cv_folds": CV_FOLDS,
            "search_iterations_per_model": SEARCH_ITERATIONS,
        },
        "dataset": {
            "file": CSV_NAME,
            "sha256": dataset_checksum,
            "source_columns": list(raw_data.columns),
            "source_license_status": "unverified local copy",
            "input_rows": int(len(raw_data)),
        },
        "holdout_size": int(len(y_test)),
        "training_size": int(len(y_train)),
        "data_validation": validation_report,
        "mutual_information": mutual_info.round(4).to_dict(),
        "permutation_importance": permutation_scores.round(4).to_dict(),
        "candidates": candidate_metrics,
        "model": {key: round(value, 4) for key, value in holdout_metrics.items()},
        "baseline": {key: round(value, 4) for key, value in baseline_metrics.items()},
        "training_environment": training_environment,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)
    binary_defaults = {
        feature: float(X_train[feature].mode().iloc[0])
        for feature in feature_schema
        if feature not in {"votes", "approx_cost"}
    }
    input_reference = {
        feature: {
            "lower": float(X_train[feature].quantile(0.01)),
            "upper": float(X_train[feature].quantile(0.99)),
        }
        for feature in feature_schema
    }
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(
            {
                "model": model,
                "features": feature_schema,
                "binary_defaults": binary_defaults,
                "input_reference": input_reference,
                "training_environment": training_environment,
            },
            file,
        )

    default_tracking_uri = f"sqlite:///{(ROOT_DIR / 'artifacts' / 'mlflow.db').as_posix()}"
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", default_tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    experiment_name = "restaurant-rating"
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        try:
            if tracking_uri.startswith(("http://", "https://")):
                experiment_id = client.create_experiment(experiment_name)
            else:
                experiment_id = client.create_experiment(
                    experiment_name,
                    artifact_location=(ROOT_DIR / "artifacts" / "mlflow").as_uri(),
                )
        except MlflowException:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise
            experiment_id = experiment.experiment_id
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=best_name):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_param("selection_metric", "3-fold CV RMSE")
        mlflow.log_param("features", json.dumps(feature_schema))
        mlflow.log_param("dataset_file", CSV_NAME)
        mlflow.log_param("dataset_sha256", dataset_checksum)
        mlflow.log_param("split_random_state", RANDOM_STATE)
        mlflow.log_param("holdout_fraction", HOLDOUT_FRACTION)
        mlflow.log_param("cv_strategy", "non-shuffled KFold")
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("search_iterations_per_model", SEARCH_ITERATIONS)
        mlflow.log_param("training_rows", len(y_train))
        mlflow.log_param("holdout_rows", len(y_test))
        mlflow.log_params(
            {
                f"best.{key}": str(value)
                for key, value in searches[best_name].best_params_.items()
            }
        )
        mlflow.log_metrics(
            {
                f"cv_rmse.{name}": result["cv_rmse"]
                for name, result in candidate_metrics.items()
            }
        )
        mlflow.log_metrics(
            {f"holdout.{key}": value for key, value in holdout_metrics.items()}
        )
        mlflow.log_metrics(
            {f"baseline.{key}": value for key, value in baseline_metrics.items()}
        )
        mlflow.log_dict(validation_report, "data_validation.json")
        mlflow.log_dict(metrics, "evaluation/metrics.json")
        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            serialization_format="pickle",
            registered_model_name="restaurant-rating",
        )
        if model_info.registered_model_version is None:
            raise RuntimeError("MLflow did not return the registered model version.")
        client.set_registered_model_alias(
            "restaurant-rating", "candidate", str(model_info.registered_model_version)
        )
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="deployment")
    print(f"Метрики сохранены в {METRICS_PATH}")
    print(f"Модель сохранена в {MODEL_PATH}")


if __name__ == "__main__":
    main()
