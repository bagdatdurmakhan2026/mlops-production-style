import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CSV_NAME = "bangalore-restaurants.csv"
MODEL_PATH = ROOT_DIR / "artifacts" / "models" / "model.pkl"
METRICS_PATH = ROOT_DIR / "artifacts" / "metrics" / "metrics.json"


def load_data():
    csv_path = DATA_DIR / CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Датасет не найден по пути: {csv_path}")

    df = pd.read_csv(csv_path)
    print("Доступные колонки в датасете:", df.columns.tolist())
    return df


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
            if binary.notna().any():
                features[column] = binary

    clean = features.assign(rate=target).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        raise ValueError("После очистки данных не осталось строк для обучения.")

    # Убираем выбросы по IQR для числовых признаков и целевого рейтинга.
    mask = pd.Series(True, index=clean.index)
    for column in ("votes", "approx_cost", "rate"):
        q1, q3 = clean[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr > 0:
            mask &= clean[column].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    clean = clean.loc[mask]

    if clean.empty:
        raise ValueError("После удаления выбросов не осталось строк для обучения.")

    X = clean.drop(columns="rate")
    y = clean["rate"]
    return X, y


def _select_features(X_train, y_train):
    scores = mutual_info_regression(X_train, y_train, random_state=42)
    importance = pd.Series(scores, index=X_train.columns).sort_values(ascending=False)
    core_features = ["votes", "approx_cost"]
    extra_features = [
        feature for feature in importance.index
        if feature not in core_features
    ][:3]
    selected = core_features + extra_features

    print("Корреляции признаков:")
    print(X_train.corr().round(3).to_string())
    print("Взаимная информация признаков:")
    print(importance.round(4).to_string())
    return selected, importance


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
            n_iter=4,
            scoring="neg_root_mean_squared_error",
            cv=3,
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
    print("Загрузка и очистка данных...")
    X, y = _clean_dataset(load_data())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    selected_features, mutual_info = _select_features(X_train, y_train)
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    baseline_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, baseline_predictions))),
        "mae": float(mean_absolute_error(y_test, baseline_predictions)),
        "r2_score": float(r2_score(y_test, baseline_predictions)),
    }

    searches = _tune_models(X_train, y_train)
    candidate_metrics = {}
    for name, search in searches.items():
        predictions = search.best_estimator_.predict(X_test)
        candidate_metrics[name] = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2_score": float(r2_score(y_test, predictions)),
            "best_params": search.best_params_,
        }

    best_name = min(candidate_metrics, key=lambda name: candidate_metrics[name]["rmse"])
    model = searches[best_name].best_estimator_
    permutation = permutation_importance(
        model, X_test, y_test, scoring="neg_root_mean_squared_error",
        n_repeats=5, random_state=42, n_jobs=-1,
    )
    permutation_scores = pd.Series(
        permutation.importances_mean, index=selected_features
    ).sort_values(ascending=False)

    print(f"Выбрана модель: {best_name}")
    print("Важность признаков (permutation):")
    print(permutation_scores.round(4).to_string())
    print(f"RMSE: {candidate_metrics[best_name]['rmse']:.4f}")
    print(f"MAE: {candidate_metrics[best_name]['mae']:.4f}")
    print(f"R2: {candidate_metrics[best_name]['r2_score']:.4f}")

    metrics = {
        "selected_model": best_name,
        "features": selected_features,
        "mutual_information": mutual_info.round(4).to_dict(),
        "permutation_importance": permutation_scores.round(4).to_dict(),
        "candidates": candidate_metrics,
        "model": {
            key: round(value, 4)
            for key, value in candidate_metrics[best_name].items()
            if key != "best_params"
        },
        "baseline": {key: round(value, 4) for key, value in baseline_metrics.items()},
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)
    binary_defaults = {
        feature: float(X[feature].mode().iloc[0])
        for feature in selected_features
        if feature not in {"votes", "approx_cost"}
    }
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(
            {
                "model": model,
                "features": selected_features,
                "binary_defaults": binary_defaults,
            },
            file,
        )
    print(f"Метрики сохранены в {METRICS_PATH}")
    print(f"Модель сохранена в {MODEL_PATH}")


if __name__ == "__main__":
    main()
