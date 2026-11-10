import hashlib
import pickle
import platform
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import sklearn

from src.geo_clustering import GeoAnalyzer
from src.recommender import RestaurantRecommender
from src.data_validation import verify_dataset_checksum
from src.train import _clean_dataset, _filter_training_outliers
from src import train as train_module
from scripts.model_registry import promote_version
from sklearn.dummy import DummyRegressor


def _restaurant_frame(size=12):
    return pd.DataFrame(
        {
            "name": [f"Restaurant {index}" for index in range(size)],
            "location": [f"Area {index % 3}" for index in range(size)],
            "rate": [f"{3.5 + (index % 5) * 0.1}/5" for index in range(size)],
            "votes": [100 + index * 10 for index in range(size)],
            "approx_cost(for two people)": [
                f"{500 + index * 50:,}" for index in range(size)
            ],
            "online_order": ["Yes" if index % 2 else "No" for index in range(size)],
            "book_table": ["No" if index % 2 else "Yes" for index in range(size)],
            "cuisines": ["Cafe, Italian" if index % 2 else "Indian" for index in range(size)],
        }
    )


def test_clean_dataset_parses_numeric_and_binary_features():
    frame = _restaurant_frame()

    features, target, report = _clean_dataset(frame)

    assert list(features.columns) == [
        "votes", "approx_cost", "online_order", "book_table"
    ]
    assert set(features["online_order"].unique()) <= {0.0, 1.0}
    assert len(features) == len(target) > 0
    assert report["input_rows"] == report["valid_rows"] == len(frame)


def test_iqr_cutoffs_are_fitted_only_on_training_rows():
    X_train = pd.DataFrame(
        {"votes": [10, 11, 12, 12, 13, 14], "approx_cost": [100, 110, 120, 120, 130, 140]}
    )
    y_train = pd.Series([3.5, 3.6, 3.7, 3.8, 3.9, 4.0])
    X_test = pd.DataFrame({"votes": [1_000_000], "approx_cost": [1_000_000]})

    filtered_X, filtered_y, bounds = _filter_training_outliers(X_train, y_train)

    assert len(filtered_X) == len(filtered_y) == len(X_train)
    assert X_test.iloc[0]["votes"] > bounds["votes"]["upper"]
    assert X_test.iloc[0]["approx_cost"] > bounds["approx_cost"]["upper"]


def test_dataset_validation_reports_invalid_rows():
    frame = _restaurant_frame()
    frame.loc[0, "rate"] = "not rated"
    frame.loc[1, "rate"] = "5.5/5"
    frame.loc[2, "votes"] = -1

    features, target, report = _clean_dataset(frame)

    assert len(features) == len(target) == len(frame) - 3
    assert report["dropped_incomplete_rows"] == 1
    assert report["dropped_domain_invalid_rows"] == 2
    assert report["negative_vote_rows"] == 1


def test_dataset_validation_rejects_missing_columns():
    with pytest.raises(ValueError, match="обязательные колонки"):
        _clean_dataset(pd.DataFrame({"rate": ["4.0/5"]}))


def test_dataset_validation_rejects_empty_clean_result():
    frame = _restaurant_frame()
    frame["rate"] = "not rated"

    with pytest.raises(ValueError, match="не осталось строк"):
        _clean_dataset(frame)


def test_dataset_validation_drops_infinite_numeric_values():
    frame = _restaurant_frame()
    frame["votes"] = frame["votes"].astype(float)
    frame.loc[0, "votes"] = float("inf")

    features, target, report = _clean_dataset(frame)

    assert len(features) == len(target) == len(frame) - 1
    assert report["dropped_incomplete_rows"] == 1
    assert np.isfinite(features.to_numpy()).all()


def test_dataset_checksum_matches_manifest_and_detects_changes(tmp_path):
    dataset_path = tmp_path / "restaurants.csv"
    dataset_path.write_bytes(b"restaurant,data\n")
    manifest_path = tmp_path / "checksums.sha256"
    manifest_path.write_text(
        f"{hashlib.sha256(dataset_path.read_bytes()).hexdigest()}  restaurants.csv\n",
        encoding="utf-8",
    )

    expected_checksum = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert verify_dataset_checksum(dataset_path, manifest_path) == expected_checksum
    dataset_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_dataset_checksum(dataset_path, manifest_path)


def test_training_reports_missing_dataset(monkeypatch, tmp_path):
    monkeypatch.setattr(train_module, "DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="Датасет не найден"):
        train_module.load_data()


def test_model_promotion_updates_artifact_and_champion_alias(tmp_path):
    source_path = tmp_path / "registered-model.pkl"
    model_path = tmp_path / "active-model.pkl"
    with source_path.open("wb") as model_file:
        pickle.dump(
            {
                "model": DummyRegressor(),
                "features": ["votes", "approx_cost"],
                "training_environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scikit-learn": sklearn.__version__,
                },
            },
            model_file,
        )

    class FakeClient:
        alias = None

        def get_model_version(self, name, version):
            assert name == "restaurant-rating"
            assert version == "3"
            return SimpleNamespace(run_id="run-id", status="READY")

        def download_artifacts(self, run_id, artifact_path):
            assert run_id == "run-id"
            assert artifact_path == "deployment/model.pkl"
            return str(source_path)

        def set_registered_model_alias(self, name, alias, version):
            self.alias = (name, alias, version)

    client = FakeClient()
    promote_version("3", client, model_path)

    assert model_path.read_bytes() == source_path.read_bytes()
    assert client.alias == ("restaurant-rating", "champion", "3")


def test_model_promotion_rejects_incompatible_artifact(tmp_path):
    source_path = tmp_path / "incompatible-model.pkl"
    model_path = tmp_path / "active-model.pkl"
    model_path.write_bytes(b"previous model")
    with source_path.open("wb") as model_file:
        pickle.dump(
            {
                "model": DummyRegressor(),
                "features": ["votes", "approx_cost"],
                "training_environment": {"scikit-learn": "0.0"},
            },
            model_file,
        )

    class FakeClient:
        alias_updated = False

        def get_model_version(self, name, version):
            return SimpleNamespace(run_id="run-id", status="READY")

        def download_artifacts(self, run_id, artifact_path):
            return str(source_path)

        def set_registered_model_alias(self, name, alias, version):
            self.alias_updated = True

    client = FakeClient()
    with pytest.raises(RuntimeError, match="was trained with scikit-learn"):
        promote_version("3", client, model_path)

    assert model_path.read_bytes() == b"previous model"
    assert not client.alias_updated


def test_recommender_returns_neighbors_and_clusters():
    recommender = RestaurantRecommender.__new__(RestaurantRecommender)
    recommender.df = _restaurant_frame()
    recommender.preprocess()

    recommendations = recommender.recommend(0, top_k=3)
    recommender.fit_kmeans(n_clusters=2)

    assert len(recommendations) == 3
    assert "similarity_score" in recommendations
    assert recommender.df["cluster"].nunique() == 2


def test_geo_module_generates_map_and_correlation_heatmap(tmp_path):
    dataset_path = tmp_path / "restaurants.csv"
    _restaurant_frame().to_csv(dataset_path, index=False)
    analyzer = GeoAnalyzer(dataset_path)

    analyzer.preprocess(
        synthetic_center=(51.1694, 71.4491),
        center_name="Astana",
        synthetic_radius_km=12,
    )
    analyzer.run_dbscan(min_samples=2)
    analyzer.run_kmeans(n_clusters=2)
    map_path = analyzer.generate_map(tmp_path / "astana-map.html")
    heatmap_path = analyzer.generate_correlation_heatmap(
        tmp_path / "correlations.html"
    )

    map_html = map_path.read_text(encoding="utf-8")
    assert "synthetic coordinates around Astana" in map_html
    assert "server.arcgisonline.com" in map_html
    assert "<table>" in heatmap_path.read_text(encoding="utf-8")
    assert analyzer.df["latitude"].between(51.0, 51.4).all()
