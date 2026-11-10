import pandas as pd

from src.geo_clustering import GeoAnalyzer
from src.recommender import RestaurantRecommender
from src.train import _clean_dataset


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

    features, target = _clean_dataset(frame)

    assert list(features.columns) == [
        "votes", "approx_cost", "online_order", "book_table"
    ]
    assert set(features["online_order"].unique()) <= {0.0, 1.0}
    assert len(features) == len(target) > 0


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
