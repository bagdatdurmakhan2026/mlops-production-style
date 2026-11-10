import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "bangalore-restaurants.csv"


class RestaurantRecommender:
    def __init__(self, data_path=None):
        data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        try:
            self.df = pd.read_csv(data_path, encoding="utf-8")
        except UnicodeDecodeError:
            self.df = pd.read_csv(data_path, encoding="latin-1")

    def preprocess(self):
        required = {"name", "location", "rate", "votes", "approx_cost(for two people)"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"В датасете отсутствуют колонки: {sorted(missing)}")

        self.df = self.df.drop_duplicates(subset=["name", "location"]).reset_index(drop=True)
        self.df["rate_clean"] = pd.to_numeric(
            self.df["rate"].astype(str).str.split("/").str[0].str.strip(),
            errors="coerce",
        )
        self.df["cost_clean"] = pd.to_numeric(
            self.df["approx_cost(for two people)"]
            .astype(str)
            .str.replace(",", "", regex=False),
            errors="coerce",
        )
        self.df["votes_clean"] = pd.to_numeric(self.df["votes"], errors="coerce")
        self.numeric_features = ["cost_clean", "rate_clean", "votes_clean"]
        self.categorical_features = [
            column for column in ("location", "city", "cuisines")
            if column in self.df.columns
        ]
        if "cuisines" in self.df.columns:
            self.df["cuisines_clean"] = (
                self.df["cuisines"].fillna("").astype(str).str.replace(",", " ", regex=False)
            )
            self.categorical_features = [
                "location" if column == "location" else column
                for column in self.categorical_features
            ]
            self.categorical_features = [
                "cuisines_clean" if column == "cuisines" else column
                for column in self.categorical_features
            ]

        transformers = [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                ),
                self.numeric_features,
            )
        ]
        if self.categorical_features:
            transformers.append(
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                    self.categorical_features,
                )
            )
        self.preprocessor = ColumnTransformer(transformers)
        self.feature_matrix = self.preprocessor.fit_transform(self.df)
        self.model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.model.fit(self.feature_matrix)

    def recommend(self, restaurant_idx, top_k=5, user_id=None, content_weight=0.7):
        if not hasattr(self, "feature_matrix"):
            raise RuntimeError("Вызовите preprocess() перед recommend().")
        if not 0 <= restaurant_idx < len(self.df):
            raise IndexError("restaurant_idx вне диапазона датафрейма.")
        if top_k < 1:
            raise ValueError("top_k должен быть не меньше 1.")

        candidate_count = min(top_k + 1, len(self.df))
        distances, indices = self.model.kneighbors(
            self.feature_matrix[restaurant_idx:restaurant_idx + 1],
            n_neighbors=candidate_count,
        )
        content_scores = {
            int(index): 1.0 - float(distance)
            for index, distance in zip(indices[0], distances[0])
            if index != restaurant_idx
        }

        if user_id is not None and hasattr(self, "collaborative_scores"):
            collab = self.collaborative_scores(user_id)
            all_indices = set(content_scores) | set(collab)
            scores = {
                index: content_weight * content_scores.get(index, 0.0)
                + (1 - content_weight) * collab.get(index, 0.0)
                for index in all_indices
                if index != restaurant_idx
            }
            ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
        else:
            ranked = sorted(content_scores, key=content_scores.get, reverse=True)[:top_k]

        columns = [
            column for column in (
                "name", "location", "city", "rate",
                "approx_cost(for two people)", "cuisines",
            ) if column in self.df.columns
        ]
        result = self.df.iloc[ranked][columns].copy()
        result["similarity_score"] = [content_scores.get(i, 0.0) for i in ranked]
        return result

    def fit_collaborative(self, interactions, user_col="user_id",
                          restaurant_col="restaurant_id", rating_col="rating"):
        required = {user_col, restaurant_col, rating_col}
        missing = required.difference(interactions.columns)
        if missing:
            raise ValueError(f"В таблице оценок отсутствуют колонки: {sorted(missing)}")

        id_column = "restaurant_id" if "restaurant_id" in self.df.columns else None
        if id_column is None:
            raise ValueError(
                "Для коллаборативной фильтрации датасет ресторанов должен содержать "
                "restaurant_id; используйте реальные пользовательские оценки."
            )
        item_map = {
            str(value): index for index, value in enumerate(self.df[id_column])
        }
        ratings = interactions[[user_col, restaurant_col, rating_col]].copy()
        ratings[rating_col] = pd.to_numeric(ratings[rating_col], errors="coerce")
        ratings = ratings.dropna(subset=[rating_col])
        ratings["item_index"] = ratings[restaurant_col].astype(str).map(item_map)
        ratings = ratings.dropna(subset=["item_index"])
        if ratings.empty:
            raise ValueError("Нет пользовательских оценок для ресторанов из каталога.")

        user_codes, users = pd.factorize(ratings[user_col])
        item_indices = ratings["item_index"].astype(int).to_numpy()
        self.user_ids = users
        self.user_matrix = csr_matrix(
            (ratings[rating_col].to_numpy(), (user_codes, item_indices)),
            shape=(len(users), len(self.df)),
        )
        self.user_neighbors = NearestNeighbors(metric="cosine", algorithm="brute")
        self.user_neighbors.fit(self.user_matrix)
        self.user_to_index = {user: index for index, user in enumerate(users)}
        return self

    def collaborative_scores(self, user_id, n_neighbors=20):
        if not hasattr(self, "user_neighbors"):
            raise RuntimeError("Перед рекомендациями вызовите fit_collaborative().")
        if user_id not in self.user_to_index:
            return {}

        user_index = self.user_to_index[user_id]
        query = self.user_matrix[user_index]
        count = min(n_neighbors + 1, self.user_matrix.shape[0])
        distances, neighbors = self.user_neighbors.kneighbors(query, n_neighbors=count)
        neighbor_weights = 1.0 - distances[0]
        if len(neighbors[0]) and neighbors[0][0] == user_index:
            neighbors = neighbors[0][1:]
            neighbor_weights = neighbor_weights[1:]
        else:
            neighbors = neighbors[0]
        if len(neighbors) == 0:
            return {}

        weighted_ratings = self.user_matrix[neighbors].multiply(
            neighbor_weights[:, None]
        )
        totals = np.asarray(weighted_ratings.sum(axis=0)).ravel()
        weights = np.asarray(
            self.user_matrix[neighbors].astype(bool).multiply(
                neighbor_weights[:, None]
            ).sum(axis=0)
        ).ravel()
        seen = self.user_matrix[user_index].toarray().ravel() > 0
        scores = np.divide(totals, weights, out=np.zeros_like(totals), where=weights > 0)
        scores[seen] = 0
        return {int(i): float(scores[i]) for i in np.flatnonzero(scores)}

    def fit_kmeans(self, n_clusters=5):
        if not hasattr(self, "feature_matrix"):
            raise RuntimeError("Вызовите preprocess() перед fit_kmeans().")
        if not 1 <= n_clusters <= len(self.df):
            raise ValueError("n_clusters должен быть от 1 до числа ресторанов.")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df["cluster"] = self.kmeans.fit_predict(self.feature_matrix)
        return self

    def get_cluster_sample(self, cluster_id, sample_size=5):
        if "cluster" not in self.df:
            raise RuntimeError("Сначала вызовите fit_kmeans().")
        columns = [
            column for column in ("name", "location", "city", "rate")
            if column in self.df.columns
        ]
        return self.df.loc[self.df["cluster"] == cluster_id, columns].head(sample_size)
if __name__ == "__main__":
    recommender = RestaurantRecommender()
    recommender.preprocess()
    target_idx = min(10, len(recommender.df) - 1)
    print(recommender.recommend(target_idx).to_string(index=False))
    recommender.fit_kmeans(n_clusters=min(5, len(recommender.df)))
    cluster = recommender.df.iloc[target_idx]["cluster"]
    print(recommender.get_cluster_sample(cluster).to_string(index=False))
