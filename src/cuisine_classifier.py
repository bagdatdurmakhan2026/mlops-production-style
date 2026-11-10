import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "bangalore-restaurants.csv"


class CuisineClassifier:
    def __init__(self, data_path=None):
        data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        try:
            self.df = pd.read_csv(data_path, encoding="utf-8")
        except UnicodeDecodeError:
            self.df = pd.read_csv(data_path, encoding="latin-1")

        self.label_encoder = LabelEncoder()
        self.models = {}

    def preprocess(self):
        required = {"cuisines", "votes", "rate", "approx_cost(for two people)"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"В датасете отсутствуют колонки: {sorted(missing)}")

        df = self.df.copy()
        df["rate_clean"] = pd.to_numeric(
            df["rate"].astype(str).str.replace("/5", "", regex=False).str.strip(),
            errors="coerce",
        )
        df["cost_clean"] = pd.to_numeric(
            df["approx_cost(for two people)"]
            .astype(str)
            .str.replace(",", "", regex=False),
            errors="coerce",
        )
        df["votes"] = pd.to_numeric(df["votes"], errors="coerce")
        df["main_cuisine"] = (
            df["cuisines"].astype(str).str.split(",").str[0].str.strip()
        )
        df = df.dropna(subset=["cuisines"])
        df = df[df["main_cuisine"].ne("") & df["main_cuisine"].ne("nan")]

        counts = df["main_cuisine"].value_counts()
        supported_classes = counts[counts >= 5].index
        df = df[df["main_cuisine"].isin(supported_classes)]
        if df["main_cuisine"].nunique() < 2:
            raise ValueError("Для классификации нужно не менее двух кухонь с 5+ примерами.")

        categorical = [
            column for column in ("location", "online_order", "book_table")
            if column in df.columns
        ]
        numeric = ["cost_clean", "rate_clean", "votes"]
        self.feature_columns = numeric + categorical
        self.X = df[self.feature_columns].copy()
        self.y = self.label_encoder.fit_transform(df["main_cuisine"])
        self.numeric_features = numeric
        self.categorical_features = categorical
        print(f"Классы ({len(self.label_encoder.classes_)}): {list(self.label_encoder.classes_)}")

    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y,
        )
        preprocessor = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    self.numeric_features,
                ),
                (
                    "categorical",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=10,
                                sparse_output=False,
                            )),
                        ]
                    ),
                    self.categorical_features,
                ),
            ],
            sparse_threshold=0,
        )
        X_train_encoded = preprocessor.fit_transform(X_train)
        X_test_encoded = preprocessor.transform(X_test)
        weights = compute_sample_weight("balanced", y_train)
        class_count = len(self.label_encoder.classes_)

        candidates = {
            "HistGradientBoosting": HistGradientBoostingClassifier(
                max_iter=150, learning_rate=0.08, max_leaf_nodes=15, random_state=42
            ),
            "XGBoost": XGBClassifier(
                objective="multi:softprob",
                num_class=class_count,
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=1,
                random_state=42,
            ),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=250,
                early_stopping=True,
                random_state=42,
            ),
        }

        results = {}
        for name, model in candidates.items():
            print(f"Обучение: {name}...")
            if name == "MLP":
                model.fit(X_train_encoded, y_train)
            else:
                model.fit(X_train_encoded, y_train, sample_weight=weights)
            predictions = model.predict(X_test_encoded)
            results[name] = {
                "model": model,
                "predictions": predictions,
                "accuracy": accuracy_score(y_test, predictions),
                "macro_f1": f1_score(y_test, predictions, average="macro"),
            }

        best_name = max(results, key=lambda name: results[name]["macro_f1"])
        self.preprocessor = preprocessor
        self.model = results[best_name]["model"]
        self.best_model_name = best_name
        print(f"Выбрана модель: {best_name}")
        for name, result in results.items():
            print(
                f"{name}: accuracy={result['accuracy']:.4f}, "
                f"macro-F1={result['macro_f1']:.4f}"
            )
        print(
            classification_report(
                y_test,
                results[best_name]["predictions"],
                labels=range(class_count),
                target_names=self.label_encoder.classes_,
                zero_division=0,
            )
        )
        return results


if __name__ == "__main__":
    classifier = CuisineClassifier()
    classifier.preprocess()
    classifier.train()
