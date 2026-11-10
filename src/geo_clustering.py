import html
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from folium.plugins import HeatMap
from sklearn.cluster import DBSCAN, KMeans

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "bangalore-restaurants.csv"


class GeoAnalyzer:
    def __init__(self, data_path=None):
        data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH
        try:
            self.df = pd.read_csv(
                data_path,
                encoding="utf-8",
            )
        except UnicodeDecodeError:
            self.df = pd.read_csv(
                data_path,
                encoding="latin-1",
            )

    def preprocess(
        self,
        allow_synthetic=True,
        synthetic_center=(12.9716, 77.5946),
        center_name="Bengaluru",
        synthetic_radius_km=12,
    ):
        coordinate_pairs = (
            ("latitude", "longitude"),
            ("lat", "long"),
            ("lat", "lon"),
            ("lat", "lng"),
        )
        pair = next(
            ((lat, lon) for lat, lon in coordinate_pairs
             if lat in self.df.columns and lon in self.df.columns),
            None,
        )
        if pair is None:
            if not allow_synthetic:
                raise ValueError(
                    "В датасете нет широты и долготы. Включите allow_synthetic "
                    "для создания демонстрационных координат."
                )
            if "location" not in self.df.columns:
                raise ValueError(
                    "Для генерации демонстрационных координат требуется колонка location."
                )
            self.df = self.df.dropna(subset=["location"]).copy()
            self.df = self.df.drop_duplicates(
                subset=[
                    column for column in ("name", "location")
                    if column in self.df.columns
                ]
            ).reset_index(drop=True)

            rng = np.random.default_rng(42)
            center_lat, center_lon = synthetic_center
            if not (-90 <= center_lat <= 90 and -180 <= center_lon <= 180):
                raise ValueError("Центр должен иметь корректные широту и долготу.")
            if synthetic_radius_km <= 0:
                raise ValueError(
                    "Радиус синтетического распределения должен быть положительным."
                )
            self.synthetic_center_name = center_name
            self.synthetic_center = [center_lat, center_lon]
            angles = rng.uniform(0, 2 * np.pi, len(self.df))
            radii = synthetic_radius_km * np.sqrt(rng.uniform(0, 1, len(self.df)))
            north_south_km = radii * np.sin(angles)
            east_west_km = radii * np.cos(angles)
            self.latitude_column, self.longitude_column = "latitude", "longitude"
            self.df["latitude"] = center_lat + north_south_km / 111.32
            self.df["longitude"] = center_lon + east_west_km / (
                111.32 * np.cos(np.radians(center_lat))
            )
            self.synthetic_coordinates = True
            print(
                "Внимание: GPS-координаты синтетические и созданы только "
                f"для демонстрации около {center_name}."
            )
        else:
            self.latitude_column, self.longitude_column = pair
            self.df[self.latitude_column] = pd.to_numeric(
                self.df[self.latitude_column], errors="coerce"
            )
            self.df[self.longitude_column] = pd.to_numeric(
                self.df[self.longitude_column], errors="coerce"
            )
            self.df = self.df.dropna(
                subset=[self.latitude_column, self.longitude_column]
            ).copy()
            self.df = self.df[
                self.df[self.latitude_column].between(-90, 90)
                & self.df[self.longitude_column].between(-180, 180)
            ].reset_index(drop=True)
            self.synthetic_coordinates = False

        if self.df.empty:
            raise ValueError("После обработки не осталось координат для геоанализа.")
        self.coords = np.radians(
            self.df[[self.latitude_column, self.longitude_column]].to_numpy()
        )
        return self

    def preprocess_and_mock_geo(
        self,
        synthetic_center=(12.9716, 77.5946),
        center_name="Bengaluru",
        synthetic_radius_km=12,
    ):
        return self.preprocess(
            allow_synthetic=True,
            synthetic_center=synthetic_center,
            center_name=center_name,
            synthetic_radius_km=synthetic_radius_km,
        )

    def run_dbscan(self, eps_km=1.0, min_samples=15):
        if not hasattr(self, "coords"):
            raise RuntimeError("Сначала вызовите preprocess().")
        if eps_km <= 0 or min_samples < 1:
            raise ValueError("eps_km и min_samples должны быть положительными.")
        earth_radius_km = 6371.0088
        self.dbscan = DBSCAN(
            eps=eps_km / earth_radius_km,
            min_samples=min_samples,
            algorithm="ball_tree",
            metric="haversine",
        )
        self.df["hub_cluster"] = self.dbscan.fit_predict(self.coords)
        cluster_count = len(set(self.df["hub_cluster"])) - (
            1 if -1 in self.df["hub_cluster"] else 0
        )
        noise_count = int((self.df["hub_cluster"] == -1).sum())
        print(f"Найдено плотных хабов: {cluster_count}")
        print(f"Ресторанов вне хабов: {noise_count}")
        return self.df["hub_cluster"]

    def run_kmeans(self, n_clusters=5):
        if not hasattr(self, "coords"):
            raise RuntimeError("Сначала вызовите preprocess().")
        if not 1 <= n_clusters <= len(self.df):
            raise ValueError("Число кластеров должно быть от 1 до числа ресторанов.")
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df["kmeans_cluster"] = self.kmeans.fit_predict(self.coords)
        return self.df["kmeans_cluster"]

    def generate_map(self, output_file="restaurant_hubs.html", sample_size=1000):
        if "hub_cluster" not in self.df:
            raise RuntimeError("Сначала вызовите run_dbscan().")
        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = ROOT_DIR / "reports" / "maps" / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lat = self.latitude_column
        lon = self.longitude_column
        map_view = folium.Map(
            location=(
                self.synthetic_center
                if getattr(self, "synthetic_coordinates", False)
                else [self.df[lat].mean(), self.df[lon].mean()]
            ),
            zoom_start=11 if getattr(self, "synthetic_coordinates", False) else 3,
            tiles=(
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
            ),
            attr="Esri, HERE, Garmin, (c) OpenStreetMap contributors",
        )
        if getattr(self, "synthetic_coordinates", False):
            folium.Element(
                "<div style='position:fixed;top:10px;left:50px;z-index:9999;"
                "background:#fff3cd;padding:10px;border:1px solid #856404'>"
                f"DEMO ONLY: synthetic coordinates around "
                f"{html.escape(self.synthetic_center_name)}; "
                "not real restaurant locations.</div>"
            ).add_to(map_view.get_root().html)
        sample = self.df.sample(
            n=min(sample_size, len(self.df)), random_state=42
        )
        HeatMap(
            sample[[lat, lon]].values.tolist(),
            name="Плотность ресторанов",
            radius=12,
            blur=10,
        ).add_to(map_view)
        colors = [
            "red", "blue", "green", "purple", "orange", "darkred",
            "cadetblue", "beige", "darkblue", "darkgreen",
        ]
        for _, row in sample.iterrows():
            cluster_id = int(row["hub_cluster"])
            color = "gray" if cluster_id == -1 else colors[cluster_id % len(colors)]
            label = str(row.get("name", "Restaurant"))
            location = str(row.get("location", ""))
            folium.CircleMarker(
                location=[row[lat], row[lon]],
                radius=4,
                tooltip=f"{label} ({location})",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
            ).add_to(map_view)
        folium.LayerControl().add_to(map_view)
        map_view.save(output_path)
        if getattr(self, "synthetic_coordinates", False):
            print("Карта использует демонстрационные синтетические координаты.")
        print(f"Карта сохранена: {output_path}")
        return output_path

    def generate_correlation_heatmap(self, output_file="restaurant_correlations.html"):
        numeric = self.df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            raise ValueError("Для корреляционной тепловой карты нужно минимум 2 числовых поля.")
        correlation = numeric.corr()
        cells = []
        for column in correlation.columns:
            cells.append(f"<th>{html.escape(str(column))}</th>")
        rows = ["    <tr><th></th>" + "".join(cells) + "</tr>"]
        for index, row in correlation.iterrows():
            cells = []
            for value in row:
                score = 0.0 if pd.isna(value) else float(value)
                red = int(255 * max(score, 0))
                blue = int(255 * max(-score, 0))
                green = int(255 * (1 - abs(score)))
                cells.append(
                    f'<td style="background:rgb({red},{green},{blue})">'
                    f"{score:.2f}</td>"
                )
            rows.append(
                f"    <tr><th>{html.escape(str(index))}</th>{''.join(cells)}</tr>"
            )

        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = ROOT_DIR / "reports" / "maps" / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        document = "\n".join(
            [
                "<!doctype html>",
                "<html>",
                "<head>",
                "  <meta charset='utf-8'>",
                "  <title>Restaurant correlations</title>",
                "  <style>",
                "    body { font: 14px sans-serif; padding: 24px; }",
                "    table { border-collapse: collapse; }",
                "    th, td { border: 1px solid #bbb; padding: 8px; text-align: center; }",
                "    th { background: #eee; }",
                "  </style>",
                "</head>",
                "<body>",
                "  <h1>Restaurant feature correlations</h1>",
                "  <table>",
                *rows,
                "  </table>",
                "</body>",
                "</html>",
                "",
            ]
        )
        output_path.write_text(document, encoding="utf-8")
        print(f"Тепловая карта корреляций сохранена: {output_path}")
        return output_path


if __name__ == "__main__":
    datasets = [
        {
            "path": DEFAULT_DATA_PATH,
            "center": (12.9716, 77.5946),
            "city": "Bengaluru",
            "map": "bangalore-restaurants.html",
            "correlations": "bangalore-restaurants-correlations.html",
        },
        {
            "path": ROOT_DIR / "data" / "zomato_astana_dirty_full.csv",
            "center": (51.1694, 71.4491),
            "city": "Astana",
            "map": "astana-restaurants.html",
            "correlations": "astana-restaurants-correlations.html",
        },
    ]
    for dataset in datasets:
        if not dataset["path"].exists():
            print(f"Датасет не найден, пропускаю: {dataset['path']}")
            continue
        print(f"Анализ датасета: {dataset['path']} ({dataset['city']})")
        geo = GeoAnalyzer(dataset["path"])
        geo.preprocess(
            synthetic_center=dataset["center"],
            center_name=dataset["city"],
        )
        geo.run_dbscan()
        geo.run_kmeans()
        geo.generate_map(dataset["map"])
        geo.generate_correlation_heatmap(dataset["correlations"])
