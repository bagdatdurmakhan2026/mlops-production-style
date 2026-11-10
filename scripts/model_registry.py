import argparse
import os
import shutil
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from api.app import MODEL_PATH, load_model_artifact

MODEL_NAME = "restaurant-rating"


def list_versions(client: MlflowClient) -> list[tuple[int, str, str]]:
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    return [
        (version.version, version.status, ", ".join(version.aliases or []))
        for version in sorted(versions, key=lambda item: int(item.version), reverse=True)
    ]


def promote_version(
    version_number: str,
    client: MlflowClient,
    model_path: Path = MODEL_PATH,
) -> None:
    model_version = client.get_model_version(MODEL_NAME, version_number)
    if model_version.status != "READY":
        raise RuntimeError(
            f"Model version {version_number} is not ready: {model_version.status}"
        )
    source_path = client.download_artifacts(
        model_version.run_id, "deployment/model.pkl"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=model_path.parent,
            prefix=f".{model_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        shutil.copyfile(source_path, temporary_path)
        load_model_artifact(temporary_path)
        client.set_registered_model_alias(MODEL_NAME, "champion", version_number)
        os.replace(temporary_path, model_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List or promote versions of the restaurant rating model."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List registered model versions and aliases.")
    promote_parser = commands.add_parser(
        "promote", help="Activate a version and update the champion alias."
    )
    promote_parser.add_argument("version", help="Registered model version number.")
    args = parser.parse_args()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        parser.error("Set MLFLOW_TRACKING_URI to the MLflow tracking server.")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    if args.command == "list":
        versions = list_versions(client)
        if not versions:
            print("No registered versions found.")
            return
        print("VERSION\tSTATUS\tALIASES")
        for version, status, aliases in versions:
            print(f"{version}\t{status}\t{aliases}")
        return

    promote_version(args.version, client)
    print(
        f"Model version {args.version} is now champion. "
        "Restart the API to load it."
    )


if __name__ == "__main__":
    main()
