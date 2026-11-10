from pathlib import Path

from src.data_validation import verify_dataset_checksum

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHECKSUMS_PATH = DATA_DIR / "checksums.sha256"


def main() -> None:
    for name in ("bangalore-restaurants.csv", "zomato_astana_dirty_full.csv"):
        dataset_path = DATA_DIR / name
        checksum = verify_dataset_checksum(dataset_path, CHECKSUMS_PATH)
        print(f"OK {name} sha256={checksum}")


if __name__ == "__main__":
    main()
