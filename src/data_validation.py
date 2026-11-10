import hashlib
import re
from pathlib import Path

CHECKSUM_PATTERN = re.compile(r"^([0-9a-f]{64})\s+\*?(.+)$")


def verify_dataset_checksum(dataset_path: Path, manifest_path: Path) -> str:
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset checksum manifest not found: {manifest_path}")

    expected_checksum = None
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        match = CHECKSUM_PATTERN.fullmatch(line.strip())
        if match is None:
            raise ValueError(
                f"Invalid checksum entry at {manifest_path}:{line_number}"
            )
        checksum, filename = match.groups()
        if filename == dataset_path.name:
            if expected_checksum is not None:
                raise ValueError(
                    f"Duplicate checksum entry for {dataset_path.name} "
                    f"in {manifest_path}"
                )
            expected_checksum = checksum

    if expected_checksum is None:
        raise ValueError(
            f"No checksum recorded for {dataset_path.name} in {manifest_path}"
        )

    digest = hashlib.sha256()
    with dataset_path.open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual_checksum = digest.hexdigest()
    if actual_checksum != expected_checksum:
        raise ValueError(
            f"Checksum mismatch for {dataset_path}: expected {expected_checksum}, "
            f"got {actual_checksum}"
        )
    return actual_checksum
