"""Small helpers shared by the stage-one CLI scripts."""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import Iterable


def project_config_path() -> Path:
    """Return the default config path based on this script's location."""

    return Path(__file__).resolve().parents[1] / "config.yaml"


def add_project_to_import_path() -> Path:
    """Make the project root importable when a script is run by file path."""

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def configure_logging(log_path: Path, name: str) -> logging.Logger:
    """Configure a file and console logger without duplicating handlers."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Calculate a file digest without loading the whole PDF into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def natural_sort_key(value: str) -> list[object]:
    """Sort filenames in human order while retaining deterministic output."""

    import re

    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value)]


def write_text_atomic(path: Path, content: str) -> None:
    """Write generated text through a sibling temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def csv_rows(path: Path) -> Iterable[dict[str, str]]:
    """Yield CSV rows using UTF-8 with BOM tolerance."""

    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)
