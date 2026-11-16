"""Configuration loading with paths resolved relative to the project root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppConfig:
    """Thin typed facade over ``config.yaml``."""

    project_root: Path
    values: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        return cls(project_root=config_path.parent, values=values)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Read a nested value without raising on a missing optional key."""

        current: Any = self.values
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    def path(self, *keys: str, default: str | None = None) -> Path:
        """Resolve a configured path against the project root."""

        value = self.get(*keys, default=default)
        if value is None:
            raise KeyError(f"Missing path configuration: {'.'.join(keys)}")
        resolved = Path(str(value)).expanduser()
        return resolved if resolved.is_absolute() else self.project_root / resolved

    @property
    def version(self) -> str:
        return str(self.get("project", "version", default="MDRPI_KB_v1.0"))

    @property
    def source_dir(self) -> Path:
        return self.path("paths", "source")

    @property
    def manifest_path(self) -> Path:
        return self.path("paths", "manifest")

    @property
    def parsed_dir(self) -> Path:
        return self.path("paths", "parsed")

    @property
    def chunks_path(self) -> Path:
        return self.path("paths", "chunks")

    @property
    def logs_dir(self) -> Path:
        return self.path("paths", "logs", default="logs")

    @property
    def qdrant_path(self) -> Path:
        return self.path("qdrant", "path", default="qdrant_storage")

    @property
    def qdrant_collection(self) -> str:
        return str(self.get("qdrant", "collection", default="mdrpi_kb_v1_dev"))


def ensure_output_dirs(config: AppConfig) -> None:
    """Create only generated-data directories required by the current stage."""

    for directory in (
        config.source_dir,
        config.manifest_path.parent,
        config.parsed_dir,
        config.chunks_path.parent,
        config.logs_dir,
        config.qdrant_path,
    ):
        directory.mkdir(parents=True, exist_ok=True)
