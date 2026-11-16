import os
from functools import lru_cache
from pathlib import Path

from .backend import Backend
from .service import ModeService

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def get_service():
    return ModeService(os.environ.get("MDRPI_MODES_DATA", str(ROOT / "data/interactive_modes")),
        Backend(os.environ.get("MDRPI_MODES_CONFIG", str(ROOT / "config.yaml")),
                os.environ.get("MDRPI_RAG_URL", "http://127.0.0.1:8020")), fixed_reference_policy="web")
