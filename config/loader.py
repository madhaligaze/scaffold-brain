from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | None = None) -> Dict[str, Any]:
    cfg_path = Path(path or os.getenv("BACKEND_CONFIG") or Path(__file__).with_name("default.yaml"))
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
