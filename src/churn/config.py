"""Project configuration: one YAML file, paths relative to the repo root."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path = ROOT / "config" / "config.yaml") -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))
