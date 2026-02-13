"""Configuration loader for FileGramPlugin."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PluginConfig:
    """Runtime configuration."""

    watch_dirs: list[str] = field(default_factory=list)
    ignore_patterns: list[str] = field(default_factory=list)
    dedup_window_sec: float = 2.0
    snapshot_interval_sec: int = 300
    binary_size_threshold: int = 1_048_576
    text_extensions: list[str] = field(default_factory=list)
    profile_id: str = "real_user"


def load_config(config_path: str | Path | None = None) -> PluginConfig:
    """Load configuration from YAML file.

    Falls back to ``config.yaml`` next to this module if *config_path* is None.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return PluginConfig()

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return PluginConfig(
        watch_dirs=raw.get("watch_dirs", []),
        ignore_patterns=raw.get("ignore_patterns", []),
        dedup_window_sec=float(raw.get("dedup_window_sec", 2.0)),
        snapshot_interval_sec=int(raw.get("snapshot_interval_sec", 300)),
        binary_size_threshold=int(raw.get("binary_size_threshold", 1_048_576)),
        text_extensions=raw.get("text_extensions", []),
        profile_id=raw.get("profile_id", "real_user"),
    )
