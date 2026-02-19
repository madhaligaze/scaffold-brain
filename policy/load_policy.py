from __future__ import annotations

from pathlib import Path

import yaml

from policy.policy_config import PolicyConfig


def _build_policy_config(raw: dict) -> PolicyConfig:
    defaults = PolicyConfig.from_config({})
    data = {
        "unknown_mode": raw.get("unknown_mode", defaults.unknown_mode),
        "unknown_buffer_m": float(raw.get("unknown_buffer_m", defaults.unknown_buffer_m)),
        "min_clearance_m": float(raw.get("min_clearance_m", defaults.min_clearance_m)),
        "readiness_observed_ratio_min": float(
            raw.get("readiness_observed_ratio_min", defaults.readiness_observed_ratio_min)
        ),
        "unknown_ratio_near_support_max": float(
            raw.get("unknown_ratio_near_support_max", defaults.unknown_ratio_near_support_max)
        ),
        "min_viewpoints": int(raw.get("min_viewpoints", defaults.min_viewpoints)),
        "min_views_per_support": int(
            raw.get("min_views_per_support", defaults.min_views_per_support)
        ),
        "min_views_per_anchor": int(raw.get("min_views_per_anchor", defaults.min_views_per_anchor)),
        "scaffold_grid_step_m": float(
            raw.get("scaffold_grid_step_m", defaults.scaffold_grid_step_m)
        ),
        "scaffold_max_bay_m": float(raw.get("scaffold_max_bay_m", defaults.scaffold_max_bay_m)),
        "scaffold_min_bay_m": float(raw.get("scaffold_min_bay_m", defaults.scaffold_min_bay_m)),
        "scaffold_default_height_m": float(
            raw.get("scaffold_default_height_m", defaults.scaffold_default_height_m)
        ),
        "scaffold_deck_levels_m": [
            float(x) for x in raw.get("scaffold_deck_levels_m", defaults.scaffold_deck_levels_m)
        ],
        "access_min_corridor_m": float(
            raw.get("access_min_corridor_m", defaults.access_min_corridor_m)
        ),
        "stability_require_diagonals": bool(
            raw.get("stability_require_diagonals", defaults.stability_require_diagonals)
        ),
        "enforce_validators_strict": bool(
            raw.get("enforce_validators_strict", defaults.enforce_validators_strict)
        ),
    }
    return PolicyConfig(**data)


def load_policy_from_yaml(path: Path) -> PolicyConfig:
    """
    Never silently fallback: if YAML exists but is invalid, raise with a clear error.
    If YAML doesn't exist, caller can decide defaults.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Policy YAML must be a mapping, got {type(raw).__name__}")
    return _build_policy_config(raw)


def find_policy_file() -> Path | None:
    """
    Resolution order (first hit wins):
      1) ./policy/policy_config.yaml
      2) ./Backend-AI/policy/policy_config.yaml (if running from repo root)
    """
    candidates = [
        Path("policy") / "policy_config.yaml",
        Path("Backend-AI") / "policy" / "policy_config.yaml",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None
