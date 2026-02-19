from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PolicyConfig:
    unknown_mode: str
    unknown_buffer_m: float
    min_clearance_m: float
    readiness_observed_ratio_min: float
    unknown_ratio_near_support_max: float

    # STAGE D scanning policy
    min_viewpoints: int = 3
    min_views_per_support: int = 2
    min_views_per_anchor: int = 3

    # --- STAGE H: scaffold solver/validators tuning (MVP) ---
    scaffold_grid_step_m: float = 2.0
    scaffold_max_bay_m: float = 3.07
    scaffold_min_bay_m: float = 1.57
    scaffold_default_height_m: float = 4.0
    scaffold_deck_levels_m: list[float] = field(default_factory=lambda: [2.0, 4.0])
    access_min_corridor_m: float = 0.7
    stability_require_diagonals: bool = True
    enforce_validators_strict: bool = True

    # --- STAGE 16: candidate search + repair loop ---
    planner_max_candidates: int = 24
    planner_repair_rounds: int = 8
    planner_max_shift_m: float = 0.60
    planner_score_w_violations: float = 10.0
    planner_score_w_unknown: float = 2.0
    planner_score_w_shift: float = 0.5

    @classmethod
    def from_config(cls, config: dict) -> "PolicyConfig":
        policy = config.get("policy", {}) or {}
        world = config.get("world", {}) or {}
        return cls(
            unknown_mode=policy.get("unknown_mode", "forbid"),
            unknown_buffer_m=float(policy.get("unknown_buffer_m", 0.5)),
            min_clearance_m=float(world.get("min_clearance_m", 0.2)),
            readiness_observed_ratio_min=float(policy.get("readiness_observed_ratio_min", 0.1)),
            unknown_ratio_near_support_max=float(policy.get("unknown_ratio_near_support_max", 0.6)),
            min_viewpoints=int(policy.get("min_viewpoints", 3)),
            min_views_per_support=int(policy.get("min_views_per_support", 2)),
            min_views_per_anchor=int(policy.get("min_views_per_anchor", 3)),
            scaffold_grid_step_m=float(policy.get("scaffold_grid_step_m", 2.0)),
            scaffold_max_bay_m=float(policy.get("scaffold_max_bay_m", 3.07)),
            scaffold_min_bay_m=float(policy.get("scaffold_min_bay_m", 1.57)),
            scaffold_default_height_m=float(policy.get("scaffold_default_height_m", 4.0)),
            scaffold_deck_levels_m=[
                float(x) for x in policy.get("scaffold_deck_levels_m", [2.0, 4.0])
            ],
            access_min_corridor_m=float(policy.get("access_min_corridor_m", 0.7)),
            stability_require_diagonals=bool(policy.get("stability_require_diagonals", True)),
            enforce_validators_strict=bool(policy.get("enforce_validators_strict", True)),
            planner_max_candidates=int(policy.get("planner_max_candidates", 24)),
            planner_repair_rounds=int(policy.get("planner_repair_rounds", 8)),
            planner_max_shift_m=float(policy.get("planner_max_shift_m", 0.60)),
            planner_score_w_violations=float(policy.get("planner_score_w_violations", 10.0)),
            planner_score_w_unknown=float(policy.get("planner_score_w_unknown", 2.0)),
            planner_score_w_shift=float(policy.get("planner_score_w_shift", 0.5)),
        )
