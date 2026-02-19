"""Constraint module for scaffold planning (Stage 4).

This module makes constraints explicit and checkable:
- geometry / collision / clearance
- unknown-space safety policy
- Layher discretization checks (length/height steps)
- basic stability and buildability checks

It is intentionally conservative: unknown-space is treated as risky by default.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import math

try:
    from modules.layher_standards import LayherStandards, ComponentType
    from modules.validators import validate_structure_stability
    from modules.voxel_world import VoxelWorld
except Exception:  # pragma: no cover
    from layher_standards import LayherStandards, ComponentType
    from validators import validate_structure_stability
    from voxel_world import VoxelWorld


@dataclass
class ConstraintConfig:
    clearance_min: float = 0.15
    clearance_tentative: float = 0.30
    clearance_needs_scan: float = 0.50

    # unknown safety policy:
    # - "forbid": UNKNOWN behaves like OCCUPIED (hard fail)
    # - "buffer": allow UNKNOWN but increase required clearance and emit warning
    unknown_policy: str = "forbid"
    unknown_buffer: float = 0.50

    # scoring weights
    w_coverage: float = 100.0
    w_parts: float = 1.0
    w_unknown: float = 2.0
    w_extra_area: float = 0.5


@dataclass
class ConstraintResult:
    ok: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "violations": list(self.violations),
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }


def _beam_length(a: Dict[str, float], b: Dict[str, float]) -> float:
    dx = float(a["x"]) - float(b["x"])
    dy = float(a["y"]) - float(b["y"])
    dz = float(a["z"]) - float(b["z"])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _validate_layher_lengths(beams: List[Dict], node_lookup: Dict[str, Dict]) -> List[str]:
    errs: List[str] = []
    for beam in beams:
        btype = (beam.get("type") or "").lower()
        if btype not in {"standard", "vertical", "ledger", "horizontal", "diagonal"}:
            continue
        a = node_lookup.get(beam.get("start"))
        b = node_lookup.get(beam.get("end"))
        if not a or not b:
            continue
        length = _beam_length(a, b)
        if btype in {"standard", "vertical"}:
            pool = LayherStandards.STANDARD_HEIGHTS
            ctype = ComponentType.STANDARD
        elif btype in {"ledger", "horizontal"}:
            pool = LayherStandards.LEDGER_LENGTHS
            ctype = ComponentType.LEDGER
        else:
            pool = LayherStandards.DIAGONAL_LENGTHS
            ctype = ComponentType.DIAGONAL

        nearest = min(pool, key=lambda x: abs(x - length))
        if abs(nearest - length) > 0.08:  # tolerant due to float + snapping/offsets
            if not LayherStandards.validate_dimensions(ctype, nearest):
                errs.append(f"Layher length pool invalid for {btype}: {nearest}")
            errs.append(f"Beam length not Layher-like: type={btype}, len={length:.3f}m (nearest {nearest:.2f}m)")
    return errs


def _sample_unknown_along_segment(
    world: VoxelWorld,
    p1: Dict[str, float],
    p2: Dict[str, float],
    *,
    step: Optional[float] = None,
) -> Tuple[int, int]:
    """Returns (unknown_samples, total_samples)."""
    dx = float(p2["x"]) - float(p1["x"])
    dy = float(p2["y"]) - float(p1["y"])
    dz = float(p2["z"]) - float(p1["z"])
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist < 1e-6:
        return (0, 1)
    if step is None:
        step = max(world.resolution / 2.0, 0.01)
    n = max(int(dist / step), 2)
    unk = 0
    for i in range(n + 1):
        t = i / n
        x = float(p1["x"]) + dx * t
        y = float(p1["y"]) + dy * t
        z = float(p1["z"]) + dz * t
        if world.get_state(x, y, z) == world.UNKNOWN:
            unk += 1
    return (unk, n + 1)


def evaluate_variant(
    *,
    variant: Dict[str, Any],
    voxel_world: Optional[VoxelWorld],
    config: ConstraintConfig,
    desired_bounds: Optional[Dict[str, float]] = None,
) -> ConstraintResult:
    """Checks constraints and produces metrics used by the optimizer."""
    violations: List[str] = []
    warnings: List[str] = []
    metrics: Dict[str, float] = {}

    nodes = list(variant.get("nodes") or [])
    beams = list(variant.get("beams") or [])

    if not nodes or not beams:
        return ConstraintResult(ok=False, violations=["Empty structure"], warnings=[], metrics={})

    node_lookup = {n.get("id"): n for n in nodes if n.get("id") is not None}

    # Layher discretization checks (explicit and reproducible)
    violations.extend(_validate_layher_lengths(beams, node_lookup))

    # Basic stability / buildability checks (cheap)
    try:
        structure_elems: List[Dict[str, Any]] = []
        for b in beams:
            a = node_lookup.get(b.get("start"))
            c = node_lookup.get(b.get("end"))
            if not a or not c:
                continue
            structure_elems.append({
                "type": b.get("type", "ledger"),
                "start": {"x": a["x"], "y": a["y"], "z": a["z"]},
                "end": {"x": c["x"], "y": c["y"], "z": c["z"]},
            })
        stab = validate_structure_stability(structure_elems)
        violations.extend(stab.get("errors", []) or [])
        warnings.extend(stab.get("warnings", []) or [])
    except Exception:
        # do not hard-fail due to validator issues
        warnings.append("Stability validator failed to run (ignored).")

    # Collision / clearance / unknown safety
    unknown_samples = 0
    total_samples = 0
    collision_hits = 0

    if voxel_world is not None:
        # conservative by default
        unknown_is_blocked = (config.unknown_policy or "forbid").lower() != "buffer"

        # recommended clearance for unknown-buffer mode
        clearance_unknown = max(config.clearance_needs_scan, float(config.unknown_buffer))

        for b in beams:
            a = node_lookup.get(b.get("start"))
            c = node_lookup.get(b.get("end"))
            if not a or not c:
                continue
            # collision with OCCUPIED always forbidden
            if voxel_world.is_blocked(a, c, clearance=config.clearance_min, unknown_is_blocked=False):
                collision_hits += 1
                violations.append("Collision with occupied geometry detected.")
                break

            # unknown safety policy
            if unknown_is_blocked:
                if voxel_world.is_blocked(a, c, clearance=config.clearance_min, unknown_is_blocked=True):
                    violations.append("Intersects UNKNOWN space (policy=forbid).")
                    break
            else:
                # allow UNKNOWN but measure and apply buffer check
                u, t = _sample_unknown_along_segment(voxel_world, a, c)
                unknown_samples += u
                total_samples += t
                if u > 0:
                    warnings.append("Structure passes through UNKNOWN space (policy=buffer).")
                    if voxel_world.is_blocked(a, c, clearance=clearance_unknown, unknown_is_blocked=True):
                        warnings.append("UNKNOWN buffer clearance not satisfied; plan should be treated as tentative.")

    metrics["unknown_samples"] = float(unknown_samples)
    metrics["unknown_ratio"] = float(unknown_samples) / float(total_samples or 1)
    metrics["collision_hits"] = float(collision_hits)
    metrics["parts"] = float(len(beams))

    # Coverage / extra area proxies (works without explicit decks)
    if desired_bounds:
        desired_area = float(desired_bounds.get("w", 0.0)) * float(desired_bounds.get("d", 0.0))
        if desired_area > 1e-6:
            dims = variant.get("dimensions") or {}
            area = float(dims.get("width", 0.0)) * float(dims.get("depth", 0.0))
            metrics["coverage"] = min(1.0, area / desired_area)
            metrics["extra_area"] = max(0.0, area - desired_area)
        else:
            metrics["coverage"] = 0.0
            metrics["extra_area"] = 0.0

    ok = len(violations) == 0
    return ConstraintResult(ok=ok, violations=violations, warnings=warnings, metrics=metrics)


def score_variant(
    *,
    constraint_result: ConstraintResult,
    config: ConstraintConfig,
) -> float:
    """Higher is better."""
    m = constraint_result.metrics
    coverage = float(m.get("coverage", 0.0))
    parts = float(m.get("parts", 0.0))
    unknown = float(m.get("unknown_ratio", 0.0))
    extra = float(m.get("extra_area", 0.0))

    score = 0.0
    score += float(config.w_coverage) * coverage
    score -= float(config.w_parts) * (parts / 100.0)
    score -= float(config.w_unknown) * (unknown * 100.0)
    score -= float(config.w_extra_area) * extra
    return float(score)
