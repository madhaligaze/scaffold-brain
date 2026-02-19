from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from trace.decision_trace import add_constraint_eval


@dataclass(frozen=True)
class UnknownPolicyConfig:
    """
    Stage 15: unknown-space policy as a first-class project setting.

    mode:
      - allow  : never gate on unknown (not recommended)
      - forbid : hard fail if unknown fraction above threshold near critical points
      - buffer : allow but require extra clearance away from unknown (conservative)
    """

    mode: str = "buffer"
    forbid_radius_m: float = 0.35
    buffer_radius_m: float = 0.50
    buffer_clearance_m: float = 0.15
    max_unknown_fraction: float = 0.05
    samples_per_region: int = 256

    def validate(self) -> None:
        if self.mode not in ("allow", "forbid", "buffer"):
            raise ValueError(f"unknown_policy.mode must be allow|forbid|buffer, got: {self.mode}")
        for k in ("forbid_radius_m", "buffer_radius_m", "buffer_clearance_m"):
            v = float(getattr(self, k))
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"unknown_policy.{k} must be finite and >= 0, got: {v}")
        if not (0.0 <= float(self.max_unknown_fraction) <= 1.0):
            raise ValueError(
                f"unknown_policy.max_unknown_fraction must be within [0,1], got: {self.max_unknown_fraction}"
            )
        n = int(self.samples_per_region)
        if n <= 0 or n > 100000:
            raise ValueError(f"unknown_policy.samples_per_region must be in 1..100000, got: {n}")


class UnknownSampler:
    """
    Adapter over your world model / occupancy / ESDF.

    Provide a callable that returns one of:
      - "unknown"
      - "free"
      - "occupied"
    """

    def __init__(self, sample_fn: Callable[[tuple[float, float, float]], str]):
        self._sample_fn = sample_fn

    def sample_label(self, p_w: tuple[float, float, float]) -> str:
        v = self._sample_fn(p_w)
        if v not in ("unknown", "free", "occupied"):
            # normalize unknown responses to avoid crashing on custom backends
            return "unknown"
        return v


def _rand_point_in_sphere(center: tuple[float, float, float], radius: float) -> tuple[float, float, float]:
    # Rejection sampling inside unit sphere, then scale.
    # Good enough for policy gating (not used for precision geometry).
    cx, cy, cz = center
    r = float(radius)
    if r <= 0:
        return center
    while True:
        x = random.uniform(-1.0, 1.0)
        y = random.uniform(-1.0, 1.0)
        z = random.uniform(-1.0, 1.0)
        if x * x + y * y + z * z <= 1.0:
            return (cx + x * r, cy + y * r, cz + z * r)


def estimate_unknown_fraction(
    sampler: UnknownSampler,
    *,
    center_w: tuple[float, float, float],
    radius_m: float,
    samples: int,
) -> float:
    if radius_m <= 0 or samples <= 0:
        return 0.0
    unknown = 0
    for _ in range(samples):
        p = _rand_point_in_sphere(center_w, radius_m)
        if sampler.sample_label(p) == "unknown":
            unknown += 1
    return float(unknown) / float(samples)


@dataclass(frozen=True)
class UnknownPolicyDecision:
    ok: bool
    mode: str
    unknown_fraction: float
    radius_m: float
    required_clearance_m: float
    reason: str | None = None


def evaluate_unknown_policy(
    cfg: UnknownPolicyConfig,
    sampler: UnknownSampler,
    *,
    critical_points_w: Iterable[tuple[float, float, float]],
    decision_id: str,
    trace: list[dict[str, Any]] | None = None,
) -> UnknownPolicyDecision:
    """
    Returns a single conservative decision aggregated over all critical points:
      - Forbid: fail if any point exceeds threshold
      - Buffer: ok, but clearance may be increased if unknown exceeds threshold
      - Allow: always ok
    """
    cfg.validate()

    if cfg.mode == "allow":
        return UnknownPolicyDecision(
            ok=True,
            mode=cfg.mode,
            unknown_fraction=0.0,
            radius_m=0.0,
            required_clearance_m=0.0,
            reason="mode=allow",
        )

    if cfg.mode == "forbid":
        radius = float(cfg.forbid_radius_m)
    else:
        radius = float(cfg.buffer_radius_m)

    worst = 0.0
    for pt in critical_points_w:
        frac = estimate_unknown_fraction(
            sampler,
            center_w=pt,
            radius_m=radius,
            samples=int(cfg.samples_per_region),
        )
        worst = max(worst, frac)

    ok = bool(worst <= float(cfg.max_unknown_fraction))
    required_clearance = 0.0
    reason: str | None = None

    if cfg.mode == "forbid":
        if not ok:
            reason = "unknown_fraction_above_threshold"
        else:
            reason = "unknown_ok"
    else:
        # buffer mode
        if not ok:
            required_clearance = float(cfg.buffer_clearance_m)
            reason = "unknown_requires_buffer"
        else:
            reason = "unknown_ok"

    if trace is not None:
        add_constraint_eval(
            trace,
            decision_id=decision_id,
            constraint_id="unknown_space_policy",
            ok=ok if cfg.mode == "forbid" else True,
            reason=reason,
            metrics={
                "mode": cfg.mode,
                "radius_m": radius,
                "worst_unknown_fraction": worst,
                "threshold": float(cfg.max_unknown_fraction),
                "required_clearance_m": required_clearance,
            },
            severity="warning" if (cfg.mode == "forbid" and not ok) else "info",
        )

    # For buffer mode we do not fail; we increase clearance.
    if cfg.mode == "buffer":
        return UnknownPolicyDecision(
            ok=True,
            mode=cfg.mode,
            unknown_fraction=worst,
            radius_m=radius,
            required_clearance_m=required_clearance,
            reason=reason,
        )

    return UnknownPolicyDecision(
        ok=ok,
        mode=cfg.mode,
        unknown_fraction=worst,
        radius_m=radius,
        required_clearance_m=0.0,
        reason=reason,
    )


def check_points_against_unknown(world, points, *, mode: str = "buffer", buffer_m: float = 0.5) -> list[dict[str, Any]]:
    """Legacy helper: returns per-point unknown violations."""
    violations: list[dict[str, Any]] = []
    occ = getattr(world, "occupancy", None)
    for i, p in enumerate(points or []):
        label = "unknown"
        try:
            if occ is not None and hasattr(occ, "query"):
                q = occ.query([p])
                if q and int(q[0]) != 0:
                    label = "known"
        except Exception:
            label = "unknown"

        if mode == "buffer" and label == "unknown":
            violations.append(
                {
                    "type": "UNKNOWN_BUFFER",
                    "point_index": i,
                    "position": [float(p[0]), float(p[1]), float(p[2])],
                    "buffer_m": float(buffer_m),
                }
            )
    return violations


def apply_unknown_policy(world, anchors, policy) -> dict[str, Any]:
    """
    Compatibility adapter expected by session routes.

    Returns a serializable policy decision for status endpoints.
    """
    cfg = UnknownPolicyConfig(
        mode=getattr(policy, "unknown_mode", "buffer"),
        forbid_radius_m=float(getattr(policy, "unknown_forbid_radius_m", 0.35)),
        buffer_radius_m=float(getattr(policy, "unknown_buffer_radius_m", 0.50)),
        buffer_clearance_m=float(getattr(policy, "unknown_buffer_clearance_m", 0.15)),
        max_unknown_fraction=float(getattr(policy, "unknown_max_fraction", 0.05)),
        samples_per_region=int(getattr(policy, "unknown_samples_per_region", 256)),
    )

    occupancy = getattr(world, "occupancy", None)

    def _sample(p_w: tuple[float, float, float]) -> str:
        if occupancy is None:
            return "unknown"
        try:
            if hasattr(occupancy, "sample_label"):
                return str(occupancy.sample_label(p_w))
            if hasattr(occupancy, "label_at_world"):
                return str(occupancy.label_at_world(p_w))
        except Exception:
            return "unknown"
        return "unknown"

    critical_points = []
    for a in anchors or []:
        try:
            p = a.get("position") if isinstance(a, dict) else None
            if p and len(p) == 3:
                critical_points.append((float(p[0]), float(p[1]), float(p[2])))
        except Exception:
            continue

    if not critical_points:
        critical_points = [(0.0, 0.0, 0.0)]

    decision = evaluate_unknown_policy(
        cfg,
        UnknownSampler(_sample),
        critical_points_w=critical_points,
        decision_id="session_status_unknown_policy",
    )

    occ_stats = {}
    occupancy = getattr(world, "occupancy", None)
    if occupancy is not None and hasattr(occupancy, "stats"):
        try:
            occ_stats = occupancy.stats(points=critical_points) or {}
        except Exception:
            occ_stats = {}

    unknown_ratio = float(decision.unknown_fraction)
    total = float(occ_stats.get("total", 0) or 0)
    if total > 0:
        unknown_ratio = float(occ_stats.get("unknown", 0) or 0) / total

    violations: list[dict[str, Any]] = []
    if cfg.mode == "forbid" and (not decision.ok):
        violations.append({"type": "UNKNOWN_FORBID", "reason": decision.reason})
    if cfg.mode == "buffer" and float(decision.required_clearance_m) > 0:
        violations.append(
            {
                "type": "UNKNOWN_BUFFER",
                "required_clearance_m": float(decision.required_clearance_m),
                "reason": decision.reason,
            }
        )

    return {
        "ok": decision.ok,
        "mode": decision.mode,
        "unknown_fraction": decision.unknown_fraction,
        "unknown_ratio_near_support": unknown_ratio,
        "radius_m": decision.radius_m,
        "required_clearance_m": decision.required_clearance_m,
        "reason": decision.reason,
        "violations": violations,
        "counts": {"violations": len(violations)},
    }
