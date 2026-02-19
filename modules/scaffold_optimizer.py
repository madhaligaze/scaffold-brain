"""Scaffold planner: constraint optimization (Stage 4).

Implements:
- explicit constraints (modules.constraints)
- beam-search over discrete Layher grid choices
- deterministic scoring and selection
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math

from .builder import ScaffoldGenerator
from .constraints import ConstraintConfig, evaluate_variant, score_variant
from .layher_standards import LayherStandards
from .voxel_world import VoxelWorld


@dataclass
class SolveMeta:
    status: str
    warnings: List[str]
    scan_hints: List[Dict[str, float]]


class ScaffoldOptimizer:
    def __init__(
        self,
        *,
        generator: ScaffoldGenerator,
        voxel_world: Optional[VoxelWorld],
        obstacles: Optional[List[Dict[str, Any]]] = None,
        config: Optional[ConstraintConfig] = None,
    ):
        self.generator = generator
        self.world = voxel_world
        self.obstacles = list(obstacles or [])
        self.config = config or ConstraintConfig()

    @staticmethod
    def _anchor_center(points: List[Dict[str, Any]]) -> Tuple[float, float]:
        if not points:
            return (0.0, 0.0)
        xs = [float(p.get("x", 0.0)) for p in points]
        ys = [float(p.get("y", 0.0)) for p in points]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @staticmethod
    def _translate_variant(variant: Dict[str, Any], dx: float, dy: float) -> Dict[str, Any]:
        nodes = list(variant.get("nodes") or [])
        for n in nodes:
            n["x"] = float(n["x"]) + dx
            n["y"] = float(n["y"]) + dy
        variant["origin"] = {"x": dx, "y": dy}
        return variant

    def _candidate_grid_params(self, bounds: Dict[str, float]) -> List[Dict[str, Any]]:
        W = max(float(bounds.get("w", 4.0)), 1.0)
        H = max(float(bounds.get("h", 3.0)), 1.0)
        D = max(float(bounds.get("d", 2.0)), 1.0)

        # Discrete Layher grid search space (small but useful)
        ledger_choices = [1.09, 1.57, 2.07]
        ledger_choices = [LayherStandards.get_nearest_ledger_length(v) for v in ledger_choices]
        ledger_choices = sorted(set(ledger_choices))

        stand_choices = [2.0, 2.57, 3.07]
        stand_choices = [LayherStandards.get_nearest_standard_height(v) for v in stand_choices]
        stand_choices = sorted(set(stand_choices))

        params: List[Dict[str, Any]] = []
        for L in ledger_choices:
            nx = max(1, int(math.ceil(W / L)))
            ny = max(1, int(math.ceil(D / L)))
            # explore +-1 bays for cost/fit tradeoffs
            nx_opts = sorted(set([max(1, nx - 1), nx, nx + 1]))
            ny_opts = sorted(set([max(1, ny - 1), ny, ny + 1]))
            for S in stand_choices:
                nz = max(1, int(math.ceil(H / S)))
                nz_opts = sorted(set([max(1, nz - 1), nz, nz + 1]))
                for nxi in nx_opts:
                    for nyi in ny_opts:
                        for nzi in nz_opts:
                            params.append({
                                "ledger_len": float(L),
                                "stand_len": float(S),
                                "nx": int(nxi),
                                "ny": int(nyi),
                                "nz": int(nzi),
                            })
        # deterministic ordering: smaller parts first (rough proxy)
        params.sort(key=lambda p: (p["nx"] * p["ny"] * p["nz"], p["ledger_len"], p["stand_len"]))
        return params

    def _scan_hints_from_unknown(self, bounds: Dict[str, float], center: Tuple[float, float]) -> List[Dict[str, float]]:
        if self.world is None:
            return []
        cx, cy = center
        W = float(bounds.get("w", 4.0))
        D = float(bounds.get("d", 2.0))
        # sample grid at ~0.25m to suggest unknown zones
        step = max(0.25, self.world.resolution * 4)
        hints: List[Dict[str, float]] = []
        x0 = cx - W / 2.0
        y0 = cy - D / 2.0
        for i in range(int(W / step) + 1):
            for j in range(int(D / step) + 1):
                x = x0 + i * step
                y = y0 + j * step
                # choose mid-height sampling for "will matter" area
                z = 1.0
                if self.world.get_state(x, y, z) == self.world.UNKNOWN:
                    hints.append({"x": float(x), "y": float(y), "z": float(z)})
        # keep small list
        return hints[:25]

    def solve(
        self,
        *,
        bounds: Dict[str, float],
        anchors: Optional[List[Dict[str, Any]]] = None,
        max_variants: int = 3,
        beam_width: int = 25,
        unknown_policy: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], SolveMeta]:
        cfg = self.config
        if unknown_policy is not None:
            cfg.unknown_policy = str(unknown_policy)

        anchor_points = list(anchors or [])
        center = self._anchor_center(anchor_points)

        desired = {
            "w": float(bounds.get("w", 4.0)),
            "h": float(bounds.get("h", 3.0)),
            "d": float(bounds.get("d", 2.0)),
        }

        candidates = self._candidate_grid_params(desired)

        scored: List[Tuple[float, Dict[str, Any], List[str]]] = []
        warnings_all: List[str] = []

        for p in candidates:
            L = p["ledger_len"]
            S = p["stand_len"]
            nx = p["nx"]
            ny = p["ny"]
            nz = p["nz"]
            width = float(nx) * L
            depth = float(ny) * L
            height = float(nz) * S

            label = f"OPT nx={nx} ny={ny} nz={nz} L={L:.2f} S={S:.2f}"
            try:
                variant = self.generator.create_grid_variant(
                    width=width,
                    height=height,
                    depth=depth,
                    stand_len=S,
                    ledger_len=L,
                    label=label,
                    obstacles=self.obstacles,
                )
            except Exception:
                continue

            # position: center to anchors centroid
            dx = float(center[0]) - width / 2.0
            dy = float(center[1]) - depth / 2.0
            variant = self._translate_variant(variant, dx, dy)

            cres = evaluate_variant(
                variant=variant,
                voxel_world=self.world,
                config=cfg,
                desired_bounds=desired,
            )

            if not cres.ok:
                continue

            score = score_variant(constraint_result=cres, config=cfg)
            variant["constraint_report"] = cres.to_dict()
            variant["score"] = float(score)

            scored.append((score, variant, cres.warnings))
            # keep beam width
            scored.sort(key=lambda t: t[0], reverse=True)
            scored = scored[:beam_width]

        if not scored:
            # fallback: safe minimum (or refuse if unknown forbid)
            scan_hints = self._scan_hints_from_unknown(desired, center)
            meta = SolveMeta(
                status="insufficient_data",
                warnings=[
                    "No feasible scaffold plan found under current constraints.",
                    "Either scan more (reduce UNKNOWN) or relax unknown_policy=buffer.",
                ],
                scan_hints=scan_hints,
            )
            return ([], meta)

        # choose top-K unique by score
        out: List[Dict[str, Any]] = []
        for score, v, w in scored[:max_variants]:
            out.append(v)
            for msg in w:
                if msg not in warnings_all:
                    warnings_all.append(msg)

        meta = SolveMeta(
            status="success",
            warnings=warnings_all,
            scan_hints=self._scan_hints_from_unknown(desired, center) if warnings_all else [],
        )
        return (out, meta)
