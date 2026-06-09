"""Engineering limit-state checks on the FEM result.

Produces structured ``violation`` dicts (consistent with
``scaffold.validators``) and a ``report`` summarising utilisation, the
governing member, and a 0–100 ``safety_score`` derived from real mechanics
rather than a heuristic.

Limit states evaluated:
  * STRUCTURAL_UNSTABLE – model is a mechanism / non-finite (mechanism is a
    genuine engineering failure, not an internal error).
  * STRENGTH_OVERSTRESS – combined axial+bending stress exceeds f_y/γ_M0.
  * MEMBER_BUCKLING     – compression exceeds Euler critical load / γ.
  * BASE_UPLIFT         – a base support develops net tension (overturning).
  * DEFLECTION_EXCEEDED – serviceability deflection beyond span/limit (advisory).
"""

from __future__ import annotations

import math
import uuid
from typing import Any

from scaffold.spec import (
    DEFAULT_CATALOG,
    DEFAULT_LOADS,
    Catalog,
    StructuralLoadSpec,
)
from scaffold.structural.fem import FemResult, solve_structure
from scaffold.structural.graph import build_structural_graph

try:
    from trace.decision_trace import add_constraint_eval
except Exception:  # tracing is optional
    add_constraint_eval = None  # type: ignore


def _score_from_utilisation(worst_util: float, *, stable: bool, uplift: bool) -> int:
    """Map worst utilisation to 0–100. ≥50 ⇒ passes ULS (util ≤ 1)."""
    if not stable:
        return 0
    score = 100.0 * (1.0 - min(worst_util, 2.0) / 2.0)
    if uplift:
        score = min(score, 25.0)
    return int(max(0.0, min(100.0, round(score))))


def structural_check(
    elements: list[dict[str, Any]],
    world_model,
    policy,
    *,
    catalog: Catalog | None = None,
    loads: StructuralLoadSpec | None = None,
    trace: list[dict[str, Any]] | None = None,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    del world_model  # structural check is geometry-of-members based, not occupancy
    catalog = catalog or DEFAULT_CATALOG
    loads = loads or _loads_from_policy(policy)

    violations: list[dict[str, Any]] = []
    decision_id = f"structural:{uuid.uuid4()}"

    graph = build_structural_graph(elements)
    result: FemResult = solve_structure(graph, catalog=catalog, loads=loads)

    report: dict[str, Any] = {
        "stable": result.stable,
        "n_members": len(graph.members),
        "n_nodes": len(graph.nodes),
        "footprint_area_m2": round(result.footprint_area_m2, 3),
        "total_self_weight_N": round(result.total_self_weight_N, 1),
        "total_live_N": round(result.total_live_N, 1),
        "notes": result.notes,
        "checks": {},
    }

    if not result.stable:
        violations.append({
            "type": "STRUCTURAL_UNSTABLE",
            "msg": "FEM model is unstable (mechanism) or produced non-finite results",
            "notes": result.notes,
        })
        report["safety_score"] = 0
        report["passes_uls"] = False
        report["worst_utilisation"] = None
        _trace(trace, decision_id, "structural_stability", False, "unstable", {"notes": result.notes})
        return False, violations, report

    worst_util = 0.0
    governing = None
    strength_fail = 0
    buckling_fail = 0

    for mr in result.members.values():
        # graph member.part_id defaults to elem_type, matching catalog keys
        sec = catalog.section_for(mr.elem_type) or catalog.section_for("post")
        mat = catalog.material_for(mr.elem_type) or catalog.material_for("post")
        if sec is None or mat is None:
            continue

        # ── Combined stress (axial + bending) ────────────────────────────
        axial_mag = max(mr.compression_N, mr.tension_N)
        sigma_axial = axial_mag / sec.A if sec.A > 0 else float("inf")
        sigma_bend = (mr.moment_Nm * sec.outer_radius_m / sec.I_min) if sec.I_min > 0 else 0.0
        sigma = sigma_axial + sigma_bend
        allow = mat.fy / loads.gamma_m0
        strength_util = sigma / allow if allow > 0 else float("inf")

        # ── Member (Euler) buckling for compression members ──────────────
        buckling_util = 0.0
        n_cr = None
        if mr.compression_N > 1.0 and mr.length_m > 0.1:
            n_cr = (math.pi ** 2) * mat.E * sec.I_min / ((loads.buckling_K * mr.length_m) ** 2)
            allow_buck = n_cr / loads.gamma_buckling
            buckling_util = mr.compression_N / allow_buck if allow_buck > 0 else float("inf")

        member_util = max(strength_util, buckling_util)
        if member_util > worst_util:
            worst_util = member_util
            governing = {
                "member_id": mr.member_id,
                "elem_type": mr.elem_type,
                "elem_index": mr.elem_index,
                "strength_util": round(strength_util, 3),
                "buckling_util": round(buckling_util, 3),
                "axial_N": round(mr.axial_N, 1),
                "moment_Nm": round(mr.moment_Nm, 1),
            }

        if strength_util > 1.0:
            strength_fail += 1
            violations.append({
                "type": "STRENGTH_OVERSTRESS",
                "element_index": mr.elem_index,
                "elem_type": mr.elem_type,
                "utilisation": round(strength_util, 3),
                "stress_Pa": round(sigma, 0),
                "allow_Pa": round(allow, 0),
                "margin": round(1.0 - strength_util, 3),
            })
        if buckling_util > 1.0:
            buckling_fail += 1
            violations.append({
                "type": "MEMBER_BUCKLING",
                "element_index": mr.elem_index,
                "elem_type": mr.elem_type,
                "utilisation": round(buckling_util, 3),
                "compression_N": round(mr.compression_N, 1),
                "n_cr_N": round(n_cr, 1) if n_cr is not None else None,
                "length_m": round(mr.length_m, 3),
                "margin": round(1.0 - buckling_util, 3),
            })

    # ── Base uplift / overturning ────────────────────────────────────────
    uplift_nodes = [b for b, rz in result.base_reactions_N.items() if rz < -1.0]
    if uplift_nodes:
        violations.append({
            "type": "BASE_UPLIFT",
            "msg": "Base support(s) develop net tension — overturning / anchoring required",
            "count": len(uplift_nodes),
            "min_reaction_N": round(min(result.base_reactions_N.values()), 1),
        })

    # ── Serviceability deflection (advisory) ─────────────────────────────
    span = max((n.z for n in graph.nodes.values()), default=0.0) - graph.min_z
    defl_limit = span / loads.deflection_limit_ratio if span > 0 else float("inf")
    defl_exceeded = result.max_deflection_m > defl_limit
    if defl_exceeded:
        violations.append({
            "type": "DEFLECTION_EXCEEDED",
            "severity": "serviceability",
            "max_deflection_m": round(result.max_deflection_m, 4),
            "limit_m": round(defl_limit, 4),
        })

    passes_uls = (worst_util <= 1.0) and not uplift_nodes
    report["checks"] = {
        "strength_failures": strength_fail,
        "buckling_failures": buckling_fail,
        "uplift_supports": len(uplift_nodes),
        "max_deflection_m": round(result.max_deflection_m, 4),
        "deflection_limit_m": round(defl_limit, 4) if math.isfinite(defl_limit) else None,
    }
    report["worst_utilisation"] = round(worst_util, 3)
    report["governing_member"] = governing
    report["passes_uls"] = bool(passes_uls)
    report["safety_score"] = _score_from_utilisation(worst_util, stable=True, uplift=bool(uplift_nodes))

    _trace(trace, decision_id, "structural_strength", strength_fail == 0,
           "ok" if strength_fail == 0 else "overstress", {"worst_util": round(worst_util, 3)})
    _trace(trace, decision_id, "structural_buckling", buckling_fail == 0,
           "ok" if buckling_fail == 0 else "buckling", {"failures": buckling_fail})
    _trace(trace, decision_id, "structural_overturning", not uplift_nodes,
           "ok" if not uplift_nodes else "uplift", {"uplift_nodes": len(uplift_nodes)})

    # Serviceability deflection alone does not fail the ULS gate, but any
    # strength/buckling/uplift/stability violation does.
    hard_ok = strength_fail == 0 and buckling_fail == 0 and not uplift_nodes and result.stable
    return hard_ok, violations, report


def _loads_from_policy(policy) -> StructuralLoadSpec:
    """Build a load spec, letting policy override the defaults when present."""
    d = DEFAULT_LOADS
    g = lambda name, default: float(getattr(policy, name, default)) if policy is not None else default
    return StructuralLoadSpec(
        live_load_kN_per_m2=g("structural_live_load_kN_per_m2", d.live_load_kN_per_m2),
        dead_load_factor=g("structural_dead_factor", d.dead_load_factor),
        live_load_factor=g("structural_live_factor", d.live_load_factor),
        wind_pressure_kN_per_m2=g("structural_wind_kN_per_m2", d.wind_pressure_kN_per_m2),
        gamma_m0=g("structural_gamma_m0", d.gamma_m0),
        gamma_buckling=g("structural_gamma_buckling", d.gamma_buckling),
        buckling_K=g("structural_buckling_K", d.buckling_K),
        deflection_limit_ratio=g("structural_deflection_ratio", d.deflection_limit_ratio),
    )


def _trace(trace, decision_id, constraint_id, ok, reason, metrics) -> None:
    if trace is None or add_constraint_eval is None:
        return
    try:
        add_constraint_eval(
            trace, decision_id=decision_id, constraint_id=constraint_id,
            ok=bool(ok), reason=reason, metrics=metrics, element_id=None,
        )
    except Exception:
        pass
