"""Finite-element static analysis of a scaffold frame (PyNiteFEA wrapper).

Modelling assumptions (documented, conservative first model):
  * Members (posts / ledgers / braces) are steel tubes from the catalog and are
    moment-connected at shared nodes (rigid joints). Real modular systems
    (Layher Allround rosette, cuplok) are *semi-rigid*; rigid joints are a
    common, defensible simplification and avoid spurious mechanisms from the
    under-braced layouts the solver currently produces.
  * Base nodes (lowest plane) are pinned: translations fixed, rotations free —
    standard for base-plate/jack supports.
  * Loads: member self-weight (dead) + uniform working load on the top working
    plane (live). Optional lateral wind pressure. Combined per EN-style ULS
    (1.35·G + 1.5·Q) for strength and SLS (1.0·G + 1.0·Q) for deflection.

Sign convention: this PyNite build reports member axial force as
*compression-positive* (verified empirically and asserted in the test-suite),
so ``compression_N = max(0, max_axial)`` and ``tension_N = max(0, -min_axial)``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from scaffold.spec import (
    DEFAULT_CATALOG,
    DEFAULT_LOADS,
    STEEL_S235,
    TUBE_48_3,
    Catalog,
    Material,
    Section,
    StructuralLoadSpec,
)
from scaffold.structural.graph import StructuralGraph

G_ACCEL = 9.81  # m/s^2

CASE_DEAD = "D"
CASE_LIVE = "L"
CASE_WIND = "W"
COMBO_ULS = "ULS"
COMBO_SLS = "SLS"


@dataclass
class MemberResult:
    member_id: str
    elem_type: str
    elem_index: int
    length_m: float
    axial_N: float          # signed peak (compression-positive)
    compression_N: float    # >= 0
    tension_N: float        # >= 0
    moment_Nm: float        # peak |bending moment| over both local axes
    deflection_m: float     # peak |transverse deflection| (SLS)


@dataclass
class FemResult:
    ok: bool
    stable: bool
    members: dict[str, MemberResult] = field(default_factory=dict)
    base_reactions_N: dict[str, float] = field(default_factory=dict)  # node_id -> RxnFZ (ULS)
    total_self_weight_N: float = 0.0
    total_live_N: float = 0.0
    max_deflection_m: float = 0.0
    footprint_area_m2: float = 0.0
    notes: list[str] = field(default_factory=list)


def _finite(x: float) -> bool:
    try:
        return bool(np.isfinite(x))
    except Exception:
        return False


def _resolve_material_section(
    catalog: Catalog, part_id: str
) -> tuple[Material, Section]:
    mat = catalog.material_for(part_id) or STEEL_S235
    sec = catalog.section_for(part_id) or TUBE_48_3
    return mat, sec


def solve_structure(
    graph: StructuralGraph,
    *,
    catalog: Catalog | None = None,
    loads: StructuralLoadSpec | None = None,
) -> FemResult:
    """Assemble and solve the FEM frame for *graph*. Never raises: any solver
    failure or non-finite result is reported as ``stable=False``."""
    catalog = catalog or DEFAULT_CATALOG
    loads = loads or DEFAULT_LOADS

    if not graph.members or not graph.base_node_ids:
        return FemResult(ok=False, stable=False, notes=["empty_or_unsupported_graph"])

    try:
        from Pynite import FEModel3D
    except Exception as exc:  # dependency missing
        return FemResult(ok=False, stable=False, notes=[f"pynite_unavailable:{exc}"])

    model = FEModel3D()

    # Register the distinct materials / sections actually used.
    used_mats: set[str] = set()
    used_secs: set[str] = set()
    for mem in graph.members:
        mat, sec = _resolve_material_section(catalog, mem.part_id)
        if mat.material_id not in used_mats:
            model.add_material(mat.material_id, mat.E, mat.G, mat.nu, mat.rho, mat.fy)
            used_mats.add(mat.material_id)
        if sec.section_id not in used_secs:
            model.add_section(sec.section_id, sec.A, sec.Iy, sec.Iz, sec.J)
            used_secs.add(sec.section_id)

    for nid, n in graph.nodes.items():
        model.add_node(nid, n.x, n.y, n.z)

    for mem in graph.members:
        mat, sec = _resolve_material_section(catalog, mem.part_id)
        model.add_member(mem.id, mem.ni, mem.nj, mat.material_id, sec.section_id)

    # Pinned base supports.
    for b in graph.base_node_ids:
        model.def_support(b, support_DX=True, support_DY=True, support_DZ=True)

    # Dead load: self weight (gravity along -Z). factor = -g ⇒ Newtons.
    model.add_member_self_weight("FZ", -G_ACCEL, case=CASE_DEAD)

    # Live load on the top working plane.
    zs = [n.z for n in graph.nodes.values()]
    max_z = max(zs)
    base_pts = np.asarray([graph.nodes[b].xyz for b in graph.base_node_ids], dtype=np.float64)
    lo = base_pts.min(axis=0)
    hi = base_pts.max(axis=0)
    footprint_area = max(0.0, float((hi[0] - lo[0]) * (hi[1] - lo[1])))
    top_nodes = [nid for nid, n in graph.nodes.items() if n.z >= max_z - 0.05]
    total_live = float(loads.live_load_kN_per_m2) * 1000.0 * footprint_area
    if top_nodes and total_live > 0.0:
        per = total_live / len(top_nodes)
        for nid in top_nodes:
            model.add_node_load(nid, "FZ", -per, case=CASE_LIVE)

    # Optional wind: lateral pressure on the windward projected area (+X).
    total_wind = 0.0
    if loads.wind_pressure_kN_per_m2 > 0.0:
        proj_area = max(0.0, float((hi[1] - lo[1]) * (max_z - lo[2])))
        total_wind = float(loads.wind_pressure_kN_per_m2) * 1000.0 * proj_area
        wind_nodes = [nid for nid, n in graph.nodes.items() if n.z > lo[2] + 0.05]
        if wind_nodes and total_wind > 0.0:
            per = total_wind / len(wind_nodes)
            for nid in wind_nodes:
                model.add_node_load(nid, "FX", per, case=CASE_WIND)

    model.add_load_combo(
        COMBO_ULS,
        {CASE_DEAD: loads.dead_load_factor, CASE_LIVE: loads.live_load_factor, CASE_WIND: loads.live_load_factor},
    )
    model.add_load_combo(COMBO_SLS, {CASE_DEAD: 1.0, CASE_LIVE: 1.0, CASE_WIND: 1.0})

    notes: list[str] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.analyze_linear(check_stability=True)
    except Exception as exc:
        return FemResult(
            ok=False,
            stable=False,
            footprint_area_m2=footprint_area,
            total_live_N=total_live,
            notes=[f"solver_unstable:{type(exc).__name__}"],
        )

    members: dict[str, MemberResult] = {}
    all_finite = True
    peak_defl = 0.0
    for mem in graph.members:
        fm = model.members[mem.id]
        try:
            ax_max = float(fm.max_axial(COMBO_ULS))
            ax_min = float(fm.min_axial(COMBO_ULS))
            m_my = max(abs(float(fm.max_moment("My", COMBO_ULS))), abs(float(fm.min_moment("My", COMBO_ULS))))
            m_mz = max(abs(float(fm.max_moment("Mz", COMBO_ULS))), abs(float(fm.min_moment("Mz", COMBO_ULS))))
            d_y = max(abs(float(fm.max_deflection("dy", COMBO_SLS))), abs(float(fm.min_deflection("dy", COMBO_SLS))))
            d_z = max(abs(float(fm.max_deflection("dz", COMBO_SLS))), abs(float(fm.min_deflection("dz", COMBO_SLS))))
        except Exception as exc:
            all_finite = False
            notes.append(f"result_read_error:{mem.id}:{type(exc).__name__}")
            continue

        moment = max(m_my, m_mz)
        defl = max(d_y, d_z)
        if not all(_finite(v) for v in (ax_max, ax_min, moment, defl)):
            all_finite = False
            continue
        axial_signed = ax_max if abs(ax_max) >= abs(ax_min) else ax_min
        members[mem.id] = MemberResult(
            member_id=mem.id,
            elem_type=mem.elem_type,
            elem_index=mem.elem_index,
            length_m=mem.length_m,
            axial_N=axial_signed,
            compression_N=max(0.0, ax_max),
            tension_N=max(0.0, -ax_min),
            moment_Nm=moment,
            deflection_m=defl,
        )
        peak_defl = max(peak_defl, defl)

    base_reactions: dict[str, float] = {}
    self_weight_total = 0.0
    if all_finite:
        for b in graph.base_node_ids:
            try:
                rz = float(model.nodes[b].RxnFZ[COMBO_ULS])
                base_reactions[b] = rz
            except Exception:
                all_finite = False
        self_weight_total = sum(
            _resolve_material_section(catalog, m.part_id)[1].A
            * _resolve_material_section(catalog, m.part_id)[0].rho
            * m.length_m
            * G_ACCEL
            for m in graph.members
        )

    stable = all_finite and bool(members)
    if not stable and "solver_unstable" not in " ".join(notes):
        notes.append("non_finite_results")

    return FemResult(
        ok=stable,
        stable=stable,
        members=members,
        base_reactions_N=base_reactions,
        total_self_weight_N=self_weight_total,
        total_live_N=total_live,
        max_deflection_m=peak_defl,
        footprint_area_m2=footprint_area,
        notes=notes,
    )
