from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


# ───────────────────────────────────────────────────────────────────────────
# Mechanical model (SI units: metres, Newtons, Pascals, kg)
#
# Values below are *representative* of Layher-style steel scaffold tube
# (≈ Ø48.3 × 3.2 mm, S235). They are parameters of the engineering check, not
# certified design values — production deployments MUST replace allowable
# capacities with the manufacturer's approval tables (e.g. Layher Z-8.22).
# ───────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Material:
    """Linear-elastic isotropic material."""

    material_id: str
    E: float          # Young's modulus [Pa]
    G: float          # Shear modulus [Pa]
    nu: float         # Poisson's ratio [-]
    rho: float        # Density [kg/m^3]
    fy: float         # Yield strength [Pa]


@dataclass(frozen=True)
class Section:
    """Structural cross-section. For a circular hollow tube the geometric
    properties are derived from outer diameter and wall thickness."""

    section_id: str
    A: float          # Cross-sectional area [m^2]
    Iy: float         # Second moment of area about local y [m^4]
    Iz: float         # Second moment of area about local z [m^4]
    J: float          # Torsion constant [m^4]
    outer_radius_m: float  # Extreme fibre distance c for bending stress [m]

    @property
    def I_min(self) -> float:
        return min(self.Iy, self.Iz)

    @staticmethod
    def circular_tube(section_id: str, outer_diameter_m: float, wall_m: float) -> "Section":
        ro = float(outer_diameter_m) / 2.0
        ri = max(0.0, ro - float(wall_m))
        A = math.pi * (ro * ro - ri * ri)
        I = (math.pi / 4.0) * (ro ** 4 - ri ** 4)
        J = 2.0 * I
        return Section(section_id=section_id, A=A, Iy=I, Iz=I, J=J, outer_radius_m=ro)


# Default materials / sections ------------------------------------------------

STEEL_S235 = Material(material_id="steel_s235", E=200e9, G=79.3e9, nu=0.3, rho=7850.0, fy=235e6)
TUBE_48_3 = Section.circular_tube("tube_48.3x3.2", outer_diameter_m=0.0483, wall_m=0.0032)


@dataclass(frozen=True)
class Part:
    part_id: str
    name: str
    unit_weight_kg: float
    meta: dict[str, Any] | None = None
    # Engineering properties (optional — non-structural parts leave these None)
    material_id: str | None = None
    section_id: str | None = None
    # Representative member capacity for quick rule-checks / fallback when a
    # full FEM solve is unavailable. None ⇒ rely on FEM stress/buckling only.
    allowable_axial_kN: float | None = None
    is_structural: bool = False


class Catalog:
    def __init__(self) -> None:
        self.parts: dict[str, Part] = {}
        self.materials: dict[str, Material] = {}
        self.sections: dict[str, Section] = {}
        self._init_defaults()

    def _init_defaults(self) -> None:
        self.add_material(STEEL_S235)
        self.add_section(TUBE_48_3)

        steel = STEEL_S235.material_id
        tube = TUBE_48_3.section_id
        # Load-bearing tubular members carry the FEM analysis.
        self.add(Part("post", "Vertical post", 8.5, {"unit": "pcs"},
                      material_id=steel, section_id=tube, allowable_axial_kN=40.0, is_structural=True))
        self.add(Part("ledger", "Ledger (horizontal)", 5.0, {"unit": "pcs"},
                      material_id=steel, section_id=tube, allowable_axial_kN=20.0, is_structural=True))
        self.add(Part("brace", "Diagonal brace", 3.2, {"unit": "pcs"},
                      material_id=steel, section_id=tube, allowable_axial_kN=15.0, is_structural=True))
        # Non-tubular / accessory parts: weight only (not modelled as FEM members).
        self.add(Part("deck", "Deck/Plank", 12.0, {"unit": "pcs"}))
        self.add(Part("base_jack", "Base jack", 2.0, {"unit": "pcs"}))
        self.add(Part("guardrail", "Guardrail", 4.0, {"unit": "pcs"},
                      material_id=steel, section_id=tube, is_structural=True))
        self.add(Part("toe_board", "Toe board", 2.5, {"unit": "pcs"}))
        self.add(Part("ladder", "Access ladder", 7.0, {"unit": "pcs"}))

    def add(self, part: Part) -> None:
        self.parts[part.part_id] = part

    def add_material(self, material: Material) -> None:
        self.materials[material.material_id] = material

    def add_section(self, section: Section) -> None:
        self.sections[section.section_id] = section

    def get(self, part_id: str) -> Part | None:
        return self.parts.get(part_id)

    def material_for(self, part_id: str) -> Material | None:
        p = self.parts.get(part_id)
        if p is None or p.material_id is None:
            return None
        return self.materials.get(p.material_id)

    def section_for(self, part_id: str) -> Section | None:
        p = self.parts.get(part_id)
        if p is None or p.section_id is None:
            return None
        return self.sections.get(p.section_id)


DEFAULT_CATALOG = Catalog()


@dataclass(frozen=True)
class ScaffoldSpec:
    default_height_m: float = 4.0
    min_bay_m: float = 1.2
    max_bay_m: float = 3.0
    post_radius_m: float = 0.03
    ledger_radius_m: float = 0.025
    brace_radius_m: float = 0.02
    ledger_lengths_m: tuple[float, ...] = (1.2, 1.8, 2.4, 3.0)


DEFAULT_SPEC = ScaffoldSpec()


@dataclass(frozen=True)
class StructuralLoadSpec:
    """Design loads and safety factors for the structural check.

    Defaults follow common scaffold practice (EN 12811 load class 3 ≈ 2.0
    kN/m² working load) but are configurable via policy. Safety factors are
    applied to *capacity* (resistance side); loads here are characteristic.
    """

    live_load_kN_per_m2: float = 2.0      # working load on decks (load class 3)
    dead_load_factor: float = 1.35         # γ_G
    live_load_factor: float = 1.5          # γ_Q
    wind_pressure_kN_per_m2: float = 0.0   # optional lateral pressure (off by default)
    gamma_m0: float = 1.1                  # material partial factor (yield)
    gamma_buckling: float = 1.1            # buckling resistance factor
    buckling_K: float = 1.0                # effective length factor (pinned-pinned)
    deflection_limit_ratio: float = 200.0  # serviceability: span / ratio
    bearing_area_per_deck_m2: float = 2.5  # tributary deck area used to size live load


DEFAULT_LOADS = StructuralLoadSpec()
