"""Layher standards and BOM helpers used by backend modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List


class ComponentType(str, Enum):
    STANDARD = "standard"
    LEDGER = "ledger"
    DIAGONAL = "diagonal"


@dataclass
class BillOfMaterials:
    components: Dict[str, int] = field(default_factory=dict)

    def add_component(self, article: str, qty: int = 1) -> None:
        self.components[article] = self.components.get(article, 0) + qty

    def get_total_quantity(self) -> int:
        return sum(self.components.values())

    def get_total_weight(self) -> float:
        return sum(LayherStandards.ARTICLE_WEIGHTS.get(code, 0.0) * qty for code, qty in self.components.items())

    def get_total_cost(self) -> float:
        return sum(LayherStandards.ARTICLE_PRICES.get(code, 0.0) * qty for code, qty in self.components.items())


class LayherStandards:
    STANDARD_HEIGHTS: List[float] = [1.0, 2.0, 2.57, 3.07, 4.0]
    LEDGER_LENGTHS: List[float] = [0.73, 1.09, 1.4, 1.57, 2.07, 2.57, 3.07]
    DIAGONAL_LENGTHS: List[float] = [1.57, 2.07, 2.57, 3.07]
    DECK_LENGTHS: List[float] = [1.09, 1.57, 2.07, 2.57, 3.07]

    DECK_ARTICLES = {length: f"P-{int(length * 100)}" for length in DECK_LENGTHS}
    ARTICLE_NAMES = {**{f"S-{int(v * 100)}": f"Standard {v}m" for v in STANDARD_HEIGHTS},
                     **{f"L-{int(v * 100)}": f"Ledger {v}m" for v in LEDGER_LENGTHS},
                     **{f"D-{int(v * 100)}": f"Diagonal {v}m" for v in DIAGONAL_LENGTHS},
                     **{f"P-{int(v * 100)}": f"Deck {v}m" for v in DECK_LENGTHS}}
    ARTICLE_WEIGHTS = {k: 10.0 for k in ARTICLE_NAMES}
    ARTICLE_PRICES = {k: 30.0 for k in ARTICLE_NAMES}

    @staticmethod
    def get_nearest_standard_height(value: float) -> float:
        return min(LayherStandards.STANDARD_HEIGHTS, key=lambda x: abs(x - float(value)))

    @staticmethod
    def get_nearest_ledger_length(value: float) -> float:
        return min(LayherStandards.LEDGER_LENGTHS, key=lambda x: abs(x - float(value)))

    @staticmethod
    def get_nearest_deck_length(value: float) -> float:
        return min(LayherStandards.DECK_LENGTHS, key=lambda x: abs(x - float(value)))

    @staticmethod
    def validate_dimensions(component_type: ComponentType, value: float) -> bool:
        pools = {
            ComponentType.STANDARD: LayherStandards.STANDARD_HEIGHTS,
            ComponentType.LEDGER: LayherStandards.LEDGER_LENGTHS,
            ComponentType.DIAGONAL: LayherStandards.DIAGONAL_LENGTHS,
        }
        return any(abs(v - float(value)) < 1e-3 for v in pools[component_type])


def snap_to_layher_grid(value: float, component_type: str) -> float:
    ctype = (component_type or "").lower()
    if ctype in ("standard", "vertical"):
        return LayherStandards.get_nearest_standard_height(value)
    if ctype in ("diagonal",):
        return min(LayherStandards.DIAGONAL_LENGTHS, key=lambda x: abs(x - float(value)))
    return LayherStandards.get_nearest_ledger_length(value)


def validate_scaffold_dimensions(nodes, beams):
    """Return a list of human-readable validation errors."""
    errors = []
    for beam in beams or []:
        btype = (beam.get("type") or "ledger").lower()
        length = float(beam.get("length", 0.0) or 0.0)
        snapped = snap_to_layher_grid(length, btype)
        if abs(snapped - length) > 1e-3:
            errors.append(f"Beam {beam.get('id', '?')} has non-standard length {length}")
    return errors

