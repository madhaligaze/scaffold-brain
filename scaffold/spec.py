from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Part:
    part_id: str
    name: str
    unit_weight_kg: float
    meta: dict[str, Any] | None = None


class Catalog:
    def __init__(self) -> None:
        self.parts: dict[str, Part] = {}
        self._init_defaults()

    def _init_defaults(self) -> None:
        self.add(Part("post", "Vertical post", 8.5, {"unit": "pcs"}))
        self.add(Part("ledger", "Ledger (horizontal)", 5.0, {"unit": "pcs"}))
        self.add(Part("brace", "Diagonal brace", 3.2, {"unit": "pcs"}))
        self.add(Part("deck", "Deck/Plank", 12.0, {"unit": "pcs"}))
        self.add(Part("base_jack", "Base jack", 2.0, {"unit": "pcs"}))
        self.add(Part("guardrail", "Guardrail", 4.0, {"unit": "pcs"}))
        self.add(Part("toe_board", "Toe board", 2.5, {"unit": "pcs"}))
        self.add(Part("ladder", "Access ladder", 7.0, {"unit": "pcs"}))

    def add(self, part: Part) -> None:
        self.parts[part.part_id] = part

    def get(self, part_id: str) -> Part | None:
        return self.parts.get(part_id)


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
