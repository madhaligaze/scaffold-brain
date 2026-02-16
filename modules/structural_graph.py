"""StructuralGraph — живой граф конструкции в памяти сессии."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Dict, List

try:
    import networkx as nx

    NX_AVAILABLE = True
except ImportError:
    nx = None
    NX_AVAILABLE = False


@dataclass
class StructuralElement:
    id: str
    type: str
    data: Dict
    load_ratio: float = 0.0
    is_fixed: bool = False
    color: str = "green"


class StructuralGraph:
    def __init__(self):
        self._g = nx.Graph() if NX_AVAILABLE else None
        self._nodes: Dict[str, StructuralElement] = {}
        self._beams: Dict[str, StructuralElement] = {}
        self._version: int = 0

    def load_from_variant(self, variant: Dict) -> None:
        self._nodes.clear()
        self._beams.clear()
        if self._g is not None:
            self._g.clear()

        for n in variant.get("nodes", []):
            elem = StructuralElement(
                id=n["id"],
                type="node",
                data=n,
                is_fixed=n.get("z", 0) <= 0.05,
            )
            self._nodes[n["id"]] = elem
            if self._g is not None:
                self._g.add_node(n["id"], **n)

        for b in variant.get("beams", []):
            lr = b.get("load_ratio", 0.0)
            elem = StructuralElement(
                id=b["id"],
                type=b.get("type", "ledger"),
                data=b,
                load_ratio=lr,
                color=self._ratio_to_color(lr),
            )
            self._beams[b["id"]] = elem
            if self._g is not None:
                self._g.add_edge(b["start"], b["end"], id=b["id"], beam=elem)

        self._version += 1

    def remove_element(self, element_id: str) -> Dict:
        affected: List[str] = []

        if element_id in self._beams:
            beam = self._beams.pop(element_id)
            if self._g is not None and self._g.has_edge(beam.data["start"], beam.data["end"]):
                self._g.remove_edge(beam.data["start"], beam.data["end"])
            affected = self._get_adjacent_beams(beam.data["start"], beam.data["end"])

        elif element_id in self._nodes:
            self._nodes.pop(element_id)
            connected = [
                bid
                for bid, b in self._beams.items()
                if b.data.get("start") == element_id or b.data.get("end") == element_id
            ]
            for bid in connected:
                self._beams.pop(bid, None)
                affected.append(bid)
            if self._g is not None and element_id in self._g:
                self._g.remove_node(element_id)

        self._version += 1
        self._recalculate_loads(affected)
        return {
            "removed": element_id,
            "affected": affected,
            "heatmap": self.get_heatmap(),
            "is_stable": self._check_stability(),
        }

    def add_beam(self, beam_data: Dict) -> Dict:
        bid = beam_data.get("id", f"beam_{uuid.uuid4().hex[:6]}")
        beam_data["id"] = bid
        elem = StructuralElement(id=bid, type=beam_data.get("type", "ledger"), data=beam_data)
        self._beams[bid] = elem

        if self._g is not None:
            s, e = beam_data["start"], beam_data["end"]
            if s in self._g and e in self._g:
                self._g.add_edge(s, e, id=bid, beam=elem)

        self._version += 1
        affected = self._get_adjacent_beams(beam_data["start"], beam_data["end"])
        self._recalculate_loads(affected)

        return {"added": bid, "heatmap": self.get_heatmap(), "is_stable": self._check_stability()}

    def get_heatmap(self) -> List[Dict]:
        return [{"id": b.id, "color": b.color, "load_ratio": b.load_ratio} for b in self._beams.values()]

    def get_nodes(self) -> List[Dict]:
        return [n.data for n in self._nodes.values()]

    def get_beams(self) -> List[Dict]:
        return [b.data for b in self._beams.values()]

    def get_summary(self) -> Dict:
        return {
            "nodes": len(self._nodes),
            "beams": len(self._beams),
            "version": self._version,
            "stable": self._check_stability(),
            "critical_count": sum(1 for b in self._beams.values() if b.color == "red"),
        }

    def _recalculate_loads(self, beam_ids: List[str]) -> None:
        if not beam_ids:
            return

        for bid in beam_ids:
            beam = self._beams.get(bid)
            if not beam:
                continue

            neighbors = self._get_adjacent_beams(beam.data.get("start", ""), beam.data.get("end", ""))
            if neighbors:
                avg_neighbor_load = sum(self._beams[nid].load_ratio for nid in neighbors if nid in self._beams) / len(neighbors)
                new_ratio = min(beam.load_ratio + avg_neighbor_load * 0.3, 1.5)
            else:
                new_ratio = min(beam.load_ratio * 1.2, 1.5)

            beam.load_ratio = new_ratio
            beam.color = self._ratio_to_color(new_ratio)

    def _get_adjacent_beams(self, node_start: str, node_end: str) -> List[str]:
        result = []
        for bid, b in self._beams.items():
            if b.data.get("start") in (node_start, node_end) or b.data.get("end") in (node_start, node_end):
                result.append(bid)
        return result

    def _check_stability(self) -> bool:
        has_fixed = any(n.is_fixed for n in self._nodes.values())
        if not has_fixed:
            return False
        if self._g is not None and len(self._g) > 0:
            return nx.is_connected(self._g)
        return True

    @staticmethod
    def _ratio_to_color(ratio: float) -> str:
        if ratio >= 0.9:
            return "red"
        if ratio >= 0.6:
            return "yellow"
        return "green"
