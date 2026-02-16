"""AI Inspector — scaffold quality checks."""
from __future__ import annotations

from typing import List, Dict, Optional, Any
from collections import defaultdict
import numpy as np


class ScaffoldInspector:
    RULES = {
        "max_load_ratio": 0.9,
        "max_vertical_spacing": 2.5,
        "min_diagonals_per_bay": 2,
        "max_cantilever": 0.5,
        "min_deck_support": 2,
    }

    def __init__(self):
        self.issues: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.suggestions: List[Dict[str, Any]] = []

    def inspect(self, elements: List[Dict[str, Any]], physics_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        self.issues, self.warnings, self.suggestions = [], [], []
        if not elements:
            return {"score": 0, "status": "CRITICAL", "issues": [{"type": "EMPTY_STRUCTURE", "severity": "CRITICAL"}], "warnings": [], "suggestions": []}

        graph = self._build_graph(elements)
        if physics_data:
            self._check_overloads(elements, physics_data)
        self._check_diagonals(graph)
        self._check_decks(graph, elements)
        self._check_vertical_continuity(elements)
        self._check_cantilevers(graph)
        self._find_simplifications(graph, elements)

        score = self._calculate_score()
        status = "GOOD" if score >= 80 else "WARNING" if score >= 60 else "CRITICAL"
        return {
            "score": score,
            "status": status,
            "issues": self.issues,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "statistics": self._get_statistics(graph, elements),
        }

    def _build_graph(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        nodes: Dict[str, Dict[str, Any]] = {}
        edges: Dict[str, Dict[str, Any]] = {}

        for i, elem in enumerate(elements):
            start = self._parse_point(elem.get("start"))
            end = self._parse_point(elem.get("end"))
            if start is None or end is None:
                continue
            s_id = self._point_to_id(start)
            e_id = self._point_to_id(end)
            nodes.setdefault(s_id, {"x": start[0], "y": start[1], "z": start[2], "connections": []})
            nodes.setdefault(e_id, {"x": end[0], "y": end[1], "z": end[2], "connections": []})
            edge_id = elem.get("id", f"e_{i}")
            edges[edge_id] = {"type": elem.get("type"), "nodes": [s_id, e_id], "start": start, "end": end}
            nodes[s_id]["connections"].append(edge_id)
            nodes[e_id]["connections"].append(edge_id)

        return {"nodes": nodes, "edges": edges, "bays": self._find_bays(edges), "levels": self._group_by_levels(nodes)}

    def _check_overloads(self, elements: List[Dict[str, Any]], physics_data: List[Dict[str, Any]]) -> None:
        by_id = {p.get("id"): p for p in physics_data}
        for elem in elements:
            p = by_id.get(elem.get("id"))
            if not p:
                continue
            ratio = p.get("load_ratio", 0)
            if ratio > self.RULES["max_load_ratio"]:
                self.issues.append({"type": "OVERLOAD", "severity": "HIGH" if ratio > 1 else "MEDIUM", "element_id": elem.get("id"), "load_ratio": ratio, "description": f"Element overloaded ({int(ratio*100)}%). Add reinforcement."})
            elif ratio > 0.7:
                self.warnings.append({"type": "HIGH_LOAD", "element_id": elem.get("id"), "load_ratio": ratio, "description": f"Element under high load ({int(ratio*100)}%)."})

    def _check_diagonals(self, graph: Dict[str, Any]) -> None:
        for bay in graph["bays"]:
            count = sum(1 for edge_id in bay["edges"] if graph["edges"][edge_id]["type"] == "diagonal")
            if count < self.RULES["min_diagonals_per_bay"]:
                self.issues.append({"type": "MISSING_DIAGONAL", "severity": "HIGH", "location": bay["center"], "description": f"Bay needs {self.RULES['min_diagonals_per_bay'] - count} more diagonal(s)."})

    def _check_decks(self, graph: Dict[str, Any], elements: List[Dict[str, Any]]) -> None:
        for deck in [e for e in elements if e.get("type") == "deck"]:
            supports = self._count_deck_supports(deck, graph)
            if supports < self.RULES["min_deck_support"]:
                self.issues.append({"type": "INSUFFICIENT_DECK_SUPPORT", "severity": "MEDIUM", "location": deck.get("start"), "description": f"Deck has only {supports} support(s)."})

    def _check_vertical_continuity(self, elements: List[Dict[str, Any]]) -> None:
        verticals = [e for e in elements if e.get("type") in ("vertical", "standard")]
        cols = defaultdict(list)
        for v in verticals:
            start = self._parse_point(v.get("start"))
            if start is None:
                continue
            cols[f"{start[0]:.1f}_{start[1]:.1f}"].append(v)
        for c in cols.values():
            c.sort(key=lambda e: self._parse_point(e.get("start"))[2])
            for i in range(len(c)-1):
                gap = self._parse_point(c[i+1].get("start"))[2] - self._parse_point(c[i].get("end"))[2]
                if gap > self.RULES["max_vertical_spacing"]:
                    self.warnings.append({"type": "VERTICAL_GAP", "gap_size": gap, "description": f"Gap {gap:.2f}m between vertical elements."})

    def _check_cantilevers(self, graph: Dict[str, Any]) -> None:
        for node in graph["nodes"].values():
            if not self._has_vertical_support(node, graph):
                dist = self._max_cantilever_distance(node, graph)
                if dist > self.RULES["max_cantilever"]:
                    self.issues.append({"type": "EXCESSIVE_CANTILEVER", "severity": "MEDIUM", "location": {"x": node["x"], "y": node["y"], "z": node["z"]}, "cantilever_length": dist, "description": f"Cantilever {dist:.2f}m exceeds limit."})

    def _find_simplifications(self, graph: Dict[str, Any], elements: List[Dict[str, Any]]) -> None:
        ledgers = [e for e in elements if e.get("type") in ("ledger", "transom")]
        by_level = defaultdict(list)
        for l in ledgers:
            p = self._parse_point(l.get("start"))
            if p is not None:
                by_level[round(p[2], 1)].append(l)
        for z, group in by_level.items():
            if len(group) > 4:
                self.suggestions.append({"type": "SIMPLIFY_LEDGERS", "location": {"z": z}, "description": f"Level z={z:.1f}m has {len(group)} ledgers.", "potential_savings": f"{len(group)//2} ledgers"})

    def _calculate_score(self) -> int:
        penalty = 0
        for issue in self.issues:
            sev = issue.get("severity", "LOW")
            penalty += 20 if sev == "CRITICAL" else 15 if sev == "HIGH" else 10 if sev == "MEDIUM" else 5
        penalty += len(self.warnings) * 5
        bonus = min(10, len(self.suggestions) * 2)
        return max(0, min(100, 100 - penalty + bonus))

    def _get_statistics(self, graph: Dict[str, Any], elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_type = defaultdict(int)
        for e in elements:
            by_type[e.get("type", "unknown")] += 1
        return {
            "total_elements": len(elements),
            "nodes": len(graph["nodes"]),
            "bays": len(graph["bays"]),
            "levels": len(graph["levels"]),
            "by_type": dict(by_type),
            "issues_count": len(self.issues),
            "warnings_count": len(self.warnings),
            "suggestions_count": len(self.suggestions),
        }

    def _parse_point(self, point: Any) -> Optional[np.ndarray]:
        if point is None:
            return None
        if isinstance(point, dict):
            return np.array([point.get("x", 0), point.get("y", 0), point.get("z", 0)], dtype=float)
        if isinstance(point, (list, tuple)) and len(point) >= 3:
            return np.array(point[:3], dtype=float)
        return None

    def _point_to_id(self, point: np.ndarray, precision: int = 2) -> str:
        return f"{point[0]:.{precision}f}_{point[1]:.{precision}f}_{point[2]:.{precision}f}"

    def _group_by_levels(self, nodes: Dict[str, Dict[str, Any]]) -> Dict[float, List[str]]:
        levels = defaultdict(list)
        for node_id, node in nodes.items():
            levels[round(node["z"], 1)].append(node_id)
        return dict(levels)

    def _find_bays(self, edges: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        ledgers_by_level = defaultdict(list)
        for edge_id, edge in edges.items():
            if edge["type"] in ("ledger", "transom"):
                ledgers_by_level[round(edge["start"][2], 1)].append(edge_id)
        bays = []
        for z, ids in ledgers_by_level.items():
            if len(ids) >= 4:
                bays.append({"level": z, "edges": ids, "center": {"x": 0, "y": 0, "z": z}})
        return bays

    def _count_deck_supports(self, deck: Dict[str, Any], graph: Dict[str, Any]) -> int:
        p = self._parse_point(deck.get("start"))
        if p is None:
            return 0
        z = p[2]
        return sum(1 for e in graph["edges"].values() if e["type"] in ("ledger", "transom") and abs(e["start"][2] - z) < 0.1)

    def _has_vertical_support(self, node: Dict[str, Any], graph: Dict[str, Any]) -> bool:
        for conn in node["connections"]:
            edge = graph["edges"].get(conn)
            if edge and edge["type"] in ("vertical", "standard") and edge["end"][2] == node["z"]:
                return True
        return False

    def _max_cantilever_distance(self, node: Dict[str, Any], graph: Dict[str, Any]) -> float:
        min_dist = float("inf")
        for other in graph["nodes"].values():
            if other["z"] < node["z"] and self._has_vertical_support(other, graph):
                dist = np.sqrt((node["x"] - other["x"])**2 + (node["y"] - other["y"])**2)
                min_dist = min(min_dist, dist)
        return 0.0 if min_dist == float("inf") else float(min_dist)
