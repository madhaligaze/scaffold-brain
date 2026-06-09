"""AI Inspector — scaffold quality checks."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


class ScaffoldInspector:
    """AI-инспектор качества строительных лесов."""

    RULES = {
        "max_load_ratio": 0.9,
        "max_vertical_spacing": 2.5,
        "min_diagonals_per_bay": 2,
        "max_cantilever": 0.5,
        "min_deck_support": 2,
    }

    def __init__(self) -> None:
        self.issues: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.suggestions: List[Dict[str, Any]] = []

    def inspect(
        self,
        elements: List[Dict[str, Any]],
        physics_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        self.issues = []
        self.warnings = []
        self.suggestions = []

        if not elements:
            return {
                "score": 0,
                "status": "CRITICAL",
                "issues": [{"type": "EMPTY_STRUCTURE", "severity": "CRITICAL"}],
                "warnings": [],
                "suggestions": [],
            }

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

        for elem in elements:
            start = self._parse_point(elem.get("start"))
            end = self._parse_point(elem.get("end"))
            if start is None or end is None:
                continue

            start_id = self._point_to_id(start)
            end_id = self._point_to_id(end)

            if start_id not in nodes:
                nodes[start_id] = {
                    "x": start[0],
                    "y": start[1],
                    "z": start[2],
                    "connections": [],
                }
            if end_id not in nodes:
                nodes[end_id] = {
                    "x": end[0],
                    "y": end[1],
                    "z": end[2],
                    "connections": [],
                }

            edge_id = elem.get("id", f"e_{len(edges)}")
            edges[edge_id] = {
                "type": elem.get("type"),
                "nodes": [start_id, end_id],
                "load_ratio": elem.get("load_ratio", 0),
                "start": start,
                "end": end,
            }

            nodes[start_id]["connections"].append(edge_id)
            nodes[end_id]["connections"].append(edge_id)

        levels = self._group_by_levels(nodes)
        bays = self._find_bays(edges)

        return {"nodes": nodes, "edges": edges, "bays": bays, "levels": levels}

    def _check_overloads(self, elements: List[Dict[str, Any]], physics_data: List[Dict[str, Any]]) -> None:
        physics_by_id = {item.get("id"): item for item in physics_data}

        for elem in elements:
            elem_id = elem.get("id")
            phys = physics_by_id.get(elem_id)
            if not phys:
                continue

            load_ratio = phys.get("load_ratio", 0)
            if load_ratio > self.RULES["max_load_ratio"]:
                self.issues.append(
                    {
                        "type": "OVERLOAD",
                        "severity": "HIGH" if load_ratio > 1.0 else "MEDIUM",
                        "element_id": elem_id,
                        "element_type": elem.get("type"),
                        "location": elem.get("start"),
                        "load_ratio": load_ratio,
                        "description": f"Element overloaded ({int(load_ratio * 100)}%). Add reinforcement.",
                    }
                )
            elif load_ratio > 0.7:
                self.warnings.append(
                    {
                        "type": "HIGH_LOAD",
                        "element_id": elem_id,
                        "load_ratio": load_ratio,
                        "description": f"Element under high load ({int(load_ratio * 100)}%).",
                    }
                )

    def _check_diagonals(self, graph: Dict[str, Any]) -> None:
        for bay in graph["bays"]:
            diagonal_count = sum(
                1
                for edge_id in bay["edges"]
                if graph["edges"][edge_id]["type"] == "diagonal"
            )

            if diagonal_count < self.RULES["min_diagonals_per_bay"]:
                self.issues.append(
                    {
                        "type": "MISSING_DIAGONAL",
                        "severity": "HIGH",
                        "location": bay["center"],
                        "description": f"Bay needs {self.RULES['min_diagonals_per_bay'] - diagonal_count} more diagonal(s) for stability.",
                        "suggestion": "Add X-bracing diagonals",
                    }
                )

    def _check_decks(self, graph: Dict[str, Any], elements: List[Dict[str, Any]]) -> None:
        decks = [e for e in elements if e.get("type") == "deck"]
        for deck in decks:
            supports = self._count_deck_supports(deck, graph)
            if supports < self.RULES["min_deck_support"]:
                self.issues.append(
                    {
                        "type": "INSUFFICIENT_DECK_SUPPORT",
                        "severity": "MEDIUM",
                        "location": deck.get("start"),
                        "description": f"Deck has only {supports} support(s). Minimum {self.RULES['min_deck_support']} required.",
                        "suggestion": "Add ledger support",
                    }
                )

    def _check_vertical_continuity(self, elements: List[Dict[str, Any]]) -> None:
        verticals = [e for e in elements if e.get("type") in ("vertical", "standard")]
        columns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for v in verticals:
            start = self._parse_point(v.get("start"))
            if start is None:
                continue
            columns[f"{start[0]:.1f}_{start[1]:.1f}"].append(v)

        for col_elements in columns.values():
            col_elements.sort(key=lambda e: self._parse_point(e.get("start"))[2])
            for i in range(len(col_elements) - 1):
                e1 = col_elements[i]
                e2 = col_elements[i + 1]
                end_z = self._parse_point(e1.get("end"))[2]
                start_z = self._parse_point(e2.get("start"))[2]
                gap = start_z - end_z

                if gap > self.RULES["max_vertical_spacing"]:
                    self.warnings.append(
                        {
                            "type": "VERTICAL_GAP",
                            "location": e1.get("end"),
                            "gap_size": gap,
                            "description": f"Gap {gap:.2f}m between vertical elements. May need intermediate standard.",
                        }
                    )

    def _check_cantilevers(self, graph: Dict[str, Any]) -> None:
        for node in graph["nodes"].values():
            if not self._has_vertical_support(node, graph):
                max_dist = self._max_cantilever_distance(node, graph)
                if max_dist > self.RULES["max_cantilever"]:
                    self.issues.append(
                        {
                            "type": "EXCESSIVE_CANTILEVER",
                            "severity": "MEDIUM",
                            "location": {"x": node["x"], "y": node["y"], "z": node["z"]},
                            "cantilever_length": max_dist,
                            "description": f"Cantilever {max_dist:.2f}m exceeds limit. Add vertical support.",
                            "suggestion": "Add standard below this point",
                        }
                    )

    def _find_simplifications(self, graph: Dict[str, Any], elements: List[Dict[str, Any]]) -> None:
        ledgers = [e for e in elements if e.get("type") in ("ledger", "transom")]

        by_level: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
        for ledger in ledgers:
            start = self._parse_point(ledger.get("start"))
            if start is None:
                continue
            by_level[round(start[2], 1)].append(ledger)

        for z, group in by_level.items():
            if len(group) > 4:
                self.suggestions.append(
                    {
                        "type": "SIMPLIFY_LEDGERS",
                        "location": {"z": z},
                        "description": f"Level z={z:.1f}m has {len(group)} ledgers. Consider optimizing layout.",
                        "potential_savings": f"{len(group) // 2} ledgers",
                    }
                )

        for bay in graph["bays"]:
            diagonal_count = sum(
                1
                for edge_id in bay["edges"]
                if graph["edges"][edge_id]["type"] == "diagonal"
            )
            if diagonal_count > 4:
                self.suggestions.append(
                    {
                        "type": "EXCESSIVE_DIAGONALS",
                        "location": bay["center"],
                        "description": f"Bay has {diagonal_count} diagonals. Standard X-bracing (2-4) is sufficient.",
                        "potential_savings": f"{diagonal_count - 4} diagonals",
                    }
                )

    def _calculate_score(self) -> int:
        penalty = 0
        for issue in self.issues:
            severity = issue.get("severity", "LOW")
            if severity == "CRITICAL":
                penalty += 20
            elif severity == "HIGH":
                penalty += 15
            elif severity == "MEDIUM":
                penalty += 10
            else:
                penalty += 5

        for _ in self.warnings:
            penalty += 5

        bonus = min(10, len(self.suggestions) * 2)
        score = 100 - penalty + bonus
        return max(0, min(100, score))

    def _get_statistics(self, graph: Dict[str, Any], elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        by_type: Dict[str, int] = defaultdict(int)
        for elem in elements:
            by_type[elem.get("type", "unknown")] += 1

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
        levels: Dict[float, List[str]] = defaultdict(list)
        for node_id, node in nodes.items():
            z = round(node["z"], 1)
            levels[z].append(node_id)
        return dict(levels)

    def _find_bays(self, edges: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        bays: List[Dict[str, Any]] = []
        ledgers_by_level: Dict[float, List[str]] = defaultdict(list)

        for edge_id, edge in edges.items():
            if edge["type"] in ("ledger", "transom"):
                z = round(edge["start"][2], 1)
                ledgers_by_level[z].append(edge_id)

        for z, ledger_ids in ledgers_by_level.items():
            if len(ledger_ids) >= 4:
                bays.append(
                    {
                        "level": z,
                        "edges": ledger_ids,
                        "center": {"x": 0, "y": 0, "z": z},
                    }
                )

        return bays

    def _count_deck_supports(self, deck: Dict[str, Any], graph: Dict[str, Any]) -> int:
        deck_start = self._parse_point(deck.get("start"))
        if deck_start is None:
            return 0
        deck_z = deck_start[2]

        supports = 0
        for edge in graph["edges"].values():
            if edge["type"] in ("ledger", "transom"):
                edge_z = edge["start"][2]
                if abs(edge_z - deck_z) < 0.1:
                    supports += 1
        return supports

    def _has_vertical_support(self, node: Dict[str, Any], graph: Dict[str, Any]) -> bool:
        for conn_id in node["connections"]:
            edge = graph["edges"].get(conn_id)
            if edge and edge["type"] in ("vertical", "standard"):
                if edge["end"][2] == node["z"]:
                    return True
        return False

    def _max_cantilever_distance(self, node: Dict[str, Any], graph: Dict[str, Any]) -> float:
        min_dist = float("inf")
        for other in graph["nodes"].values():
            if other["z"] < node["z"] and self._has_vertical_support(other, graph):
                dist = np.sqrt((node["x"] - other["x"]) ** 2 + (node["y"] - other["y"]) ** 2)
                min_dist = min(min_dist, dist)
        return 0.0 if min_dist == float("inf") else float(min_dist)
