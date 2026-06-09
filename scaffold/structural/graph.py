"""Assemble a finite-element frame graph from scaffold elements.

The scaffold solver emits independent elements:
  * ``post``   – vertical member, base at ``pose.position``, height ``dims.height_m``
  * ``ledger`` – horizontal member between ``dims.a`` and ``dims.b``
  * ``brace``  – diagonal member between ``dims.a`` and ``dims.b``

These do not share node objects, and ledgers/braces typically connect to a post
at an *intermediate* height rather than at the post's two endpoints. A correct
frame model therefore requires:
  1. merging coincident endpoints into shared nodes (within a tolerance), and
  2. subdividing each post at every height where another member connects to its
     vertical axis, so the assembly is structurally continuous (not a mechanism).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

# Member element types that carry load and are modelled as FEM frame members.
STRUCTURAL_TYPES = ("post", "ledger", "brace", "guardrail")

DEFAULT_MERGE_TOL_M = 0.03   # endpoints closer than this collapse to one node
DEFAULT_XY_TOL_M = 0.05      # column matching tolerance for post subdivision


@dataclass
class Node:
    id: str
    x: float
    y: float
    z: float

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Member:
    id: str
    ni: str
    nj: str
    elem_type: str
    part_id: str
    elem_index: int          # index into the source element list (-1 for synthetic)
    length_m: float


@dataclass
class StructuralGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    members: list[Member] = field(default_factory=list)
    base_node_ids: list[str] = field(default_factory=list)
    min_z: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def node_xyz(self, node_id: str) -> tuple[float, float, float]:
        return self.nodes[node_id].xyz


def _type(e: dict[str, Any]) -> str:
    return str(e.get("type") or e.get("kind") or "unknown")


def _vec(p: Any) -> np.ndarray | None:
    if isinstance(p, (list, tuple)) and len(p) == 3:
        return np.asarray(p, dtype=np.float64).reshape(3)
    return None


def iter_member_segments(
    elements: list[dict[str, Any]],
) -> Iterable[tuple[int, str, np.ndarray, np.ndarray]]:
    """Yield ``(elem_index, elem_type, a, b)`` for every load-bearing member."""
    for idx, e in enumerate(elements or []):
        t = _type(e)
        if t not in STRUCTURAL_TYPES:
            continue
        dims = e.get("dims") or {}
        pose = e.get("pose") or {}
        if t == "post":
            base = _vec(pose.get("position", pose.get("pos")))
            if base is None:
                continue
            h = float(dims.get("height_m") or dims.get("height") or dims.get("z") or 0.0)
            if h <= 1e-6:
                continue
            yield idx, t, base, base + np.asarray([0.0, 0.0, h])
        else:  # ledger / brace / guardrail
            a = _vec(dims.get("a"))
            b = _vec(dims.get("b"))
            if a is None or b is None:
                # fall back to pose.position + length along nothing — skip if unusable
                continue
            if float(np.linalg.norm(b - a)) <= 1e-6:
                continue
            yield idx, t, a, b


class _NodeRegistry:
    """Spatial-hash node de-duplication within a merge tolerance."""

    def __init__(self, tol: float) -> None:
        self.tol = float(tol)
        self._cells: dict[tuple[int, int, int], list[str]] = {}
        self.nodes: dict[str, Node] = {}
        self._counter = 0

    def _cell(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        t = self.tol
        return (int(round(x / t)), int(round(y / t)), int(round(z / t)))

    def get_or_create(self, p: np.ndarray) -> str:
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        cx, cy, cz = self._cell(x, y, z)
        best_id: str | None = None
        best_d = self.tol
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for nid in self._cells.get((cx + dx, cy + dy, cz + dz), ()):
                        n = self.nodes[nid]
                        d = float(np.hypot(np.hypot(n.x - x, n.y - y), n.z - z))
                        if d <= best_d:
                            best_d = d
                            best_id = nid
        if best_id is not None:
            return best_id
        nid = f"N{self._counter}"
        self._counter += 1
        self.nodes[nid] = Node(id=nid, x=x, y=y, z=z)
        self._cells.setdefault((cx, cy, cz), []).append(nid)
        return nid


def build_structural_graph(
    elements: list[dict[str, Any]],
    *,
    merge_tol_m: float = DEFAULT_MERGE_TOL_M,
    xy_tol_m: float = DEFAULT_XY_TOL_M,
    part_of: dict[str, str] | None = None,
) -> StructuralGraph:
    """Build a connected frame graph from scaffold elements.

    ``part_of`` optionally maps elem_type → catalog part_id (defaults to the
    elem_type itself, which matches the default catalog keys).
    """
    reg = _NodeRegistry(merge_tol_m)
    segments = list(iter_member_segments(elements))

    # Pass 1: register every member endpoint so connection points exist as nodes.
    seg_nodes: list[tuple[int, str, str, str]] = []  # (elem_index, type, ni, nj)
    posts: list[tuple[int, str, str]] = []           # (elem_index, base_node, top_node)
    for idx, t, a, b in segments:
        na = reg.get_or_create(a)
        nb = reg.get_or_create(b)
        if t == "post":
            posts.append((idx, na, nb))
        else:
            seg_nodes.append((idx, t, na, nb))

    graph = StructuralGraph(nodes=reg.nodes)
    member_counter = 0

    def _add_member(ni: str, nj: str, t: str, idx: int) -> None:
        nonlocal member_counter
        if ni == nj:
            return
        a = np.asarray(reg.nodes[ni].xyz)
        b = np.asarray(reg.nodes[nj].xyz)
        L = float(np.linalg.norm(b - a))
        pid = (part_of or {}).get(t, t)
        graph.members.append(
            Member(id=f"M{member_counter}", ni=ni, nj=nj, elem_type=t, part_id=pid,
                   elem_index=idx, length_m=L)
        )
        member_counter += 1

    # Pass 2: add horizontal/diagonal members directly.
    for idx, t, ni, nj in seg_nodes:
        _add_member(ni, nj, t, idx)

    # Pass 3: subdivide each post through every node lying on its vertical axis.
    # Group existing nodes into (x,y) columns for fast lookup.
    columns: dict[tuple[int, int], list[str]] = {}
    for nid, n in reg.nodes.items():
        key = (int(round(n.x / xy_tol_m)), int(round(n.y / xy_tol_m)))
        columns.setdefault(key, []).append(nid)

    for idx, base_id, top_id in posts:
        base = reg.nodes[base_id]
        top = reg.nodes[top_id]
        key = (int(round(base.x / xy_tol_m)), int(round(base.y / xy_tol_m)))
        z_lo, z_hi = sorted((base.z, top.z))
        on_axis: list[str] = []
        for nid in columns.get(key, ()):
            n = reg.nodes[nid]
            # same vertical line and within the post's height span (incl. ends)
            if abs(n.x - base.x) <= xy_tol_m and abs(n.y - base.y) <= xy_tol_m:
                if z_lo - merge_tol_m <= n.z <= z_hi + merge_tol_m:
                    on_axis.append(nid)
        on_axis = sorted(set(on_axis), key=lambda i: reg.nodes[i].z)
        if len(on_axis) < 2:
            on_axis = [base_id, top_id]
        for k in range(len(on_axis) - 1):
            _add_member(on_axis[k], on_axis[k + 1], "post", idx)

    # Supports: nodes on the lowest horizontal plane (base jacks on the ground).
    if reg.nodes:
        graph.min_z = min(n.z for n in reg.nodes.values())
        graph.base_node_ids = [
            nid for nid, n in reg.nodes.items() if n.z <= graph.min_z + merge_tol_m
        ]

    if not graph.members:
        graph.warnings.append("no_structural_members")
    if len(graph.base_node_ids) < 1:
        graph.warnings.append("no_base_supports")

    return graph
