"""Structural engineering core for scaffold designs.

Turns the loose scaffold *element list* (posts / ledgers / braces produced by
``scaffold.solver``) into a proper finite-element frame model, runs a linear
static analysis (PyNiteFEA), and evaluates engineering limit states
(strength, member buckling, base uplift / overturning, serviceability).

Public surface:
    build_structural_graph(elements)           -> StructuralGraph
    solve_structure(graph, catalog, loads)     -> FemResult
    structural_check(elements, world, policy)  -> (ok, violations, report)
"""

from __future__ import annotations

from scaffold.structural.graph import (
    Member,
    Node,
    StructuralGraph,
    build_structural_graph,
)

__all__ = [
    "Member",
    "Node",
    "StructuralGraph",
    "build_structural_graph",
]
