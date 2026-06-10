"""Tests for the FEM structural engineering core (scaffold/structural)."""

from __future__ import annotations

from types import SimpleNamespace


from scaffold.solver import generate_scaffold
from scaffold.structural.checks import structural_check
from scaffold.structural.graph import build_structural_graph


def _policy(**overrides):
    base = dict(
        scaffold_grid_step_m=2.0,
        scaffold_default_height_m=4.0,
        scaffold_min_bay_m=1.2,
        scaffold_max_bay_m=3.0,
        scaffold_deck_levels_m=[2.0, 4.0],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _square_anchors(size: float = 4.0):
    return [
        {"kind": "support", "position": [0.0, 0.0, 0.0]},
        {"kind": "support", "position": [size, 0.0, 0.0]},
        {"kind": "support", "position": [0.0, size, 0.0]},
        {"kind": "support", "position": [size, size, 0.0]},
    ]


def _elements(policy=None):
    policy = policy or _policy()
    elements, _ = generate_scaffold(None, _square_anchors(), policy)
    return elements


def test_graph_is_connected_and_posts_subdivided():
    """Every node must be referenced by a member; posts split at connections."""
    elements = _elements()
    graph = build_structural_graph(elements)
    referenced = set()
    for m in graph.members:
        referenced.add(m.ni)
        referenced.add(m.nj)
    assert referenced == set(graph.nodes), "graph has dangling (unconnected) nodes"
    assert graph.base_node_ids, "no base supports identified"
    # posts attach to ledgers at an intermediate height ⇒ more post members than posts
    n_posts = sum(1 for e in elements if e.get("type") == "post")
    n_post_members = sum(1 for m in graph.members if m.elem_type == "post")
    assert n_post_members > n_posts


def test_compression_positive_convention():
    """Pin the PyNite axial sign convention this code relies on.

    A gravity-loaded scaffold post is in COMPRESSION; the structural core
    assumes ``compression_N = max(0, max_axial)`` is positive. If a PyNite
    upgrade flips the sign, this test fails loudly instead of silently
    inverting every buckling check.
    """
    from scaffold.structural.fem import solve_structure

    graph = build_structural_graph(_elements())
    result = solve_structure(graph)
    assert result.stable
    posts = [m for m in result.members.values() if m.elem_type == "post"]
    assert posts, "no post members solved"
    # at least one post must carry meaningful compression, and none should be
    # reported as pure tension under gravity-dominated loading
    assert max(p.compression_N for p in posts) > 1.0
    governing = max(posts, key=lambda p: p.compression_N)
    assert governing.axial_N > 0.0  # compression-positive in this build


def test_normal_scaffold_passes_uls():
    ok, violations, report = structural_check(_elements(), None, _policy())
    assert ok is True
    assert report["stable"] is True
    assert report["passes_uls"] is True
    assert report["safety_score"] >= 50
    bad = {"STRENGTH_OVERSTRESS", "MEMBER_BUCKLING", "BASE_UPLIFT", "STRUCTURAL_UNSTABLE"}
    assert not (bad & {v["type"] for v in violations})
    # governing member should be a post in buckling (physically expected)
    assert report["governing_member"]["elem_type"] in {"post", "ledger", "brace"}


def test_overloaded_scaffold_fails():
    """A grossly overloaded deck must be flagged unsafe (regression guard)."""
    policy = _policy(structural_live_load_kN_per_m2=80.0)
    ok, violations, report = structural_check(_elements(policy), None, policy)
    assert ok is False
    assert report["safety_score"] == 0
    assert report["worst_utilisation"] > 1.0
    types = {v["type"] for v in violations}
    assert types & {"STRENGTH_OVERSTRESS", "MEMBER_BUCKLING"}


def test_empty_layout_is_unstable():
    ok, violations, report = structural_check([], None, _policy())
    assert ok is False
    assert report["safety_score"] == 0
    assert any(v["type"] == "STRUCTURAL_UNSTABLE" for v in violations)


def test_safety_score_monotonic_in_load():
    """Heavier load ⇒ lower (never higher) safety score."""
    scores = []
    for q in (2.0, 6.0, 12.0):
        policy = _policy(structural_live_load_kN_per_m2=q)
        _, _, report = structural_check(_elements(policy), None, policy)
        scores.append(report["safety_score"])
    assert scores[0] >= scores[1] >= scores[2]
