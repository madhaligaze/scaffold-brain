from __future__ import annotations

from policy.unknown_space import (
    UnknownPolicyConfig,
    UnknownSampler,
    evaluate_unknown_policy,
)


def test_unknown_policy_allow_always_ok() -> None:
    cfg = UnknownPolicyConfig(mode="allow")
    sampler = UnknownSampler(lambda _p: "unknown")
    d = evaluate_unknown_policy(cfg, sampler, critical_points_w=[(0.0, 0.0, 0.0)], decision_id="d1", trace=[])
    assert d.ok is True
    assert d.required_clearance_m == 0.0


def test_unknown_policy_forbid_blocks_when_all_unknown() -> None:
    cfg = UnknownPolicyConfig(mode="forbid", max_unknown_fraction=0.01, samples_per_region=64, forbid_radius_m=0.5)
    sampler = UnknownSampler(lambda _p: "unknown")
    d = evaluate_unknown_policy(cfg, sampler, critical_points_w=[(0.0, 0.0, 0.0)], decision_id="d2", trace=[])
    assert d.ok is False


def test_unknown_policy_forbid_passes_when_all_known() -> None:
    cfg = UnknownPolicyConfig(mode="forbid", max_unknown_fraction=0.01, samples_per_region=64, forbid_radius_m=0.5)
    sampler = UnknownSampler(lambda _p: "free")
    d = evaluate_unknown_policy(cfg, sampler, critical_points_w=[(0.0, 0.0, 0.0)], decision_id="d3", trace=[])
    assert d.ok is True


def test_unknown_policy_buffer_adds_clearance_when_unknown_high() -> None:
    cfg = UnknownPolicyConfig(
        mode="buffer",
        max_unknown_fraction=0.01,
        samples_per_region=64,
        buffer_radius_m=0.5,
        buffer_clearance_m=0.2,
    )
    sampler = UnknownSampler(lambda _p: "unknown")
    d = evaluate_unknown_policy(cfg, sampler, critical_points_w=[(0.0, 0.0, 0.0)], decision_id="d4", trace=[])
    assert d.ok is True
    assert d.required_clearance_m == 0.2


def test_unknown_policy_trace_is_written() -> None:
    cfg = UnknownPolicyConfig(mode="buffer", max_unknown_fraction=0.01, samples_per_region=16)
    sampler = UnknownSampler(lambda _p: "unknown")
    trace: list[dict] = []
    _ = evaluate_unknown_policy(cfg, sampler, critical_points_w=[(0.0, 0.0, 0.0)], decision_id="d5", trace=trace)
    assert any(ev.get("event") == "constraint_eval" for ev in trace)
