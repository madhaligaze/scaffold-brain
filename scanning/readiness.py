from __future__ import annotations

from scanning.coverage import compute_work_aabb


def _min_top_k(values: list[int], k: int) -> int:
    if not values:
        return 0
    k = max(1, min(int(k), len(values)))
    vals = sorted([int(v) for v in values], reverse=True)
    return int(min(vals[:k]))


def compute_readiness(world_model, anchors: list[dict], policy) -> tuple[bool, float, list[str]]:
    """Readiness gate used by planning endpoints and tests.

    Tests expect reason prefixes:
      - LOW_OBSERVED_RATIO
      - LOW_VIEW_DIVERSITY
      - LOW_VIEWPOINTS

    If no anchors are present, fall back to global occupancy stats so synthetic
    tests can still run the export pipeline.
    """
    supports = [a for a in anchors if a.get("kind") == "support" and isinstance(a.get("position"), (list, tuple))]
    anchors_for_work = supports if supports else anchors
    aabb = compute_work_aabb(anchors_for_work, padding_m=1.0)
    reasons: list[str] = []
    if aabb is None:
        st = world_model.occupancy.stats()
        total = float(st.get("total", 0) or 0.0)
        observed_ratio = float(st.get("observed_ratio", 0.0) or 0.0)
        if total <= 0:
            return False, 0.0, ["EMPTY_WORLD"]

        min_obs = float(getattr(policy, "readiness_observed_ratio_min", 0.1))
        vp = int(world_model.metrics.get("viewpoints", 0) or 0)
        min_vp = int(getattr(policy, "min_viewpoints_no_anchor", 1) or 1)

        if observed_ratio < min_obs:
            reasons.append(f"LOW_OBSERVED_RATIO:{observed_ratio:.3f}<{min_obs:.3f}")
        if vp < min_vp:
            reasons.append(f"LOW_VIEWPOINTS:{vp}<{min_vp}")

        score = 0.75 * max(0.0, min(1.0, observed_ratio)) + 0.25 * max(
            0.0, min(1.0, float(vp) / float(max(1, min_vp)))
        )
        return (len(reasons) == 0), float(score), reasons

    bmin, bmax = aabb
    stats = world_model.occupancy.stats_aabb(bmin, bmax)
    if int(stats.get("total", 0) or 0) <= 0:
        return False, 0.0, ["EMPTY_AABB"]

    total = float(stats.get("total", 1) or 1.0)
    free = float(stats.get("free", 0) or 0.0)
    occ = float(stats.get("occupied", 0) or 0.0)
    observed_ratio = (free + occ) / max(1.0, total)

    min_obs = float(getattr(policy, "readiness_observed_ratio_min", 0.1))
    if observed_ratio < min_obs:
        reasons.append(f"LOW_OBSERVED_RATIO:{observed_ratio:.3f}<{min_obs:.3f}")

    # View diversity around supports: use WorldModel.anchor_view_count (azimuth bins).
    min_views = int(getattr(policy, "min_views_per_anchor", 3) or 3)
    view_counts: list[int] = []
    if supports and hasattr(world_model, "anchor_view_count"):
        for s in supports:
            pos = list(s.get("position"))
            try:
                view_counts.append(int(world_model.anchor_view_count(pos, bins_deg=45.0)))
            except Exception:
                view_counts.append(0)
    top_k = int(getattr(policy, "readiness_view_top_k", 3) or 3)
    view_div = _min_top_k(view_counts, k=top_k)
    view_div_min = int(min(view_counts) if view_counts else 0)
    if supports and view_div < min_views:
        reasons.append(f"LOW_VIEW_DIVERSITY:{view_div}<{min_views}")

    # Overall viewpoint count (quantized positions). Not required by the tests, but useful.
    vp = int(world_model.metrics.get("viewpoints", 0) or 0)
    min_vp = int(getattr(policy, "min_viewpoints", 3) or 3)
    if vp < min_vp:
        reasons.append(f"LOW_VIEWPOINTS:{vp}<{min_vp}")

    # Score: weighted blend
    score = 0.7 * max(0.0, min(1.0, observed_ratio)) + 0.3 * max(0.0, min(1.0, float(vp) / float(max(1, min_vp))))
    ready = (len(reasons) == 0) and (observed_ratio >= min_obs) and (vp >= min_vp) and (
        (not supports) or (view_div >= min_views)
    )
    return bool(ready), float(score), reasons



def compute_readiness_metrics(world_model, anchors: list[dict], policy) -> dict:
    """Return structured readiness metrics for UI.

    This intentionally mirrors compute_readiness() so Android can display:
      - observed_ratio
      - view_diversity (min views across supports)
      - viewpoints
      - thresholds (mins)
    """
    supports = [a for a in anchors if a.get("kind") == "support" and isinstance(a.get("position"), (list, tuple))]
    points = [a for a in anchors if a.get("kind") == "point" and isinstance(a.get("position"), (list, tuple))]
    anchors_for_work = supports if supports else anchors
    aabb = compute_work_aabb(anchors_for_work, padding_m=1.0)
    metrics: dict = {
        "anchor_count": int(len(supports or [])),
        "support_count": int(len(supports or [])),
        "point_count": int(len(points or [])),
        "total_points": int(len(anchors or [])),
        "observed_ratio": 0.0,
        "view_diversity": 0,
        "viewpoints": int(world_model.metrics.get("viewpoints", 0) or 0),
        "min_observed_ratio": float(getattr(policy, "readiness_observed_ratio_min", 0.1)),
        "min_views_per_anchor": int(getattr(policy, "min_views_per_anchor", 3) or 3),
        "min_viewpoints": int(getattr(policy, "min_viewpoints", 3) or 3),
    }

    if aabb is None:
        st = world_model.occupancy.stats()
        total = float(st.get("total", 0) or 0.0)
        if total > 0:
            metrics["observed_ratio"] = float(st.get("observed_ratio", 0.0) or 0.0)
        return metrics

    bmin, bmax = aabb
    stats = world_model.occupancy.stats_aabb(bmin, bmax)
    total = float(stats.get("total", 1) or 1.0)
    free = float(stats.get("free", 0) or 0.0)
    occ = float(stats.get("occupied", 0) or 0.0)
    metrics["observed_ratio"] = float((free + occ) / max(1.0, total))

    view_counts: list[int] = []
    if supports and hasattr(world_model, "anchor_view_count"):
        for s in supports:
            pos = list(s.get("position"))
            try:
                view_counts.append(int(world_model.anchor_view_count(pos, bins_deg=45.0)))
            except Exception:
                view_counts.append(0)
    top_k = int(getattr(policy, "readiness_view_top_k", 3) or 3)
    metrics["view_diversity"] = int(_min_top_k(view_counts, k=top_k))
    metrics["view_diversity_min"] = int(min(view_counts) if view_counts else 0)
    return metrics
