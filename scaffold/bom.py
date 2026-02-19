from __future__ import annotations

from typing import Any

from scaffold.spec import DEFAULT_CATALOG


def bom_from_elements(elements: list[dict[str, Any]], catalog=DEFAULT_CATALOG) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for e in elements or []:
        t = str(e.get("type", "unknown"))
        counts[t] = int(counts.get(t, 0) + 1)

    lines: list[dict[str, Any]] = []
    total_kg = 0.0
    for part_id, qty in sorted(counts.items(), key=lambda x: x[0]):
        part = catalog.get(part_id) if catalog is not None else None
        unit_w = float(part.unit_weight_kg) if part is not None else 0.0
        line_w = unit_w * float(qty)
        total_kg += line_w
        lines.append({
            "part_id": part_id,
            "name": part.name if part is not None else part_id,
            "qty": int(qty),
            "unit_weight_kg": unit_w,
            "line_weight_kg": float(line_w),
            "meta": part.meta if part is not None else None,
        })

    return {"lines": lines, "totals": {"total_weight_kg": float(total_kg), "unique_parts": int(len(lines))}}
