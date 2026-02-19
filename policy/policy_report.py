from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyReport:
    """
    Small, stable JSON payload for client-side overlays and for audit/export bundles.
    Store large arrays/maps separately; keep only summary + references here.
    """

    policy_id: str
    mode: str
    ok: bool
    reason: str | None = None
    metrics: dict[str, Any] | None = None
    artifacts: dict[str, str] | None = None  # name -> path/url

    def to_json_bytes(self) -> bytes:
        return json.dumps(asdict(self), ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def write_policy_report(path: str, report: PolicyReport) -> None:
    with open(path, "wb") as f:
        f.write(report.to_json_bytes())
