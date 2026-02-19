from __future__ import annotations

from typing import Any

from trace.decision_trace import add_constraint_eval, add_trace_event


def trace_solver_start(trace: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> None:
    add_trace_event(trace, "solver_start", meta or {})


def trace_candidate_grid(trace: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> None:
    add_trace_event(trace, "solver_candidate_grid", meta or {})


def trace_element_added(
    trace: list[dict[str, Any]],
    element: dict[str, Any],
    reason: str,
) -> None:
    add_trace_event(
        trace,
        "solver_element_added",
        {
            "reason": str(reason),
            "type": element.get("type"),
            "pose": element.get("pose"),
            "dims": element.get("dims"),
        },
    )


def trace_validator_result(trace: list[dict[str, Any]], valid: bool, violations: list[dict]) -> None:
    add_trace_event(trace, "solver_validated", {"valid": bool(valid), "violations": violations})


def trace_why_element(trace: list[dict[str, Any]], *, decision_id: str, element_id: str, reason: str, metrics: dict[str, Any] | None = None) -> None:
    add_constraint_eval(
        trace,
        decision_id=decision_id,
        constraint_id="element_placed",
        ok=True,
        reason=str(reason),
        metrics=metrics or {},
        element_id=str(element_id),
        severity="info",
    )
