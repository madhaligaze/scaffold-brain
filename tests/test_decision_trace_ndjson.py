import json

from trace.decision_trace import add_trace_event, trace_to_ndjson_bytes


def test_trace_ndjson_is_valid_json_per_line():
    t = []
    add_trace_event(t, "a", {"x": 1})
    add_trace_event(t, "b", {"y": "z"}, level="warn")
    data = trace_to_ndjson_bytes(t).decode("utf-8").strip().splitlines()
    assert len(data) == 2
    a = json.loads(data[0])
    b = json.loads(data[1])
    assert a["event"] == "a"
    assert b["level"] == "warn"
    assert "id" in a and "ts" in a
