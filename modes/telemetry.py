"""Measured server-side turn timing; never inferred progress percentages."""
import math
import statistics
import time
from collections import Counter


class TurnTelemetry:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.phase_started = self.started
        self.phase = "waiting_for_session"
        self.phases = []
        self.first_output = None
        self.first_output_kind = None

    def event(self, kind, value):
        now = self.clock()
        if kind == "status" and str(value) != self.phase:
            self.phases.append({"phase": self.phase, "seconds": round(now - self.phase_started, 6)})
            self.phase, self.phase_started = str(value), now
        elif kind == "token" and isinstance(value, str) and value.strip() and self.first_output is None:
            self.first_output, self.first_output_kind = now - self.started, "token"

    def finish(self, state):
        now = self.clock()
        status = "cancelled" if state.get("cancelled") else "error" if state.get("error") else "done"
        if self.first_output is None and status == "done" and str(state.get("response") or "").strip():
            self.first_output, self.first_output_kind = now - self.started, "complete"
        return {"schema_version": 1, "mode": state["mode"], "status": status,
                "session_id": state["session_id"], "turn_id": state["turn_id"],
                "total_seconds": round(now - self.started, 6),
                "first_output_seconds": round(self.first_output, 6) if self.first_output is not None else None,
                "first_output_kind": self.first_output_kind,
                "phases": self.phases + [{"phase": self.phase, "seconds": round(now - self.phase_started, 6)}],
                "measurement": "server_execution_events; excludes browser/network delivery"}


def summarize_turns(rows):
    """Use successful turns for latency percentiles; failures keep their count."""
    def timing(values):
        values = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v) and v >= 0)
        return {"count": len(values), "median_seconds": statistics.median(values) if values else None,
                "p95_seconds": values[max(0, math.ceil(len(values) * .95) - 1)] if values else None}

    groups = {}
    for mode in sorted({row["mode"] for row in rows}):
        sample = [row for row in rows if row["mode"] == mode]
        successful = [row for row in sample if row["status"] == "done"]
        groups[mode] = {"count": len(sample), "status_counts": dict(Counter(row["status"] for row in sample)),
                        "total": timing(row["total_seconds"] for row in successful),
                        "first_output": timing(row.get("first_output_seconds") for row in successful),
                        "first_output_kinds": dict(Counter(row.get("first_output_kind") for row in successful))}
    return {"schema_version": 1, "sample_count": len(rows), "modes": groups,
            "measurement": "observed server turns; successful-turn latency; no clinical quality inference"}
