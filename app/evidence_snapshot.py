"""Append-only, hash-chained research artifacts and one shared run deadline."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from threading import Lock
from datetime import datetime, timezone
from pathlib import Path


class BudgetExceeded(TimeoutError):
    pass


class RunBudget:
    LIMITS = {"search_requests": 4, "fetch_requests": 6}

    def __init__(self, seconds=600):
        self.started = time.monotonic()
        self.deadline = self.started + max(1, seconds)
        self.counts = {"model_calls": 0, "search_requests": 0, "fetch_requests": 0, "local_batches": 0}
        self.stage_seconds = {}
        self._lock = Lock()

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise BudgetExceeded("Shared run deadline exceeded")
        return remaining

    def consume(self, kind):
        with self._lock:
            self.remaining()
            maximum = self.LIMITS.get(kind)
            if maximum is not None and self.counts[kind] >= maximum:
                raise BudgetExceeded(f"{kind} budget exhausted")
            self.counts[kind] += 1

    def available(self, kind):
        with self._lock:
            self.remaining()
            return max(0, self.LIMITS[kind] - self.counts[kind])


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


class RunLog:
    def __init__(self, root, run_id=None):
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:12]
        self.path = Path(root) / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.entries = []

    def write(self, name, value):
        if not name.replace("_", "").isalnum():
            raise ValueError("Invalid artifact name")
        data = encoded(value)
        filename = f"{len(self.entries):04}_{name}.json"
        with (self.path / filename).open("xb") as handle:
            handle.write(data)
        entry = {"file": filename, "sha256": hashlib.sha256(data).hexdigest(),
                 "previous": self.entries[-1]["entry_sha256"] if self.entries else None}
        entry["entry_sha256"] = hashlib.sha256(encoded(entry)).hexdigest()
        with (self.path / "manifest.jsonl").open("ab") as handle:
            handle.write(json.dumps(entry, sort_keys=True).encode() + b"\n")
        self.entries.append(entry)
        return filename


def load_artifacts(root, run_id):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_id):
        raise ValueError("Invalid replay run ID")
    base = Path(root).resolve()
    path = (base / run_id).resolve()
    if path.parent != base:
        raise ValueError("Replay must stay within the configured log root")
    previous, result = None, []
    for line in (path / "manifest.jsonl").read_text().splitlines():
        entry = json.loads(line)
        digest = entry.pop("entry_sha256")
        if entry["previous"] != previous or hashlib.sha256(encoded(entry)).hexdigest() != digest:
            raise ValueError("Replay manifest integrity failed")
        target = (path / entry["file"]).resolve()
        if target.parent != path:
            raise ValueError("Replay artifact escaped run")
        data = target.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("Replay artifact hash mismatch")
        result.append((entry["file"], json.loads(data)))
        previous = digest
    if not result:
        raise ValueError("Empty replay manifest")
    return result


class ModelRunner:
    def __init__(self, llm, budget, log):
        self.llm, self.budget, self.log = llm, budget, log
        self.context_ids = {}

    def call(self, prompt, payload, schema, name, max_tokens=1800):
        self.budget.consume("model_calls")
        started = time.monotonic()
        if name.startswith("six_step_"):
            self.context_ids[name[len("six_step_"):]] = {s["evidence_id"] for s in payload.get("sources", [])}
        elif name in {"six_steps_draft", "six_steps_revision"}:
            for step in payload["steps"]:
                self.context_ids[step] = {s["evidence_id"] for s in payload.get("sources", [])}
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]
        self.log.write("model_request", {"name": name, "messages": messages, "schema": schema, "max_tokens": max_tokens})
        try:
            value = self.llm.structured(messages, schema, name, max_tokens=max_tokens, deadline=self.budget.deadline)
            self.budget.remaining()
            self.log.write("model_response", {"name": name, "value": value, "seconds": time.monotonic() - started,
                                              "usage": getattr(self.llm, "last_generation_usage", None)})
            return value
        except Exception as exc:
            self.log.write("model_error", {"name": name, "error_type": type(exc).__name__, "message": str(exc),
                                            "raw_response": getattr(exc, "raw_response", None),
                                            "seconds": time.monotonic() - started})
            raise
        finally:
            self.budget.stage_seconds[name] = self.budget.stage_seconds.get(name, 0) + time.monotonic() - started
