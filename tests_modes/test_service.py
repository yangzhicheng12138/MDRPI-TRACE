# Public release: shared synthetic test fixtures; private integration cases are not distributed.
"""Control-flow tests use explicit doubles; these are NOT real-model validation."""
import copy

import pytest

from modes.service import ModeService
from modes.prompts import DIMENSIONS


def review(ready=False, explained=False):
    return {"coverage": {k: "covered" if ready else "partial" for k in DIMENSIONS},
            "strengths": ["Recognizes pressure"], "gaps": [] if ready else ["Explain reassessment"],
            "material_errors": [], "ready_for_summary": ready, "reflection_explained": explained,
            "feedback": "What would you reassess, and why?"}


class ControlBackend:
    def __init__(self):
        self.reviews = [review()]
        self.assessed = []
        self.fail_assess = False
        self.fail_summary = False
        self.summary_calls = 0
        self.source_calls = 0
        self.source_questions = []
        self.messages = []
        self.performance_calls = []

    def sources(self, question, policy, log, emit):
        self.source_calls += 1
        self.source_questions.append(question)
        return [], [{"provider": "local", "status": "index_unavailable"}]

    def assess(self, state, log):
        self.assessed.append(copy.deepcopy(state))
        if self.fail_assess:
            raise RuntimeError("control-test generation failure")
        return copy.deepcopy(self.reviews[min(len(self.assessed) - 1, len(self.reviews) - 1)])

    def summary(self, state, log, emit):
        self.summary_calls += 1
        if self.fail_summary:
            raise RuntimeError("control-test summary failure")
        return "TEST DOUBLE SUMMARY"

    def chat(self, messages, sources, log, emit, runtime=None):
        self.messages = copy.deepcopy(messages)
        emit("token", "test token")
        return "TEST DOUBLE CHAT"

    def plan_chat(self, state, question, log, emit):
        tools = []
        if state["policy"] in {"local", "web"}:
            tools.append("knowledge_base")
        if any(f.get("active", True) for f in state.get("attachments", [])):
            tools.append("files")
        previous = [m["content"] for m in state["messages"] if m["role"] == "user"][:-1]
        return {"tools": tools, "query": question + (" Prior user context: " + previous[-1] if previous else ""),
                "timezone": "Asia/Shanghai", "clock_only": False, "reason": "explicit control fixture"}

    def web_sources(self, query, log, emit):
        return [], [{"provider": "web", "status": "unavailable", "error": "control-test web failure"}]

    def prepare_history(self, history, memory, sources, log, emit):
        return history, memory, {"status": "unchanged"}

    def teaching_start(self, text, sources, log, generate=False, **kwargs):
        return {"kind": "generated_case" if generate else "provided_case", "case": text, "prompt": ""}

    def performance(self, query, web_mode, replay_run_id, log, emit):
        self.performance_calls.append((query, web_mode, replay_run_id))
        return {"answer": "TEST DOUBLE PERFORMANCE", "run_id": "unit-test-only"}


@pytest.fixture
def env(tmp_path):
    backend = ControlBackend()
    service = ModeService(tmp_path, backend)
    yield service, backend
    service.close()


def case(service):
    sid = service.create("teaching", "local")["session_id"]
    state = service.turn(sid, "Development case: an adult wears an oxygen mask.")
    assert state["stage"] == "answer"
    assert not state["attempts"]
    return sid
