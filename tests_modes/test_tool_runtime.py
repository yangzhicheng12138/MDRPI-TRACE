"""Control tests for chained tools; no model or live network is simulated as real."""
import pytest

from modes.service import ModeService
from modes.tool_runtime import from_calls
from test_chat_tools import Backend as BaseBackend


class Backend(BaseBackend):
    def __init__(self):
        super().__init__()
        self.next = []
        self.observed = []
        self.pages = []

    def continue_chat(self, state, observations, *args, **kwargs):
        self.observed.append(observations)
        return from_calls({"calls": self.next.pop(0) if self.next else [], "reason": "control fixture"})

    def web_sources(self, query, *args, **kwargs):
        self.web_calls.append(query)
        return [{"evidence_id": "WEB", "source_url": "https://example.org/price", "text": "Unit price: 21", "title": "Price"}], [{"provider": "web", "status": "available"}]

    def daily_sources(self, plan, *args, **kwargs):
        self.pages.extend(plan["urls"])
        return [{"evidence_id": "PAGE", "source_url": plan["urls"][0], "text": "Two units", "title": "Details"}], [{"provider": "page", "status": "available"}]


@pytest.fixture
def env(tmp_path):
    backend = Backend(); service = ModeService(tmp_path, backend)
    yield service, backend
    service.close()


def test_search_read_calculate_uses_results_and_finishes_in_three_rounds(env):
    service, backend = env
    backend.plan.update(tools=["web_search"], query="unit price")
    backend.next = [[{"name": "read_url", "args": {"urls": ["https://example.org/price"], "query": "quantity"}}],
                    [{"name": "calculator", "args": {"expression": "21*2"}}]]
    state = service.turn(service.create()["session_id"], "Look up the price, read details, and calculate the total")
    assert backend.web_calls == ["unit price"] and backend.pages == ["https://example.org/price"]
    assert backend.observed[0][0]["sources"][0]["text"] == "Unit price: 21"
    assert state["tool_results"]["calculation"]["result"] == 42
    assert state["tool_results"]["tool_loop"]["rounds"] == 3
    assert state["tool_results"]["tool_loop"]["calls"] == 3


def test_duplicate_call_does_not_search_again(env):
    service, backend = env
    backend.plan.update(tools=["web_search"], query="unit price")
    backend.next = [[{"name": "web_search", "args": {"query": "unit price", "freshness": ""}}]]
    state = service.turn(service.create()["session_id"], "Look up the unit price")
    assert backend.web_calls == ["unit price"]
    assert state["tool_results"]["tool_loop"]["stop_reason"] == "duplicate_calls"


def test_continuation_cannot_enable_forbidden_network(env):
    service, backend = env
    backend.plan.update(tools=["knowledge_base"], query="MDRPI")
    backend.next = [[{"name": "web_search", "args": {"query": "MDRPI", "freshness": ""}}]]
    state = service.turn(service.create(policy="local")["session_id"], "Find MDRPI information")
    assert backend.source_calls == 1 and not backend.web_calls and not state["error"]


def test_empty_continuation_stops_news_without_forcing_another_search(env):
    service, backend = env
    backend.plan.update(tools=[], query="")
    state = service.turn(service.create()["session_id"], "最近有什么新闻")
    assert len(backend.web_calls) == 1 and state["tool_results"]["tool_loop"]["rounds"] == 1


def test_invalid_tool_arguments_return_observation_instead_of_executing(env):
    service, backend = env
    backend.plan = from_calls({"calls": [{"name": "calculator", "args": {"expression": 123}}], "reason": "invalid fixture"})
    state = service.turn(service.create()["session_id"], "Calculate a value")
    assert not state["error"]
    observation = state["tool_results"]["tool_observations"][0]
    assert observation["status"] == "unavailable" and "calculation" not in state["tool_results"]
