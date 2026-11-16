"""Control-flow tests; tool doubles are separate from real-model validation."""
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from modes.chat_tools import simple_clock_question, source_notes, PlannedWebQuery, require_current_news_search
from modes.service import ModeService
from modes.server import create_app
from test_service import ControlBackend


class Backend(ControlBackend):
    def __init__(self):
        super().__init__()
        self.plan = {"tools": [], "query": "test query", "timezone": "Asia/Shanghai", "clock_only": False, "reason": "test plan"}
        self.plans = 0
        self.web_calls = []
        self.web_freshness = []
        self.runtime = {}

    def plan_chat(self, *args):
        self.plans += 1
        return dict(self.plan)

    def chat(self, messages, sources, log, emit, runtime=None):
        self.runtime = runtime
        return super().chat(messages, sources, log, emit, runtime)

    def web_sources(self, query, log, emit, **kwargs):
        self.web_calls.append(query)
        self.web_freshness.append(kwargs.get("freshness", ""))
        return super().web_sources(query, log, emit)


@pytest.fixture
def env(tmp_path):
    backend = Backend(); service = ModeService(tmp_path, backend)
    yield service, backend
    service.close()


@pytest.mark.parametrize("question", ["你好你知道现在的时间吗", "现在几点了？", "今天是几号？", "What time is it now?", "What's today's date?"])
def test_actual_clock_without_model_or_retrieval_even_with_web_permission(env, question):
    service, backend = env
    sid = service.create(policy="web")["session_id"]
    before = datetime.now(ZoneInfo("Asia/Shanghai"))
    state = service.turn(sid, question)
    after = datetime.now(ZoneInfo("Asia/Shanghai"))
    observed = datetime.fromisoformat(state["tool_results"]["requested_clock"]["datetime"])
    assert before.timestamp() - 1 <= observed.timestamp() <= after.timestamp() + 1
    assert not backend.plans and not backend.source_calls and not backend.web_calls and not backend.messages
    assert not state["sources"] and not source_notes(state["source_events"])
    assert state["chat_plan"]["tools"] == ["clock"] and "+08:00" in state["response"]


@pytest.mark.parametrize("question", ["把‘现在几点’翻译成英文", "Explain how to obtain the current time in Python", "现在几点？再写一首诗", "What time should I assess skin after NIV?"])
def test_time_words_in_another_task_do_not_activate_fast_path(question):
    assert not simple_clock_question(question)


def test_open_chat_does_not_inherit_knowledge_or_file_retrieval(env):
    service, backend = env
    sid = service.create()["session_id"]
    service.upload(sid, "old.txt", b"A prior MDRPI teaching note.")
    state = service.turn(sid, "Write a cheerful birthday message.")
    assert state["policy"] == "auto"
    assert state["chat_plan"]["tools"] == [] and not state["sources_loaded"]
    assert not state["sources"] and not state["source_events"]
    assert not backend.source_calls and not backend.web_calls


def test_local_and_general_web_are_independent_tools(env):
    service, backend = env
    sid = service.create()["session_id"]
    backend.plan.update(tools=["web_search"], query="Python current release official")
    state = service.turn(sid, "Look up the current Python release")
    assert not backend.source_calls and backend.web_calls == ["Python current release official"]
    assert backend.runtime["tool_outcomes"][0]["status"] == "unavailable"
    assert "网页搜索暂不可用" in source_notes(state["source_events"])[0]
    backend.plan.update(tools=["knowledge_base"], query="nasal mask pressure injury")
    service.turn(sid, "MDRPI prevention from our knowledge base")
    assert backend.source_calls == 1 and len(backend.web_calls) == 1


@pytest.mark.parametrize("question", ["今天有什么新闻呢", "最近有什么新闻什么的", "What's in the latest news?"])
def test_current_news_search_runs_when_model_selects_no_tools(env, question):
    service, backend = env
    backend.plan.update(tools=[], query="")
    state = service.turn(service.create()["session_id"], question)
    assert state["chat_plan"]["tools"] == ["web_search"]
    assert len(backend.web_calls) == 1 and question in backend.web_calls[0]
    assert state["tool_results"]["selected_tools"] == ["web_search"]
    assert not backend.source_calls
    if "今天" in question:
        assert backend.web_freshness == ["d"]


@pytest.mark.parametrize("policy", ["none", "local"])
def test_current_news_does_not_override_disabled_network(env, policy):
    service, backend = env
    backend.plan.update(tools=[], query="")
    state = service.turn(service.create(policy=policy)["session_id"], "今天有什么新闻呢")
    assert "web_search" in state["chat_plan"]["blocked_tools"]
    assert not backend.web_calls and not state["sources"]


@pytest.mark.parametrize("question", [
    "把‘今天有什么新闻呢’翻译成英文", 'Translate "What is the latest news?" into Chinese.',
    "写一则虚构的今日新闻", "解释新闻的定义，以及新闻历史。", "今天天气怎么样？",
    "别联网，介绍一下最近新闻常见的写作方式。", "Without browsing the web, discuss recent news writing.",
])
def test_news_rule_does_not_force_search_for_other_tasks(env, question):
    service, backend = env
    backend.plan.update(tools=[], query="")
    state = service.turn(service.create()["session_id"], question)
    assert not backend.web_calls and state["chat_plan"]["tools"] == []


def test_news_rule_keeps_other_tools_and_an_existing_search_plan():
    plan = {"tools": ["weather"], "query": "Chengdu weather", "location": "成都", "freshness": ""}
    amended = require_current_news_search(plan, "告诉我成都天气和今天的新闻")
    assert amended["tools"] == ["weather", "web_search"] and amended["location"] == "成都"
    assert plan["tools"] == ["weather"]
    search = {"tools": ["web_search"], "query": "AI latest news", "freshness": "w"}
    assert require_current_news_search(search, "最近有什么AI新闻") == search


def test_current_news_still_searches_after_planner_failure(env):
    service, backend = env
    def fail(*args):
        raise RuntimeError("test planner unavailable")
    backend.plan_chat = fail
    state = service.turn(service.create()["session_id"], "今天有什么新闻呢")
    assert len(backend.web_calls) == 1 and state["chat_plan"]["planner"] == "failed"
    assert state["source_events"][0]["fallback_tools"] == ["web_search"]
    assert not any("未使用外部检索" in note for note in source_notes(state["source_events"]))


def test_clock_after_failed_web_clears_previous_source_status(env):
    service, backend = env
    sid = service.create()["session_id"]
    backend.plan.update(tools=["web_search"])
    assert source_notes(service.turn(sid, "Search the web")["source_events"])
    state = service.turn(sid, "现在的时间是多少")
    assert not source_notes(state["source_events"]) and state["context_event"] == {}
    assert not state["sources_loaded"] and len(backend.web_calls) == 1


def test_clock_timezone_from_model_and_mixed_request_keep_requested_tasks(env):
    service, backend = env
    sid = service.create()["session_id"]
    backend.plan.update(tools=["clock"], timezone="America/New_York", clock_only=True)
    state = service.turn(sid, "纽约现在几点？")
    assert backend.runtime["requested_clock"]["timezone"] == "America/New_York" and backend.messages
    # The small model can mistakenly classify a mixed task as clock-only. A tool
    # result must still go to the answer model, not discard the rest of the task.
    state = service.turn(sid, "Tell me the time in New York and write a poem.")
    assert backend.messages[-1]["content"].endswith("write a poem.")
    assert backend.runtime["requested_clock"]["timezone"] == "America/New_York"


def test_disabled_tools_are_not_called(env):
    service, backend = env
    sid = service.create(policy="none")["session_id"]
    backend.plan.update(tools=["web_search", "knowledge_base", "files"])
    state = service.turn(sid, "A request with external tools disabled")
    assert state["chat_plan"]["tools"] == []
    assert set(state["tool_results"]["blocked_tools"]) == {"web_search", "knowledge_base", "files"}
    assert not backend.web_calls and not backend.source_calls


def test_planner_failure_still_allows_normal_answer_without_claiming_retrieval(env):
    service, backend = env
    def fail(*args):
        raise RuntimeError("test planner unavailable")
    backend.plan_chat = fail
    state = service.turn(service.create()["session_id"], "Write a short greeting")
    assert state["response"] == "TEST DOUBLE CHAT" and not state["error"]
    assert state["chat_plan"]["planner"] == "failed" and not backend.web_calls
    assert backend.runtime["tool_outcomes"][0]["provider"] == "router"


def test_timezone_setting_persists_and_api_default_is_auto(tmp_path):
    service = ModeService(tmp_path, Backend())
    with TestClient(create_app(service, with_ui=False)) as client:
        state = client.post('/api/sessions', json={}).json();sid=state['session_id']
        assert state['policy'] == 'auto' and state['timezone'] == 'Asia/Shanghai'
        assert client.post('/api/sessions', json={'timezone':'invalid/zone'}).status_code == 422
        service.configure(sid, 'auto', timezone='UTC')
    service.close()
    restored = ModeService(tmp_path, Backend())
    assert restored.get(sid)['timezone'] == 'UTC'
    assert 'UTC+00:00' in restored.turn(sid, 'What time is it?')['response']
    restored.close()


def test_general_query_is_reused_without_mdrpi_query_regeneration():
    planner = PlannedWebQuery("Python latest release October 2026")
    result = planner.call("MDRPI-specific prompt ignored", {}, {}, "web_query_plan")
    assert result["search_query"] == "Python latest release October 2026"
    assert "MDRPI" not in result["verification_focus"]


def test_ordinary_code_is_not_treated_as_citation_numbers(env):
    from modes.backend import render_references
    service, backend = env
    text = "Use `items[1]`.\n```python\nvalues = [1, 2]\nprint(items[1])\n```"
    assert render_references(text, []) == text
    # The next model call must see the original indices in the prior answer.
    sid = service.create()["session_id"]
    original = backend.chat
    backend.chat = lambda *args, **kwargs: text
    service.turn(sid, "Show a list-indexing example")
    backend.chat = original
    service.turn(sid, "Explain the expression you just wrote")
    assert backend.messages[-2]["content"] == text


def test_document_citations_do_not_rewrite_code_blocks():
    from modes.backend import render_references, without_reference_context
    sources = [{"source_type": "user_upload", "title": "unused.txt", "text": "Other material"},
               {"source_type": "user_upload", "title": "syntax.txt", "text": "Use the third element"}]
    text = "Use `items[2]`, as in the file [2].\n```python\nprint(items[2])\n```"
    result = render_references(text, sources)
    assert result.startswith(text.replace("file [2]", "file [1]")) and "syntax.txt" in result
    assert without_reference_context(result) == text.replace("file [2]", "file ")


def test_web_helper_uses_existing_search_without_local_retrieval(tmp_path, monkeypatch):
    from modes.backend import Backend as RealBackend, TurnLog
    import modes.daily_tools
    captured = {}
    def search(query, log, **options):
        captured.update(query=query, options=options)
        return [], [{'provider': 'web', 'status': 'no_results'}]
    monkeypatch.setattr(modes.daily_tools, 'search_web', search)
    path = tmp_path/'config.yaml';path.write_text('web_search:\n  provider: ddgs\n')
    sources, events = RealBackend(path).web_sources('Python release', TurnLog(tmp_path/'logs'), lambda *_:None, freshness='w')
    assert captured['query'] == 'Python release'
    assert captured['options']['freshness'] == 'w'
    assert sources == [] and events[-1]['status'] == 'no_results'
