"""Planner routing, bounded evidence acquisition and request isolation contracts."""
from copy import deepcopy
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from app.api import create_app
from app.config import AppConfig
from app.deepseek_search import DeepSeekSearch, SEARCH_URL
from app.evidence_snapshot import BudgetExceeded, ModelRunner, RunBudget, RunLog, load_artifacts
from app.local_llm import InvalidModelOutput, ModelUnavailable
from app.planned_search import PlannedDeepSeekSearch
from app.search_planner import DeepSeekPlanner, planning_runner
from app.six_step_schema import obj, string
from app.web_search import create_web_search
from modes.tasks import TurnCancelled
from test_deepseek_search import BODY, ROW, transport
from test_general_web_search import QUERY
from test_rag import config as rag_config, pipeline


SCHEMA = obj({"search_query": string(128), "verification_focus": string(400)})
PLAN = {"search_query": QUERY, "verification_focus": "Check device fit and skin assessment."}


def config(tmp_path, **options):
    return AppConfig(tmp_path, {"search_planner": {"provider": "deepseek", **options},
        "web_search": {"provider": "deepseek", "deepseek_fallback_provider": ""}})


@pytest.fixture(autouse=True)
def controlled_credentials(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "controlled-planner-key")
    monkeypatch.setattr("app.deepseek_search.validate_public_url", lambda *a, **k: None)


def response_transport(calls, *, content=None, status=200, finish="stop"):
    def handle(request):
        calls.append(request)
        return httpx.Response(status, json={"choices": [{"finish_reason": finish,
            "message": {"content": json.dumps(PLAN) if content is None else content}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}})
    return httpx.MockTransport(handle)


class LocalLLM:
    def __init__(self):
        self.names = []

    def structured(self, messages, schema, name, **kwargs):
        self.names.append(name)
        return deepcopy(PLAN)


def test_default_is_original_local_runner_with_zero_cloud_requests(tmp_path):
    local = ModelRunner(LocalLLM(), RunBudget(), RunLog(tmp_path))
    cfg = AppConfig(tmp_path, {})
    assert planning_runner(cfg, local) is local
    assert local.call("JSON", {}, SCHEMA, "web_query_plan") == PLAN
    assert local.llm.names == ["web_query_plan"]


def test_cloud_planning_is_json_validated_and_answer_or_grading_stays_local(tmp_path):
    calls, log, llm = [], RunLog(tmp_path), LocalLLM()
    local = ModelRunner(llm, RunBudget(), log)
    runner = planning_runner(config(tmp_path), local, transport=response_transport(calls))
    assert runner.call("Return JSON", {}, SCHEMA, "web_query_plan") == PLAN
    for name in ("teaching_assess", "six_steps_draft", "chat_answer"):
        runner.call("Local", {"steps": []}, SCHEMA, name)
    assert llm.names == ["teaching_assess", "six_steps_draft", "chat_answer"]
    assert len(calls) == 1 and runner.budget.counts["model_calls"] == 4
    payload = json.loads(calls[0].content)
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}
    assert "maxLength" in payload["messages"][0]["content"]
    assert runner.providers_used == ["deepseek"]
    assert "controlled-planner-key" not in json.dumps(load_artifacts(tmp_path, log.run_id))


@pytest.mark.parametrize("kwargs", [
    {"status": 401}, {"status": 402}, {"status": 429}, {"status": 500},
    {"content": "not json"}, {"content": '{"search_query":"a","search_query":"b"}'},
    {"content": json.dumps({**PLAN, "search_query": "x" * 129})}, {"finish": "length"},
])
def test_failure_falls_back_once_with_both_attempts_accounted(tmp_path, kwargs):
    calls, llm = [], LocalLLM()
    local = ModelRunner(llm, RunBudget(), RunLog(tmp_path))
    runner = planning_runner(config(tmp_path), local, transport=response_transport(calls, **kwargs))
    assert runner.call("JSON", {}, SCHEMA, "web_query_plan") == PLAN
    assert len(calls) == 1 and llm.names == ["web_query_plan"]
    assert runner.budget.counts["model_calls"] == 2 and runner.providers_used == ["local_fallback"]


def test_explicit_error_policy_and_sanitized_network_error(tmp_path):
    def fail(request):
        raise httpx.ConnectError("upstream echoed controlled-planner-key", request=request)
    log = RunLog(tmp_path)
    runner = planning_runner(config(tmp_path, fallback="error"),
        ModelRunner(LocalLLM(), RunBudget(), log), transport=httpx.MockTransport(fail))
    with pytest.raises(ModelUnavailable, match="transport failed:ConnectError"):
        runner.call("JSON", {}, SCHEMA, "web_query_plan")
    assert runner.budget.counts["model_calls"] == 1
    assert "controlled-planner-key" not in json.dumps(load_artifacts(tmp_path, log.run_id))


def test_expired_deadline_and_cancellation_never_call_cloud_or_fallback(tmp_path):
    calls = []
    client = DeepSeekPlanner(config(tmp_path), transport=response_transport(calls))
    with pytest.raises(BudgetExceeded):
        client.structured([{"role": "system", "content": "JSON"}], SCHEMA,
                          "web_query_plan", deadline=time.monotonic() - 1)
    local = ModelRunner(LocalLLM(), RunBudget(), RunLog(tmp_path))
    def cancelled():
        raise TurnCancelled()
    runner = planning_runner(config(tmp_path), local, cancel_check=cancelled,
                              transport=response_transport(calls))
    with pytest.raises(TurnCancelled):
        runner.call("JSON", {}, SCHEMA, "web_query_plan")
    assert not calls and not local.llm.names


def test_key_file_rotation_and_outbound_key_redaction(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    key = tmp_path / "key"
    key.write_text("first-private-key")
    calls = []
    client = DeepSeekPlanner(config(tmp_path, api_key_file="key"), transport=response_transport(calls))
    messages = [{"role": "system", "content": "JSON"},
                {"role": "user", "content": "accidental sk-controlled1234567890"}]
    client.structured(messages, SCHEMA, "web_query_plan")
    key.write_text("rotated-private-key")
    client.structured(messages, SCHEMA, "web_query_plan")
    assert [r.headers["authorization"] for r in calls] == ["Bearer first-private-key", "Bearer rotated-private-key"]
    assert all("sk-controlled1234567890" not in r.content.decode() for r in calls)
    assert "sk-controlled1234567890" in messages[1]["content"]


class EvidencePlanner:
    def __init__(self, *, sufficient=True, duplicate=False, fail_review=False, fabricated=False):
        self.names = []
        self.sufficient, self.duplicate = sufficient, duplicate
        self.fail_review, self.fabricated = fail_review, fabricated
        self.providers_used = ["deepseek"]

    def call(self, prompt, payload, schema, name, **kwargs):
        self.names.append(name)
        self.providers_used.append("deepseek")
        if name == "web_query_plan":
            return deepcopy(PLAN)
        if self.fail_review:
            raise InvalidModelOutput("controlled review failure")
        value = {"sufficient": self.sufficient,
                 "next_query": "" if self.sufficient else (QUERY if self.duplicate else "different prevention evidence"),
                 "selected_ids": ["fabricated"] if self.fabricated else [c["id"] for c in payload["candidates"][:2]],
                 "assessments":[{"evidence_id":c['id'],"applicable":True,"source_suitable":True,
                     "supports_need":True,"quotes":["Adults using medical devices require skin assessment and individualized pressure injury prevention."],
                     "limitation":"Controlled fixture only."} for c in payload['candidates'][:2]],
                 "reason": "Controlled test; compare covered topics."}
        # A real ModelRunner enforces exactly this validation before returning.
        try:
            Draft202012Validator(schema).validate(value)
        except Exception:
            raise InvalidModelOutput("controlled schema rejection") from None
        return value


def test_insufficient_material_triggers_one_followup_and_cannot_invent_citations(tmp_path):
    calls, planner, log, budget = [], EvidencePlanner(sufficient=False), RunLog(tmp_path), RunBudget()
    web = PlannedDeepSeekSearch(budget, log, planner, mode="live", fetcher=transport(calls))
    evidence = web.search(QUERY)
    assert evidence and len(evidence) <= 2
    assert [n for n in planner.names if n == "search_sufficiency"] == ["search_sufficiency"] * 2
    assert sum(c[0] == SEARCH_URL for c in calls) == 2
    assert budget.counts["search_requests"] <= 4 and budget.counts["fetch_requests"] <= 6
    assert web.events[-1]["search_sufficient"] is False
    assert web.events[-1]["planner_provider"] == "deepseek"
    assert all(s["source_url"] == ROW["url"] and s["text"] in BODY for s in evidence.values())
    replay = PlannedDeepSeekSearch(RunBudget(), RunLog(tmp_path), None, mode="replay", replay_run_id=log.run_id,
        fetcher=lambda *a, **k: pytest.fail("replay performed a network call"))
    assert replay.search(QUERY) == evidence
    assert not replay.budget.counts["model_calls"] and not replay.budget.counts["search_requests"]


@pytest.mark.parametrize("options", [{"sufficient": True}, {"sufficient": False, "duplicate": True},
                                     {"fail_review": True}, {"fabricated": True}])
def test_search_stops_on_sufficiency_duplicate_or_failed_review_preserving_actual_bodies(tmp_path, options):
    calls, planner = [], EvidencePlanner(**options)
    web = PlannedDeepSeekSearch(RunBudget(), RunLog(tmp_path), planner, mode="live", fetcher=transport(calls))
    evidence = web.search(QUERY)
    assert len(evidence) == 1 and sum(c[0] == SEARCH_URL for c in calls) == 1
    assert "fabricated" not in evidence
    if options.get("fail_review") or options.get("fabricated"):
        assert web.events[-1]["error"] == "InvalidModelOutput"


def test_empty_material_cannot_be_claimed_sufficient_and_off_makes_no_calls(tmp_path):
    planner = EvidencePlanner(sufficient=True)
    web = PlannedDeepSeekSearch(RunBudget(), RunLog(tmp_path), planner, mode="live", fetcher=transport([], rows=[]))
    assert not web.search(QUERY) and web.events[-1]["search_sufficient"] is False
    off = PlannedDeepSeekSearch(RunBudget(), RunLog(tmp_path), None, mode="off",
        fetcher=lambda *a, **k: pytest.fail("off performed a network call"))
    assert not off.search(QUERY) and off.budget.counts["model_calls"] == 0


def test_factory_keeps_default_search_and_bounds_opt_in_rounds(tmp_path):
    local = ModelRunner(LocalLLM(), RunBudget(), RunLog(tmp_path))
    default = AppConfig(tmp_path, {"web_search": {"provider": "deepseek"}})
    assert type(create_web_search(default, local.budget, local.log, local)) is DeepSeekSearch
    trial = create_web_search(config(tmp_path, max_search_rounds=1), local.budget, local.log, local, max_search_rounds=2)
    assert isinstance(trial, PlannedDeepSeekSearch) and trial.max_search_rounds == 1


def test_research_override_is_request_local_and_shares_existing_gpu_objects(monkeypatch):
    engine, seen = pipeline(), []
    base_values = deepcopy(engine.config.values)
    def answer(selected, *a, **kwargs):
        seen.append(selected)
        assert selected.retriever is engine.retriever and selected.llm is engine.llm
        return {"status": "partial"}
    monkeypatch.setattr("app.reflection_workflow.reflective_answer", answer)
    with TestClient(create_app(rag_config(), engine)) as client:
        assert client.post("/v1/research/answer", json={"query": "NIV皮肤评估",
            "search_planning": {"provider": "deepseek"}}).status_code == 200
        assert seen[0] is not engine and seen[0].config.get("search_planner", "provider") == "deepseek"
        assert engine.config.values == base_values
        assert client.post("/v1/research/answer", json={"query": "NIV皮肤评估"}).status_code == 200
        assert seen[-1] is engine and engine.config.values == base_values
        for invalid in ({"provider": "other"}, {"provider": "deepseek", "api_key_file": "/tmp/other"},
                        {"provider": "deepseek", "base_url": "https://example.org"},
                        {"provider": "deepseek", "max_search_rounds": 3}):
            assert client.post("/v1/research/answer", json={"query": "NIV皮肤评估",
                "search_planning": invalid}).status_code == 422


def test_request_override_cannot_bypass_shared_generation_queue(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    entered, release = Event(), Event()
    def answer(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return {"status": "partial"}
    monkeypatch.setattr("app.reflection_workflow.reflective_answer", answer)
    with TestClient(create_app(rag_config(), pipeline())) as client, ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.post, "/v1/research/answer", json={"query": "NIV皮肤评估",
            "search_planning": {"provider": "deepseek"}})
        try:
            assert entered.wait(3)
            assert client.post("/v1/research/answer", json={"query": "NIV皮肤评估"}).status_code == 429
            assert client.post("/v1/retrieve", json={"query": "NIV皮肤评估"}).status_code == 200
        finally:
            release.set()
        assert future.result(timeout=3).status_code == 200
