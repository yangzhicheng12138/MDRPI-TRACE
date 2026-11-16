"""Controlled engineering fixtures: never presented as real search/model output."""
from copy import deepcopy
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from app.evidence_snapshot import RunBudget, RunLog, load_artifacts
from app.external_evidence import ExternalFailure, safe_get, validate_public_url
from app.web_search import BraveSearch, DDGSSearch, page_excerpt


HTML = b"""<html><head><title>Pressure injury guide</title></head><body><nav>IGNORE NAVIGATION</nav>
<main><h1>Medical device related pressure injury prevention</h1>
<p>Adults using medical devices require careful skin assessment. Pressure injury prevention
includes checking the skin under the device and assessing fit without interrupting prescribed therapy.</p>
<p>For non-invasive ventilation, inspect the mask interface and the skin during routine care.
Device choice and pressure redistribution must be individualized to the patient and the clinical situation.</p>
<h2>Limitations</h2><p>This controlled development fixture is not a clinical guideline and supplies no patient outcomes.</p>
</main><script>REMOVE SCRIPT</script></body></html>"""
QUERY = "adult medical device pressure injury skin assessment prevention"


class Planner:
    def __init__(self):
        self.calls = []

    def call(self, prompt, payload, schema, name, **kwargs):
        assert name == "web_query_plan"
        self.calls.append(deepcopy(payload))
        return {"search_query": QUERY, "verification_focus": "Inspect device interface skin."}


def transport(calls, *, fail_page=False, fail_search=False, page_html=HTML):
    def fetch(url, budget, kind, **kwargs):
        budget.consume(kind)
        calls.append((kind, url, kwargs))
        if kind == "search_requests":
            if fail_search:
                raise ExternalFailure("external_http_429")
            return json.dumps({"web": {"results": [{"title": "Controlled guide",
                "url": "https://example.org/guide", "description": "DISCOVERY ONLY SNIPPET"}]}}).encode(), url
        if fail_page:
            raise ExternalFailure("external_http_403")
        return page_html, url
    return fetch


def ddgs_client(calls, error=None):
    class Client:
        def __init__(self, timeout):
            assert 0 < timeout <= 8
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def text(self, **params):
            calls.append(("search_requests", params["query"], params))
            if error:
                raise error
            return [{"title": "Controlled guide", "href": "https://example.org/guide",
                     "body": "DISCOVERY ONLY SNIPPET"}]
    return Client


def test_ddgs_needs_no_key_and_replays_without_search_or_planning(tmp_path, monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    calls, planner, log = [], Planner(), RunLog(tmp_path)
    engine = DDGSSearch(RunBudget(), log, planner, mode="live", search_client=ddgs_client(calls), fetcher=transport(calls))
    result = engine.search(QUERY)
    assert result and engine.events[0]["provider"] == "DDGS"
    assert calls[0][2] == {"query": QUERY, "region": "us-en", "max_results": 5, "backend": "brave"}
    assert len(planner.calls) == 1
    replay = DDGSSearch(RunBudget(), RunLog(tmp_path), None, mode="replay", replay_run_id=log.run_id,
                       search_client=lambda **k: pytest.fail("Unexpected DDGS call"),
                       fetcher=lambda *a, **k: pytest.fail("Unexpected page fetch"))
    assert replay.search(QUERY) == result
    assert replay.budget.counts["model_calls"] == replay.budget.counts["search_requests"] == replay.budget.counts["fetch_requests"] == 0


def test_ddgs_off_does_not_plan_or_search(tmp_path):
    engine = DDGSSearch(RunBudget(), RunLog(tmp_path), None, mode="off",
                       search_client=lambda **k: pytest.fail("Unexpected DDGS call"))
    assert engine.search(QUERY) == {}
    assert engine.budget.counts["search_requests"] == 0


def test_ddgs_unavailable_backend_does_not_silently_search_auto(tmp_path):
    engine = DDGSSearch(RunBudget(), RunLog(tmp_path), Planner(), mode="live", backend="unavailable_backend",
                       search_client=lambda **k: pytest.fail("Unexpected automatic fallback"))
    assert engine.search(QUERY) == {}
    assert engine.events[0]["status"] == "technical_failure"
    assert engine.events[0]["error"] == "ddgs_backend_unavailable:unavailable_backend"
    assert engine.budget.counts["search_requests"] == 0


@pytest.mark.parametrize("message", ["No results found.", "upstream rate limit"])
def test_ddgs_failure_is_recorded_not_called_a_successful_empty_search(tmp_path, message):
    from ddgs.exceptions import DDGSException
    engine = DDGSSearch(RunBudget(), RunLog(tmp_path), Planner(), mode="live",
                       search_client=ddgs_client([], DDGSException(message)),
                       fetcher=lambda *a, **k: pytest.fail("Should not fetch after failed search"))
    assert engine.search(QUERY) == {}
    assert engine.events[0]["status"] == "technical_failure"
    assert message in engine.events[0]["error"]
    assert engine.budget.counts["search_requests"] == 1


def test_brave_dynamic_query_body_provenance_and_no_key_in_logs(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-secret-do-not-record")
    calls, planner, log = [], Planner(), RunLog(tmp_path)
    engine = BraveSearch(RunBudget(), log, planner, mode="live", fetcher=transport(calls))
    context = {"question": "Controlled question", "step": "answer_verification", "draft_claims": {}}
    result = engine.search("Controlled question", context=context)
    params = parse_qs(urlsplit(calls[0][1]).query)
    assert params["q"] == [QUERY] and params["count"] == ["5"]
    assert calls[0][2]["headers"]["X-Subscription-Token"] == "test-secret-do-not-record"
    assert "headers" not in calls[1][2]
    assert planner.calls == [context]
    source = next(iter(result.values()))
    assert source["material_level"] == "web_page_excerpt"
    assert source["source_url"] == "https://example.org/guide"
    assert "skin assessment" in source["text"]
    assert "DISCOVERY ONLY SNIPPET" not in source["text"]
    assert "REMOVE SCRIPT" not in source["text"]
    assert source["admission"]["decision"] == "candidate"
    assert "test-secret-do-not-record" not in "".join(p.read_text() for p in log.path.glob("*.json"))
    # Offline replay needs neither credentials, query planning nor transport.
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY")
    replay_log = RunLog(tmp_path)
    replay = BraveSearch(RunBudget(), replay_log, None, mode="replay", replay_run_id=log.run_id,
                         fetcher=lambda *a, **k: pytest.fail("Replay attempted HTTP"))
    assert replay.search("Controlled question", context=context) == result
    assert replay.events[0]["replayed"]


@pytest.mark.parametrize("mode", ["off", "live"])
def test_no_key_does_not_call_planner_or_http(tmp_path, monkeypatch, mode):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    planner = Planner()
    adapter = BraveSearch(RunBudget(), RunLog(tmp_path), planner, mode=mode,
                          fetcher=lambda *a, **k: pytest.fail("Unexpected HTTP"))
    assert adapter.search(QUERY) == {} and planner.calls == []
    assert adapter.events[0]["status"] == ("disabled" if mode == "off" else "technical_failure")
    if mode == "live":
        assert adapter.events[0]["category"] == "configuration"


@pytest.mark.parametrize("failure", ["page", "search"])
def test_network_failures_are_not_promoted_from_snippets(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "controlled-key")
    adapter = BraveSearch(RunBudget(), RunLog(tmp_path), Planner(), mode="live",
                          fetcher=transport([], fail_page=failure == "page", fail_search=failure == "search"))
    assert adapter.search(QUERY) == {}
    assert adapter.events[0]["status"] == "technical_failure"


def test_private_page_redirects_rejected_and_credentials_not_forwarded():
    def resolver(host, *args, **kwargs):
        return [(None, None, None, None, ("127.0.0.1" if host == "localhost" else "93.184.216.34", 443))]
    with pytest.raises(ExternalFailure, match="not_public"):
        validate_public_url("https://localhost/secret", resolver)
    requests = []
    class Response:
        def __init__(self, redirect):
            self.status = 302 if redirect else 200
        def getheader(self, name, default=None):
            return "https://example.org/landing" if name == "Location" else default
        def read1(self, size):
            return b""
    class Connection:
        sock = None
        def __init__(self, host, address, timeout):
            self.host = host
        def request(self, method, path, headers):
            requests.append((self.host, headers))
        def getresponse(self):
            return Response(len(requests) == 1)
        def close(self):
            pass
    safe_get("https://api.search.brave.com/search", RunBudget(), "search_requests",
             resolver=resolver, connector=Connection, validator=validate_public_url,
             headers={"X-Subscription-Token": "controlled-key"})
    assert "X-Subscription-Token" in requests[0][1]
    assert "X-Subscription-Token" not in requests[1][1]


def test_unavailable_or_unrelated_body_is_not_evidence():
    assert page_excerpt(b"", {}, "https://example.org", QUERY)[0] is None
    assert page_excerpt(HTML, {}, "https://example.org", "astronomy telescope galaxies")[0] is None


def test_zero_results_are_recorded_without_fetching_pages(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "controlled-key")
    def empty(url, budget, kind, **kwargs):
        assert kind == "search_requests"
        budget.consume(kind)
        return b'{"query":{"original":"controlled"}}', url
    budget = RunBudget()
    adapter = BraveSearch(budget, RunLog(tmp_path), Planner(), mode="live", fetcher=empty)
    assert adapter.search(QUERY) == {}
    assert adapter.events[0]["status"] == "no_evidence"
    assert budget.counts["search_requests"] == 1 and budget.counts["fetch_requests"] == 0


def test_replay_does_not_consume_another_providers_batch(tmp_path):
    log = RunLog(tmp_path)
    log.write("external_batch", {"query": QUERY, "provider": "EuropePMC", "status": "no_evidence",
                                 "evidence": [], "events": []})
    replay = BraveSearch(RunBudget(), RunLog(tmp_path), None, mode="replay", replay_run_id=log.run_id)
    assert replay.search(QUERY) == {}
    assert replay.events[0]["status"] == "replay_miss"


def test_verification_body_changes_invalidate_audit_reuse():
    from app.reflection_checks import run_checks
    from test_research_latency import ReviewRunner, complete_steps, sources
    from test_reflection_v2 import facts
    runner, cache, evidence = ReviewRunner(), {}, sources()
    evidence["WEB1"] = {"evidence_id": "WEB1", "text": "Controlled first body", "origin": "external"}
    kwargs = dict(research_mode=True, compact=True, scoped=True, consolidated=True, cache=cache,
                  verification_ids={"WEB1"})
    run_checks(runner, "成人使用NIV", facts(), complete_steps(), evidence, **kwargs)
    count = len(runner.calls)
    run_checks(runner, "成人使用NIV", facts(), complete_steps(), evidence, **kwargs)
    assert len(runner.calls) == count
    evidence["WEB1"]["text"] = "Controlled changed body"
    run_checks(runner, "成人使用NIV", facts(), complete_steps(), evidence, **kwargs)
    assert len(runner.calls) > count
    assert runner.calls[-1]["verification_sources"][0]["text"] == "Controlled changed body"


def test_verification_context_keeps_both_long_pages_and_existing_citations():
    from app.six_step_answer import evidence_context
    local = [{"evidence_id": f"L{i}", "text": "Existing complete citation. " * 10} for i in range(6)]
    web = [{"evidence_id": f"WEB{i}", "text": "Complete fetched paragraph. " * 220} for i in range(2)]
    required = {s["evidence_id"] for s in local}
    sources = evidence_context(local + web, maximum_chars=6000, required_ids=required,
                               verification_ids={s["evidence_id"] for s in web})
    assert [(s["evidence_id"], s["text"]) for s in sources] == [
        (s["evidence_id"], s["text"]) for s in web + local]
    assert len(evidence_context(local + web, maximum_chars=6000, required_ids=required)) == 6


@pytest.mark.parametrize("finding", ["none", "correction", "knowledge", "revision_failure"])
@pytest.mark.parametrize("provider", ["brave", "ddgs", "deepseek"])
@pytest.mark.parametrize("long_cited", [False, True])
def test_pre_draft_search_reaches_generation_reflection_revision_and_replay(tmp_path, monkeypatch, finding, provider, long_cited):
    import app.reflection_workflow as workflow
    from test_reflection_web_flow import setup_flow, WebFlowLLM
    from test_lean_research_output import lean_draft, complete_wire
    from test_research_latency import focused_wire

    class LLM(WebFlowLLM):
        def structured(self, messages, schema, name, **kwargs):
            payload = json.loads(messages[1]["content"])
            if name == "web_query_plan":
                assert not getattr(self, "draft_started", False)
                assert "draft_claims" not in payload
                assert payload["local_sources"]
                return {"search_query": QUERY, "verification_focus": "Check skin assessment"}
            if name == "research_task_scope":
                return {"literature_requested": False, "requirement_quote": "", "assigned_steps": []}
            if name == "six_steps_draft":
                self.draft_started = True
                web_source = next(s for s in payload["sources"] if s["evidence_id"].startswith("EXT_WEB_"))
                local_source = next(s for s in payload["sources"] if not s["evidence_id"].startswith("EXT_WEB_"))
                assert "skin assessment" in web_source["text"]
                if long_cited:
                    assert len(web_source["text"]) > 5900
                result = lean_draft()
                for step in result.values():
                    for claim in step["claims"]:
                        claim.update(kind="model_inference", fact_ids=[], evidence_ids=[])
                if long_cited:
                    result["recognize"]["claims"][0].update(kind="source_statement",
                        evidence_ids=[local_source["evidence_id"]])
                result["solutions"]["claims"][0].update(kind="source_statement",
                    text="Controlled initial answer informed by fetched skin assessment guidance.",
                    evidence_ids=[web_source["evidence_id"]])
                return result
            if name == "six_steps_revision":
                assert any(s["evidence_id"].startswith("EXT_WEB_") for s in payload["sources"])
                if long_cited:
                    assert payload["previous_steps"]["recognize"]["claims"][0]["evidence_ids"][0] in {
                        s["evidence_id"] for s in payload["sources"]}
                    assert len(next(s["text"] for s in payload["sources"]
                                    if s["evidence_id"].startswith("EXT_WEB_"))) > 5900
                if finding == "revision_failure":
                    from app.local_llm import InvalidModelOutput
                    raise InvalidModelOutput("Controlled invalid claim link during revision")
                result = {s: complete_wire(payload["previous_steps"][s]) for s in payload["steps"]}
                c = result["solutions"]["claims"][0]
                c.update(text="Controlled revision based on fetched skin assessment guidance.", kind="source_statement",
                         evidence_ids=[s["evidence_id"] for s in payload["sources"] if s["evidence_id"].startswith("EXT_WEB_")])
                self.revised = True
                return result
            if name == "audit_consolidated":
                assert payload["verification_sources"][0]["material_level"] == "web_page_excerpt"
                assert "skin assessment" in payload["verification_sources"][0]["text"]
                web_id = payload["verification_sources"][0]["evidence_id"]
                assert web_id in payload["targets"]["evidence:solutions.c1"]["source_ids"]
                assert "它们不是初稿已有的引用" not in messages[0]["content"]
                assert "可能包含这些网页" in messages[0]["content"]
                rows = focused_wire(payload["targets"])
                if finding != "none" and not self.revised:
                    rows["evidence:solutions.c1"] = {"assessment": "Controlled disagreement with fetched page",
                        "finding": "unsupported_claim", "severity": "major",
                        "route": "correction" if finding == "revision_failure" else finding}
                return rows
            return super().structured(messages, schema, name, **kwargs)

    if provider == "brave":
        monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "controlled-key")
    else:
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    engine, _ = setup_flow(tmp_path, monkeypatch, "sufficient")
    if provider == "brave":
        engine.config.values["web_search"] = {"provider": "brave"}
    elif provider == "deepseek":
        monkeypatch.setenv("DEEPSEEK_API_KEY", "controlled-deepseek-key")
        engine.config.values["web_search"] = {"provider": "deepseek", "deepseek_fallback_provider": ""}
    engine.config.values["reflection"].update(research_compact=True, research_lean_output=True,
        research_consolidated_review=True, research_max_revision_rounds=1)
    calls = []
    # Real live run omitted a 5,992-character page whenever the draft already
    # cited local material: both competed for the same 6,000-character budget.
    page_html = (b"<html><body><main><p>" +
                 (b"Controlled complete skin assessment guidance. " * 130) +
                 b"</p></main></body></html>") if long_cited else HTML
    monkeypatch.setattr(workflow, "BraveSearch", lambda *a, **k: BraveSearch(*a, **k, fetcher=transport(calls, page_html=page_html)))
    monkeypatch.setattr(workflow, "DDGSSearch", lambda *a, **k: DDGSSearch(*a, **k,
        search_client=ddgs_client(calls), fetcher=transport(calls, page_html=page_html)))
    if provider == "deepseek":
        from app.deepseek_search import DeepSeekSearch, SEARCH_URL
        from trafilatura import extract
        def fetch_deepseek(url, budget, kind, **kwargs):
            budget.consume(kind)
            calls.append((kind, url, kwargs))
            value = ({"results": [{"title": "Controlled guide", "url": "https://example.org/guide",
                                  "summary": "DISCOVERY ONLY", "content": ""}]}
                     if url == SEARCH_URL else {"title": "Controlled guide", "content": extract(page_html)})
            return json.dumps(value).encode(), url
        monkeypatch.setattr("app.deepseek_search.validate_public_url", lambda *a, **k: None)
        monkeypatch.setattr("app.deepseek_search.DeepSeekSearch",
                            lambda *a, **k: DeepSeekSearch(*a, **k, fetcher=fetch_deepseek))
    outputs = []
    query = "NIV 皮肤评估"
    for mode in ["live", "replay"]:
        engine.llm = LLM("sufficient")
        calls.clear()
        result = workflow.reflective_answer(engine, query, research_mode=True, web_mode=mode, web_strategy="verify",
            replay_run_id=outputs[0]["run_id"] if outputs else None)
        if finding == "revision_failure":
            assert result["errors"][0]["category"] == "format"
            assert result["last_verified_fallback"]
        else:
            assert result["errors"] == []
        assert result["web"]["provider"] == {"ddgs": "DDGS", "brave": "BraveSearch", "deepseek": "DeepSeekSearch"}[provider]
        artifacts = load_artifacts(tmp_path, result["run_id"])
        filenames = [name.split("_", 1)[1] for name, _ in artifacts]
        assert filenames.index("external_batch.json") < filenames.index("draft.json") < filenames.index("audits_0.json")
        assert len([n for n in filenames if n == "external_batch.json"]) == 1
        assert result["timing"]["revision_rounds"] == int(finding == "correction")
        assert result["web"]["answer_text_changed"] == (finding == "correction")
        if finding == "knowledge":
            # The draft already saw all retrieved evidence. An unchanged local
            # supplement must not force a rewrite or a duplicate web search.
            assert result["stop_reason"] == "no_new_usable_evidence"
        assert result["web"]["used_in_final"]
        if mode == "replay":
            assert calls == []
            assert result["timing"]["search_requests"] == result["timing"]["fetch_requests"] == 0
            assert result["six_steps"] == outputs[0]["six_steps"]
        else:
            assert [c[0] for c in calls] == ["search_requests", "fetch_requests"]
        outputs.append(result)


def test_api_verify_missing_key_preserves_answer_and_off_never_searches(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.api import create_app
    from test_reflection_web_flow import setup_flow
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    engine, _ = setup_flow(tmp_path, monkeypatch, "sufficient")
    engine.config.values["web_search"] = {"provider": "brave"}
    with TestClient(create_app(engine.config, engine)) as client:
        for mode in ["live", "off"]:
            result = client.post("/v1/research/answer", json={"query": "NIV 皮肤评估", "web_mode": mode,
                                "web_strategy": "verify"}).json()
            assert result["web"]["provider"] == "BraveSearch"
            assert result["six_steps"]["solutions"]["claims"]
            assert result["timing"]["search_requests"] == 0
            if mode == "live":
                assert result["errors"][0]["category"] == "configuration"
                assert result["status"] == "partial"
            else:
                assert result["errors"] == [] and result["web"]["events"] == []


@pytest.mark.parametrize("failure", [None, "search", "page"])
def test_daily_research_searches_before_draft_and_keeps_answer_on_failure(tmp_path, monkeypatch, failure):
    from fastapi.testclient import TestClient
    from app.api import create_app
    import app.reflection_workflow as workflow
    from test_reflection_web_flow import setup_flow, WebFlowLLM
    from ddgs.exceptions import DDGSException

    calls = []
    class LLM(WebFlowLLM):
        drafting = False
        def structured(self, messages, schema, name, **kwargs):
            payload = json.loads(messages[1]["content"])
            if name == "web_query_plan":
                assert not self.drafting and "draft_claims" not in payload
                assert payload["local_sources"]
                return {"search_query": QUERY, "verification_focus": "Check skin assessment"}
            if name.startswith("six_step_"):
                self.drafting = True
                assert calls and calls[0][0] == "search_requests"
                if not failure:
                    assert any(s["evidence_id"].startswith("EXT_WEB_") for s in payload["sources"])
            return super().structured(messages, schema, name, **kwargs)

    engine, _ = setup_flow(tmp_path, monkeypatch, "sufficient")
    engine.llm = LLM("sufficient")
    monkeypatch.setattr(workflow, "DDGSSearch", lambda *a, **k: DDGSSearch(*a, **k,
        search_client=ddgs_client(calls, DDGSException("controlled network failure") if failure == "search" else None),
        fetcher=transport(calls, fail_page=failure == "page")))
    with TestClient(create_app(engine.config, engine)) as client:
        # Deliberately omit both mode and strategy to exercise daily defaults.
        response = client.post("/v1/research/answer", json={"query": "NIV 皮肤评估"})
    result = response.json()
    assert response.status_code == 200
    assert result["web"]["mode"] == "live" and result["web"]["strategy"] == "verify"
    assert result["web"]["provider"] == "DDGS"
    assert result["six_steps"]["solutions"]["claims"] and result["answer"]
    artifacts = load_artifacts(tmp_path, result["run_id"])
    names = [n.split("_", 1)[1] for n, _ in artifacts]
    assert names.index("external_batch.json") < names.index("draft.json")
    if failure:
        assert result["errors"] and not result["web"]["used_in_final"]
    else:
        assert result["errors"] == [] and result["web"]["used_in_final"]
