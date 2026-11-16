"""Controlled API contracts; live acceptance is recorded separately."""
from copy import deepcopy
import hashlib
import json

import pytest

from app.config import AppConfig
from app.deepseek_search import DeepSeekSearch, SEARCH_URL, FETCH_URL, daily_search
from app.evidence_snapshot import RunBudget, RunLog, load_artifacts
from app.external_evidence import ExternalFailure
from app.provenance import runtime_identity
from app.web_content import grounded_query_plan
from app.web_search import BAIDU_URL, create_web_search, page_excerpt
from modes.backend import TurnLog
from modes.daily_tools import read_page
from test_general_web_search import HTML, Planner, QUERY


BODY = ("# Controlled pressure injury guide\n\n"
        "Adults using medical devices require skin assessment and individualized pressure injury prevention.\n\n"
        "For non-invasive ventilation, review mask fit and skin condition without interrupting prescribed therapy.\n\n"
        "This is an engineering test fixture, not clinical guidance.")
ROW = {"title": "Controlled guide", "url": "https://example.org/guide",
       "snippet": "DISCOVERY ONLY", "summary": "SUMMARY IS NOT BODY", "content": BODY,
       "date": "2026-10-17T00:00:00+00:00", "site_name": "Fixture", "authority": 3, "rerank_score": 2}


@pytest.fixture(autouse=True)
def no_real_credentials_or_dns(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "controlled-deepseek-key")
    monkeypatch.setenv("BAIDU_SEARCH_API_KEY", "controlled-baidu-key")
    monkeypatch.setattr("app.deepseek_search.validate_public_url", lambda *a, **k: None)


def transport(calls, rows=None, search_error=None, fetch_body=BODY):
    def fetch(url, budget, kind, **kwargs):
        budget.consume(kind)
        calls.append((url, kind, kwargs))
        if url == SEARCH_URL:
            if search_error:
                raise search_error
            result = {"results": deepcopy([ROW] if rows is None else rows)}
        elif url == FETCH_URL:
            result = {"title": "Fetched guide", "content": fetch_body}
        elif url == BAIDU_URL:
            result = {"references": [{"title": "Baidu fixture", "url": "https://example.org/baidu",
                                      "content": "Baidu discovery snippet"}]}
        else:
            return HTML, url
        return json.dumps(result).encode(), url
    return fetch


def test_search_body_enters_evidence_and_replay_without_fetch_or_key(tmp_path, monkeypatch):
    calls, log, budget = [], RunLog(tmp_path), RunBudget()
    web = DeepSeekSearch(budget, log, Planner(), mode="live", fetcher=transport(calls))
    evidence = web.search(QUERY)
    assert len(evidence) == 1 and budget.counts["search_requests"] == 1
    assert budget.counts["fetch_requests"] == 0
    item = next(iter(evidence.values()))
    assert item["retrieval_method"] == "deepseek_search_content"
    assert item["publication_date"] == ROW["date"] and "SUMMARY IS NOT BODY" not in item["text"]
    assert item["fulltext_sha256"] == hashlib.sha256(BODY.encode()).hexdigest()
    assert json.loads(calls[0][2]["body"]) == {"query": QUERY, "count": 5, "need_content": True}
    assert calls[0][2]["headers"]["Authorization"] == "Bearer controlled-deepseek-key"
    assert "controlled-deepseek-key" not in json.dumps(load_artifacts(tmp_path, log.run_id))
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    replay = DeepSeekSearch(RunBudget(), RunLog(tmp_path), None, mode="replay", replay_run_id=log.run_id,
                           fetcher=lambda *a, **k: pytest.fail("Replay made a network request"))
    assert replay.search(QUERY) == evidence and replay.budget.counts["search_requests"] == 0


def test_missing_content_calls_fetch_and_never_promotes_summary(tmp_path):
    calls, budget = [], RunBudget()
    web = DeepSeekSearch(budget, RunLog(tmp_path), Planner(), mode="live",
                        fetcher=transport(calls, rows=[{**ROW, "content": ""}]))
    item = next(iter(web.search(QUERY).values()))
    assert item["retrieval_method"] == "deepseek_fetch"
    assert [c[0] for c in calls] == [SEARCH_URL, FETCH_URL]
    assert budget.counts["fetch_requests"] == 1
    assert "SUMMARY IS NOT BODY" not in item["text"]


@pytest.mark.parametrize("status", [401, 402, 429, 500])
def test_primary_http_failure_uses_baidu_without_free_engine_wait(tmp_path, status):
    calls, budget = [], RunBudget()
    web = DeepSeekSearch(budget, RunLog(tmp_path), Planner(), mode="live",
        fallback_options={"api_key_env": "BAIDU_SEARCH_API_KEY"},
        fetcher=transport(calls, search_error=ExternalFailure("external_http_" + str(status))))
    evidence = web.search(QUERY)
    assert evidence and [c[0] for c in calls] == [SEARCH_URL, BAIDU_URL, "https://example.org/baidu"]
    assert budget.counts["search_requests"] == 2
    assert web.last_search_attempts[0]["error"] == "external_http_" + str(status)
    assert web.last_search_attempts[1]["backend"] == "baidu"


def test_unusable_primary_body_also_uses_baidu(tmp_path):
    calls = []
    web = DeepSeekSearch(RunBudget(), RunLog(tmp_path), Planner(), mode="live",
        fallback_options={"api_key_env": "BAIDU_SEARCH_API_KEY"},
        fetcher=transport(calls, rows=[{**ROW, "content": "环境异常\n完成验证后即可继续访问。"}]))
    assert web.search(QUERY)
    assert BAIDU_URL in [c[0] for c in calls]
    assert any(e.get("status") == "page_access_challenge" for e in web.events[0]["events"])


def test_empty_search_is_not_successful_evidence(tmp_path):
    web = DeepSeekSearch(RunBudget(), RunLog(tmp_path), Planner(), mode="live", fetcher=transport([], rows=[]))
    assert web.search(QUERY) == {} and web.events[0]["status"] == "no_evidence"


def test_cnki_mirrors_do_not_fill_both_body_slots_with_same_paper(tmp_path):
    paper = "552d3c797d83fc7390ef2bf8ec251537"
    rows = [{**ROW, "url": "https://yxln.cbpt.cnki.net/portal/client/paper/" + paper},
            {**ROW, "url": "https://publish.cnki.net/journal/portal/client/paper/" + paper},
            {**ROW, "url": "https://example.org/different-paper"}]
    web = DeepSeekSearch(RunBudget(), RunLog(tmp_path), None, fetcher=transport([], rows=rows))
    found = web._discover(QUERY)
    assert len(found) == 2 and found[1]["url"] == rows[2]["url"]


@pytest.mark.parametrize("url", ["http://example.org/x", "https://127.0.0.1/x",
    "https://localhost/x", "https://user:password@example.org/x", "https://example.org:8443/x"])
def test_unsafe_result_urls_are_not_admitted(tmp_path, url):
    web = DeepSeekSearch(RunBudget(), RunLog(tmp_path), Planner(), mode="live",
                        fetcher=transport([], rows=[{**ROW, "url": url}]))
    assert not web.search(QUERY)


@pytest.mark.parametrize("field", ["content", "summary", "title"])
def test_malformed_api_types_are_recorded_as_failure(tmp_path, field):
    web = DeepSeekSearch(RunBudget(), RunLog(tmp_path), Planner(), mode="live",
                        fetcher=transport([], rows=[{**ROW, field: ["invalid"]}]))
    assert not web.search(QUERY)
    assert web.last_search_attempts[0]["error"] == "deepseek_invalid_result"


def test_arbitrary_transport_message_cannot_leak_key(tmp_path):
    log = RunLog(tmp_path)
    web = DeepSeekSearch(RunBudget(), log, Planner(), mode="live",
        fetcher=transport([], search_error=RuntimeError("reflected controlled-deepseek-key")))
    assert not web.search(QUERY)
    assert "controlled-deepseek-key" not in json.dumps(load_artifacts(tmp_path, log.run_id))


def test_file_key_is_reloaded_without_logging_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    key = tmp_path / "key"
    key.write_text("first-private-key")
    calls, log = [], RunLog(tmp_path / "logs")
    web = DeepSeekSearch(RunBudget(), log, None, mode="off", api_key_file=key, fetcher=transport(calls))
    web._discover(QUERY)
    key.write_text("rotated-private-key")
    web._discover(QUERY)
    assert [c[2]["headers"]["Authorization"] for c in calls] == ["Bearer first-private-key", "Bearer rotated-private-key"]
    artifacts = json.dumps(load_artifacts(tmp_path / "logs", log.run_id))
    assert "first-private-key" not in artifacts and "rotated-private-key" not in artifacts


def test_factory_and_daily_chat_share_deepseek_and_real_body(tmp_path):
    cfg = AppConfig(tmp_path, {"web_search": {"provider": "deepseek", "deepseek_fallback_provider": "",
                                            "strategy": "fallback"}})
    assert isinstance(create_web_search(cfg, RunBudget(), RunLog(tmp_path), None), DeepSeekSearch)
    calls = []
    sources, events = daily_search(QUERY, TurnLog(tmp_path / "turn"), cfg, fetcher=transport(calls))
    assert sources[0]["material_level"] == "web_page_excerpt"
    assert sources[0]["search_backend"] == "deepseek"
    assert events[-1]["full_pages"] == 1
    assert [c[0] for c in calls] == [SEARCH_URL]
    assert {"app/deepseek_search.py", "app/web_content.py"} <= runtime_identity()["files"].keys()


def test_daily_chat_failed_fetch_does_not_promote_medical_snippet(tmp_path):
    cfg = AppConfig(tmp_path, {"web_search": {"provider": "deepseek", "deepseek_fallback_provider": ""}})
    sources, events = daily_search(QUERY, TurnLog(tmp_path), cfg,
        fetcher=transport([], rows=[{**ROW, "content": ""}], fetch_body=""))
    assert sources == []
    assert events[-1]["full_pages"] == 0


def test_general_chat_can_keep_explicitly_labelled_discovery_snippet(tmp_path):
    cfg = AppConfig(tmp_path, {"web_search": {"provider": "deepseek", "deepseek_fallback_provider": ""}})
    sources, events = daily_search('Find cooking lessons', TurnLog(tmp_path), cfg,
        fetcher=transport([], rows=[{**ROW, "content": ""}], fetch_body=""))
    assert len(sources)==1 and sources[0]['material_level']=='search_snippet'
    assert events[-1]['full_pages']==0


def test_long_query_and_requested_freshness_fit_api_limit(tmp_path):
    calls = []
    web = DeepSeekSearch(RunBudget(), RunLog(tmp_path), None, freshness="w",
                        fetcher=transport(calls), mode="off")
    web._discover("无创通气面罩预防 " * 50)
    query = json.loads(calls[0][2]["body"])["query"]
    assert len(query) <= 128 and " after:" in query


@pytest.mark.parametrize("date", ["2022年6月前", "before:2022-06-01", "before 2022"])
def test_local_source_cutoff_is_removed_from_query_and_focus(date):
    plan = {"search_query": f"无创通气面罩 预防指南 {date}", "verification_focus": f"检索{date}资料"}
    result, changes = grounded_query_plan(plan, "请找无创通气面罩相关压力性损伤预防的指南")
    assert "2022" not in result["search_query"] and "2022" not in result["verification_focus"] and changes
    kept, _ = grounded_query_plan(plan, "请找2022年6月前发表的指南")
    assert "2022" in kept["search_query"]


@pytest.mark.parametrize("date", ["2021-2026年", "2021年至2026年", "2021–2026", "2021 to 2026"])
def test_unrequested_year_ranges_are_removed_without_partial_years(date):
    plan = {"search_query": f"无创通气面罩 预防指南 {date}", "verification_focus": f"核对{date}资料"}
    result, changes = grounded_query_plan(plan, "查找无创通气面罩相关压力性损伤指南")
    assert result["search_query"] == "无创通气面罩 预防指南" and changes
    kept, _ = grounded_query_plan(plan, "查找" + date + "发表的指南")
    assert kept == plan


@pytest.mark.parametrize("url,text", [
    ("https://mp.weixin.qq.com/mp/wappoc_appmsgcaptcha", "完成验证后即可继续访问。"),
    ("https://example.org/article", "环境异常\n当前环境异常，完成验证后即可继续访问。\n去验证"),
])
def test_wechat_challenge_is_rejected_by_both_readers(url, text):
    item, event = read_page("https://example.org/article", fetcher=lambda *a, **k: (text.encode(), url))
    assert item is None and event["error"] == "page_access_challenge"
    item, status = page_excerpt(text.encode(), {}, url, "验证", body_text=text)
    assert item is None and status == "page_access_challenge"
