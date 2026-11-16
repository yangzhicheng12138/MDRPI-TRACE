"""Controlled query/routing tests; never a substitute for real model validation."""

from copy import deepcopy
import json
from urllib.parse import parse_qs, urlsplit

import pytest

from app.evidence_snapshot import RunBudget, RunLog, load_artifacts
from app.external_evidence import EuropePMC, ExternalFailure
from app.local_llm import InvalidModelOutput, ModelUnavailable


def context(question="寻找2026年成人NIV面罩压力性损伤危险因素的原始研究。", needed="缺少预测因素研究"):
    return {"question": question, "step": "solutions", "devices": ["niv_mask"],
            "gaps": [{"route": "knowledge", "needed": needed}]}


def plan(topic="risk factors"):
    return {"evidence_question": "Original adult NIV pressure injury research in 2026 on " + topic,
            "concept_groups": [["pressure injury", "pressure ulcer"],
                               ["non-invasive ventilation", "NIV"], [topic]],
            "date_from": "2026-01-01", "date_to": "2026-12-31",
            "publication_types": ["Journal Article"], "exclude_publication_types": ["Review"]}


class Planner:
    def __init__(self, value=None, error=None):
        self.value = value or plan()
        self.error = error
        self.calls = []

    def call(self, prompt, payload, schema, name, **kwargs):
        self.calls.append((name, payload))
        if self.error:
            raise self.error
        if name == "external_query_plan":
            return deepcopy(self.value)
        return {"decision": "admit", "applicability": "Controlled adult source", "limitations": "Test only"}


def transport(calls, xml=None):
    def fetch(url, budget, kind):
        budget.consume(kind)
        calls.append((url, kind))
        if kind == "search_requests":
            items = [{"pmcid": "PMC123", "title": "Controlled adult research", "license": "cc by"}] if xml else []
            return json.dumps({"hitCount": len(items), "resultList": {"result": items}}).encode(), url
        return xml, url
    return fetch


def test_dynamic_terms_and_explicit_constraints_reach_actual_request(tmp_path):
    calls, runner = [], Planner()
    adapter = EuropePMC(RunBudget(), RunLog(tmp_path), runner, "live", fetcher=transport(calls))
    adapter.search("adult fixed routing key", context=context())
    actual = parse_qs(urlsplit(calls[0][0]).query)["query"][0]
    assert '(TITLE_ABS:"risk" AND TITLE_ABS:"factors")' in actual
    assert '(TITLE_ABS:"non-invasive" AND TITLE_ABS:"ventilation")' in actual
    assert 'FIRST_PDATE:[2026-01-01 TO 2026-12-31]' in actual
    assert 'PUB_TYPE:"Journal Article"' in actual and 'NOT (PUB_TYPE:"Review")' in actual
    assert "OPEN_ACCESS:Y" in actual
    assert runner.calls[0][1]["question"] == context()["question"]
    assert runner.calls[0][1]["gaps"] == context()["gaps"]


def test_changed_gap_changes_request_for_same_device_and_step(tmp_path):
    queries = []
    for topic in ["risk factors", "hydrocolloid dressing"]:
        calls = []
        adapter = EuropePMC(RunBudget(), RunLog(tmp_path), Planner(plan(topic)), "live", fetcher=transport(calls))
        adapter.search("same device and step", context=context(needed=topic))
        queries.append(parse_qs(urlsplit(calls[0][0]).query)["query"][0])
    assert queries[0] != queries[1]
    assert 'TITLE_ABS:"hydrocolloid" AND TITLE_ABS:"dressing"' in queries[1]


def test_new_device_terms_are_not_limited_to_old_vocabulary(tmp_path):
    value = plan("extracorporeal membrane oxygenation cannula")
    value["concept_groups"] = [["pressure injury"], ["ECMO", "extracorporeal membrane oxygenation"]]
    value.update(date_from="", date_to="", publication_types=[], exclude_publication_types=[])
    calls = []
    adapter = EuropePMC(RunBudget(), RunLog(tmp_path), Planner(value), "live", fetcher=transport(calls))
    adapter.search("device pressure injury", context=context("ECMO导管固定部位压力损伤研究。"))
    actual = parse_qs(urlsplit(calls[0][0]).query)["query"][0]
    assert 'TITLE_ABS:"ECMO"' in actual and "FIRST_PDATE" not in actual


def test_dynamic_question_reaches_selection_and_admission(tmp_path):
    xml = (b'<article><license>CC BY</license><body><sec><title>Results</title>'
           b'<p>Adult participants received hydrocolloid dressings for protection during treatment. '
           b'The study compared preventive dressing strategies and reported their outcomes.</p>'
           b'</sec></body></article>')
    calls, runner = [], Planner(plan("hydrocolloid dressings"))
    adapter = EuropePMC(RunBudget(), RunLog(tmp_path), runner, "live", fetcher=transport(calls, xml))
    found = adapter.search("unrelated legacy routing words", context=context())
    assert found
    admission = next(payload for name, payload in runner.calls if name == "external_admission")
    assert "hydrocolloid dressings" in admission["professional_query"]


def test_off_does_not_plan_or_search(tmp_path):
    calls, runner = [], Planner()
    adapter = EuropePMC(RunBudget(), RunLog(tmp_path), runner, "off", fetcher=transport(calls))
    assert adapter.search("key", context=context()) == {}
    assert calls == runner.calls == []


def test_dynamic_replay_reuses_saved_plan_without_model_or_network(tmp_path):
    source, runner, calls = RunLog(tmp_path), Planner(), []
    live = EuropePMC(RunBudget(), source, runner, "live", fetcher=transport(calls))
    live.search("key", context=context())
    target, budget = RunLog(tmp_path), RunBudget()
    replay = EuropePMC(budget, target, None, "replay", source.run_id,
                       fetcher=lambda *a: pytest.fail("Replay must not send requests"))
    replay.search("key", context=context(needed="Review wording may differ in a fresh run"))
    assert replay.events[0]["replayed"]
    assert replay.events[0]["search_plan"] == live.events[0]["search_plan"]
    assert budget.counts["model_calls"] == budget.counts["search_requests"] == 0
    batch = next(v for n, v in load_artifacts(tmp_path, target.run_id) if n.endswith("_external_batch.json"))
    assert batch["events"][0]["search_query"] == live.events[0]["events"][0]["search_query"]


def test_dynamic_replay_does_not_use_same_device_snapshot_for_different_question(tmp_path):
    source = RunLog(tmp_path)
    live = EuropePMC(RunBudget(), source, Planner(), "live", fetcher=transport([]))
    live.search("key", context=context())
    replay = EuropePMC(RunBudget(), RunLog(tmp_path), None, "replay", source.run_id)
    assert replay.search("key", context=context("另一个问题")) == {}
    assert replay.events[-1]["status"] == "replay_miss"
    replay.search("key", context=context())
    assert replay.events[-1]["replayed"]


@pytest.mark.parametrize("error,category", [(InvalidModelOutput("bad plan"), "format"),
                                           (ModelUnavailable("model offline"), "model")])
def test_planner_failure_is_recorded_without_fixed_query_fallback(tmp_path, error, category):
    calls = []
    adapter = EuropePMC(RunBudget(), RunLog(tmp_path), Planner(error=error), "live", fetcher=transport(calls))
    assert adapter.search("key", context=context()) == {}
    assert calls == []
    assert adapter.events[0]["status"] == "technical_failure"
    assert adapter.events[0]["category"] == category


def test_failed_transport_still_records_actual_generated_query(tmp_path):
    def fail(*a):
        raise ExternalFailure("network down")
    log = RunLog(tmp_path)
    adapter = EuropePMC(RunBudget(), log, Planner(), "live", fetcher=fail)
    adapter.search("key", context=context())
    assert adapter.events[0]["status"] == "technical_failure"
    assert 'TITLE_ABS:"risk" AND TITLE_ABS:"factors"' in adapter.events[0]["events"][0]["search_query"]


def test_literal_terms_cannot_change_query_operators():
    from app.external_query import compile_query
    value = plan()
    value["concept_groups"] = [['mask" OR TITLE:unrelated']]
    actual = compile_query(value)
    assert 'TITLE_ABS:"mask\\"" AND TITLE_ABS:"OR" AND TITLE_ABS:"TITLE:unrelated"' in actual


def test_zero_hits_can_rephrase_once_without_losing_requested_constraints(tmp_path):
    class Replanner(Planner):
        def call(self, prompt, payload, schema, name, **kwargs):
            value = super().call(prompt, payload, schema, name, **kwargs)
            if payload.get("previous_plan"):
                value.update(concept_groups=[["pressure injury"], ["NIV"]],
                             date_from="", date_to="", publication_types=[], exclude_publication_types=[])
            return value
    calls, runner = [], Replanner()
    adapter = EuropePMC(RunBudget(), RunLog(tmp_path), runner, "live", fetcher=transport(calls))
    adapter.search("key", context=context())
    assert len(calls) == len(runner.calls) == 2
    actual = parse_qs(urlsplit(calls[1][0]).query)["query"][0]
    assert 'TITLE_ABS:"NIV"' in actual
    assert 'FIRST_PDATE:[2026-01-01 TO 2026-12-31]' in actual
    assert 'PUB_TYPE:"Journal Article"' in actual and 'NOT (PUB_TYPE:"Review")' in actual
    assert len(adapter.events[0]["search_plans"]) == 2


def test_no_date_or_study_filter_without_an_explicit_request(tmp_path):
    from jsonschema import validate, ValidationError
    from app.external_query import query_schema
    schema = query_schema("成人NIV面罩压力性损伤有哪些护理研究证据？")
    with pytest.raises(ValidationError):
        validate(plan(), schema)
    value = plan()
    value.update(date_from="", date_to="", publication_types=[], exclude_publication_types=[])
    validate(value, schema)
    validate(plan(), query_schema(context()["question"]))
