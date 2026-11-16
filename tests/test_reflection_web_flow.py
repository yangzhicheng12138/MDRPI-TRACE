"""Six workflow scenarios with explicit model/transport controls, not clinical labels."""

import json

import pytest

import app.reflection_workflow as workflow
from app.evidence_snapshot import load_artifacts
from app.external_evidence import EuropePMC, ExternalFailure
from app.six_step_schema import empty_steps
from test_reflection_v2 import ScriptedLLM, claim, wire_pass
from test_rag import config, evidence, pipeline


class WebFlowLLM(ScriptedLLM):
    def __init__(self, scenario):
        super().__init__()
        self.scenario = scenario
        self.revised = False
        self.query_plans = []

    def structured(self, messages, schema, name, **kwargs):
        payload = json.loads(messages[1]["content"])
        if name == "external_query_plan":
            self.query_plans.append(payload)
            return {"evidence_question": "Adult medical device pressure injury prevention management",
                    "concept_groups": [["pressure injury", "pressure ulcer"], ["NIV", "ventilation"]],
                    "date_from": "", "date_to": "", "publication_types": [], "exclude_publication_types": []}
        if name.startswith("external_admission"):
            return {"decision": "admit", "applicability": "Synthetic adult study", "limitations": "Test only"}
        if name.startswith("six_step_"):
            step = payload["step"]
            value = empty_steps()[step]
            if step == "solutions":
                self.revised = "previous_step" in payload
                if self.scenario in {"sufficient", "conflict"} or self.revised:
                    sources = payload["sources"]
                    chosen = next((s for s in sources if s["evidence_id"].startswith("EXT_")), sources[0])
                    c = claim("solutions.c1")
                    c.update(kind="source_statement", fact_ids=[], evidence_ids=[chosen["evidence_id"]],
                             text="New source qualified statement" if self.revised else "Local source statement")
                    value["claims"] = [c]
                    value["coverage"]["measures"].update(status="partial", claim_ids=[c["claim_id"]])
            return value
        if name.startswith("audit_"):
            rows = {target: wire_pass() for target in payload["targets"]}
            target = "analyze.information" if self.scenario == "patient" else "solutions.measures"
            if name == "audit_case" and target in rows and self.scenario != "sufficient" and not self.revised:
                rows[target] = {"finding": "source_conflict" if self.scenario == "conflict" else "content_omission",
                                "severity": "minor", "assessment": "Development gap",
                                "needed": "Assessment time" if self.scenario == "patient" else "Additional evidence",
                                "route": "patient" if self.scenario == "patient" else
                                         "conflict" if self.scenario == "conflict" else "knowledge"}
            return rows
        if name == "case_facts" and self.scenario == "patient":
            value = super().structured(messages, schema, name, **kwargs)
            value["missing_information"] = ["Assessment time"]
            return value
        return super().structured(messages, schema, name, **kwargs)


def setup_flow(tmp_path, monkeypatch, scenario):
    calls = []
    xml = (b'<article><license>CC BY</license><body><sec><title>Results</title>'
           b'<p>Adults with medical device related pressure injury received prevention management and monitoring care.</p>'
           b'</sec></body></article>')

    def fetcher(url, budget, kind):
        budget.consume(kind)
        calls.append(kind)
        if scenario == "failure":
            raise ExternalFailure("Injected network outage")
        if kind == "search_requests":
            data = {"resultList": {"result": [{"pmcid": "PMC123", "pmid": "1", "license": "cc by",
                    "title": "Synthetic adult development study"}]}}
            return json.dumps(data).encode(), url
        return xml, url

    def adapter(*args, **kwargs):
        return EuropePMC(*args, **kwargs, fetcher=fetcher)

    monkeypatch.setattr(workflow, "EuropePMC", adapter)
    cfg = config()
    cfg.values["reflection"] = {"log_dir": str(tmp_path)}
    engine = pipeline(cfg=cfg, llm=WebFlowLLM(scenario))
    return engine, calls


@pytest.mark.parametrize("scenario", ["sufficient", "patient", "knowledge", "conflict"])
def test_research_api_explicit_on_demand_preserves_existing_trigger(tmp_path, monkeypatch, scenario):
    """Keep the existing optional Europe PMC path; daily mode still defaults live."""
    from fastapi.testclient import TestClient
    from app.api import create_app
    engine, calls = setup_flow(tmp_path, monkeypatch, scenario)
    with TestClient(create_app(engine.config, engine)) as client:
        response = client.post("/v1/research/answer", json={"query": "NIV 皮肤评估", "web_strategy": "on_demand"})
    assert response.status_code == 200
    result = response.json()
    assert result["web"]["mode"] == "live"
    artifacts = load_artifacts(tmp_path, result["run_id"])
    inputs = next(v for name, v in artifacts if name.endswith("_input_config.json"))
    assert inputs["web_mode"] == "live"
    if scenario in {"sufficient", "patient"}:
        assert calls == [] and result["timing"]["search_requests"] == 0
        if scenario == "patient":
            assert result["questions"] == ["Assessment time"]
    else:
        assert calls[0] == "search_requests"
        triggers = [v for name, v in artifacts if name.endswith("_web_trigger.json")]
        assert triggers[0]["route"] == ("conflict" if scenario == "conflict" else "knowledge")


@pytest.mark.parametrize("scenario", ["sufficient", "knowledge", "patient", "conflict", "failure", "replay"])
@pytest.mark.parametrize("research_mode", [False, True])
def test_six_web_scenarios_through_workflow(tmp_path, monkeypatch, scenario, research_mode):
    engine, calls = setup_flow(tmp_path, monkeypatch, scenario)
    result = workflow.reflective_answer(engine, "NIV 皮肤评估", web_mode="live", research_mode=research_mode)
    artifacts = load_artifacts(tmp_path, result["run_id"])
    triggers = [value for name, value in artifacts if name.endswith("_web_trigger.json")]
    if scenario in {"sufficient", "patient"}:
        assert calls == [] and triggers == []
        assert engine.llm.query_plans == []
        if scenario == "patient":
            assert result["questions"] == ["Assessment time"]
    else:
        assert calls[0] == "search_requests" and triggers
        assert engine.llm.query_plans[0]["question"] == "NIV 皮肤评估"
        assert engine.llm.query_plans[0]["gaps"][0]["route"] in {"knowledge", "conflict"}
        assert triggers[0]["route"] == ("conflict" if scenario == "conflict" else "knowledge")
        assert any(name.endswith("_routing_1.json") for name, _ in artifacts)
        assert any(name.endswith("_web_result.json") for name, _ in artifacts)
        if scenario == "failure":
            assert result["status"] == "technical_failure"
            assert any(e["category"] == "network" for e in result["errors"])
            assert not result["citations"]
        else:
            assert result["status"] == "partial" and result["timing"]["revision_rounds"] == 1
            assert any(c["origin"] == "external" for c in result["citations"])
            assert any(name.endswith("_changes_1.json") for name, _ in artifacts)
            assert any(name.endswith("_audits_1.json") for name, _ in artifacts)
    if scenario == "replay":
        calls.clear()
        engine.llm = WebFlowLLM("replay")
        replay = workflow.reflective_answer(engine, "NIV 皮肤评估", web_mode="replay", replay_run_id=result["run_id"],
                                            research_mode=research_mode)
        assert calls == []
        assert replay["citations"] == result["citations"]
        assert replay["six_steps"] == result["six_steps"]
        assert replay["timing"]["search_requests"] == replay["timing"]["fetch_requests"] == 0
        assert engine.llm.query_plans == []


@pytest.mark.parametrize("research_mode", [False, True])
@pytest.mark.parametrize("gap_persists", [False, True])
@pytest.mark.parametrize("local_count", [8, 14])
def test_local_passage_rotation_does_not_prevent_needed_web_search(tmp_path, monkeypatch, research_mode, gap_persists, local_count):
    """Controlled model, corpus and transport; this is not real network evidence."""
    class LocalGapLLM(WebFlowLLM):
        external_seen = False

        def structured(self, messages, schema, name, **kwargs):
            payload = json.loads(messages[1]["content"])
            if name == "six_step_solutions":
                self.external_seen = any(s["evidence_id"].startswith("EXT_") for s in payload["sources"])
            value = super().structured(messages, schema, name, **kwargs)
            if (gap_persists and name == "audit_case" and "solutions.measures" in value
                    and not self.external_seen):
                value["solutions.measures"] = {
                    "assessment": "The local supplement did not resolve the professional knowledge gap.",
                    "finding": "content_omission", "severity": "minor", "route": "knowledge",
                    "needed": "Additional professional evidence is needed.",
                }
            return value

    engine, calls = setup_flow(tmp_path, monkeypatch, "knowledge")
    engine.llm = LocalGapLLM("knowledge")
    if research_mode:
        # The faster research setting must still allow the existing bounded
        # local-supplement -> persistent gap -> external evidence follow-up.
        engine.config.values["reflection"]["research_max_revision_rounds"] = 1
    # More results than the existing six-passage generation context. The same
    # local set is returned on every retrieval, with no new corpus evidence.
    engine.config.values["rag"]["max_evidence"] = local_count
    engine.retriever.items = [evidence(f"DOC{i:03d}_P0001_C001") for i in range(1, local_count + 1)]
    result = workflow.reflective_answer(engine, "NIV 皮肤评估", web_mode="live", research_mode=research_mode)
    artifacts = load_artifacts(tmp_path, result["run_id"])
    triggers = [v for n, v in artifacts if n.endswith("_web_trigger.json")]
    if gap_persists:
        assert calls == ["search_requests", "fetch_requests"]
        assert len(triggers) == 1 and triggers[0]["round"] == 2
        assert any(c["origin"] == "external" for c in result["citations"])
    else:
        assert calls == [] and triggers == []
        assert result["timing"]["revision_rounds"] == 1
