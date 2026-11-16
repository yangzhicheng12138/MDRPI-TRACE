"""Controlled engineering tests; these are not real model validation results."""
from copy import deepcopy
import json

import pytest

from app.local_llm import InvalidModelOutput
from app.six_step_schema import STEPS, empty_steps, validate_steps
from test_reflection_v2 import facts, claim, steps_fixture


def complete_steps():
    result = steps_fixture()
    for step in STEPS:
        if not result[step]["claims"]:
            c = claim(step + ".c1")
            c.update(kind="case_fact" if step == "recognize" else "model_inference", evidence_ids=[])
            result[step]["claims"] = [c]
            first = next(iter(result[step]["coverage"]))
            result[step]["coverage"][first].update(status="partial", claim_ids=[c["claim_id"]])
    return result


def sources():
    return {"E1": {"evidence_id": "E1", "text": "Inspect skin without compromising treatment.", "origin": "local"}}


def focused_wire(targets, issues=()):
    values = {cid: "pass" for cid in targets}
    values.update({row["check_id"]: {k: v for k, v in row.items() if k != "check_id"} for row in issues})
    return values


def test_focused_generation_contract_requires_evidence_and_safety_keys():
    from jsonschema import validate, ValidationError
    from app.reflection_checks import focused_review_schema
    targets = ["case:F001", "evidence:solutions.c1", "safety:action.c1"]
    with pytest.raises(ValidationError):
        validate({"case:F001": "pass"}, focused_review_schema(targets))
    validate(focused_wire(targets), focused_review_schema(targets))


def test_grouped_generation_keeps_all_six_steps_and_dependencies():
    from app.six_step_answer import generate_research_steps
    calls = []
    class Runner:
        def call(self, prompt, payload, schema, name, **kwargs):
            calls.append((payload, name))
            return complete_steps()
    result = generate_research_steps(Runner(), "成人使用NIV", facts(), sources())
    validate_steps(result, facts(), sources(), research_mode=True)
    assert len(calls) == 1 and set(result) == set(STEPS)
    assert result["action"]["claims"][0]["depends_on"] == ["solutions.c1"]


def test_grouped_revision_preserves_unrequested_steps():
    from app.six_step_answer import generate_research_steps
    before = complete_steps()
    updated = deepcopy(before["solutions"])
    updated["claims"][0]["text"] = "修改后的措施，保持治疗连续性。"
    class Runner:
        def call(self, prompt, payload, schema, name, **kwargs):
            assert payload["steps"] == ["solutions"]
            return {"solutions": updated}
    after = generate_research_steps(Runner(), "成人使用NIV", facts(), sources(), before,
        revisions={"solutions": {"previous_step": before["solutions"], "affected_claim_ids": ["solutions.c1"],
                                 "affected_requirements": ["measures"], "issues": []}})
    assert after["solutions"] == updated
    assert all(after[s] == before[s] for s in STEPS if s != "solutions")
    assert before["solutions"]["claims"][0]["text"] != updated["claims"][0]["text"]


class ReviewRunner:
    def __init__(self):
        self.calls = []
        self.scope_calls = []
    def call(self, prompt, payload, schema, name, **kwargs):
        if name == "research_task_scope":
            self.scope_calls.append(deepcopy(payload))
            return {"literature_requested": True, "requirement_quote": "Controlled general study request", "assigned_steps": ["unspecified"]}
        self.calls.append(deepcopy(payload))
        return focused_wire(payload["targets"])


def test_focused_review_covers_three_kinds_in_one_call_and_reuses_unchanged():
    from app.reflection_checks import run_checks
    runner, cache = ReviewRunner(), {}
    before = run_checks(runner, "成人使用NIV", facts(), complete_steps(), sources(),
                        research_mode=True, compact=True, cache=cache)
    after = run_checks(runner, "成人使用NIV", facts(), complete_steps(), sources(),
                       research_mode=True, compact=True, cache=cache)
    assert len(runner.calls) == 1
    assert {r["check_kind"] for r in after} == {"case", "evidence", "safety"}
    assert before == after and len({r["check_id"] for r in after}) == len(after)


def test_compact_review_omits_duplicate_target_metadata_but_keeps_full_content():
    from app.reflection_checks import run_checks
    runner = ReviewRunner()
    steps, evidence = complete_steps(), sources()
    run_checks(runner, "成人使用NIV", facts(), steps, evidence, research_mode=True, compact=True)
    sent = runner.calls[0]
    assert sent["steps"] == steps and sent["facts"] == facts()
    assert sent["sources"][0]["text"] == evidence["E1"]["text"]
    assert sent["targets"]["case:solutions.measures"] == STEPS["solutions"][1]["measures"]
    assert sent["targets"]["evidence:solutions.c1"] == "evidence"


def test_changed_source_and_dependency_are_reaudited():
    from app.reflection_checks import run_checks
    runner, cache = ReviewRunner(), {}
    steps = complete_steps()
    run_checks(runner, "成人使用NIV", facts(), steps, sources(), research_mode=True, compact=True, cache=cache)
    changed = deepcopy(steps)
    changed["solutions"]["claims"][0]["text"] = "更改了前置方案"
    evidence = sources(); evidence["E1"]["text"] = "Different original passage."
    run_checks(runner, "成人使用NIV", facts(), changed, evidence, research_mode=True, compact=True, cache=cache)
    targets = runner.calls[-1]["targets"]
    assert "evidence:solutions.c1" in targets and "safety:action.c1" in targets
    assert "case:F001" not in targets


def test_focused_review_never_turns_missing_target_into_pass():
    from app.reflection_checks import run_checks
    class Missing(ReviewRunner):
        def call(self, prompt, payload, schema, name, **kwargs):
            return focused_wire(list(payload["targets"])[1:])
    with pytest.raises(InvalidModelOutput):
        run_checks(Missing(), "成人使用NIV", facts(), complete_steps(), sources(), research_mode=True, compact=True)


@pytest.mark.parametrize("change", ["condition", "source"])
def test_changed_parent_condition_invalidates_dependent_review(change):
    from app.reflection_checks import run_checks
    runner, cache = ReviewRunner(), {}
    steps = complete_steps()
    steps["action"]["claims"][0].update(evidence_ids=[], kind="model_inference")
    evidence = sources()
    run_checks(runner, "成人使用NIV", facts(), steps, evidence, research_mode=True, compact=True, cache=cache)
    if change == "condition":
        steps["solutions"]["claims"][0]["conditions"] = "New prerequisite for the dependent action"
    else:
        evidence["E1"]["text"] = "The parent's source has changed"
    run_checks(runner, "成人使用NIV", facts(), steps, evidence, research_mode=True, compact=True, cache=cache)
    assert "safety:action.c1" in runner.calls[-1]["targets"]
    assert "evidence:action.c1" in runner.calls[-1]["targets"]


def test_focused_review_preserves_real_issue_for_existing_router():
    from app.reflection_checks import run_checks
    from app.evidence_router import retrieval_routes
    class Issue(ReviewRunner):
        def call(self, prompt, payload, schema, name, **kwargs):
            cid = "evidence:solutions.c1"
            return focused_wire(payload["targets"], [{
                "check_id": cid, "assessment": "原文不支持这个数值", "finding": "numeric_error",
                "severity": "major", "route": "correction", "needed": "去掉无依据数值"}])
    records = run_checks(Issue(), "成人使用NIV", facts(), complete_steps(), sources(), research_mode=True, compact=True)
    issues = [r for r in records if r["verdict"] == "issue"]
    assert len(issues) == 1 and issues[0]["target_id"] == "solutions.c1"
    assert retrieval_routes(issues, research_mode=True)["corrections"] == ["evidence:solutions.c1"]


def test_focused_context_split_retains_every_target():
    from app.reflection_checks import run_checks
    class Split(ReviewRunner):
        def call(self, prompt, payload, schema, name, **kwargs):
            if len(payload["targets"]) > 15:
                raise InvalidModelOutput("Structured context budget exceeded")
            return super().call(prompt, payload, schema, name, **kwargs)
    runner = Split()
    records = run_checks(runner, "成人使用NIV", facts(), complete_steps(), sources(), research_mode=True, compact=True)
    checked = [cid for call in runner.calls for cid in call["targets"]]
    assert len(checked) == len(set(checked)) == len(records)


@pytest.mark.parametrize("fail_review,persistent", [(False, None), (True, None), (False, "minor"), (False, "major")])
def test_compact_workflow_revises_and_preserves_audited_fallback(tmp_path, fail_review, persistent):
    from app.reflection_workflow import reflective_answer
    from app.evidence_snapshot import load_artifacts
    from test_reflection_v2 import ScriptedLLM
    from test_rag import config, pipeline
    class LLM(ScriptedLLM):
        revised = False
        revisions = 0
        def structured(self, messages, schema, name, **kwargs):
            payload = json.loads(messages[1]["content"])
            if name == "case_facts":
                return {"classifications": {key: {"category": "device" if i == 0 else "not_fact", "state": "present"}
                        for i, key in enumerate(payload["candidate_clauses"])}, "missing_information": [], "case_conflicts": []}
            if name in {"six_steps_draft", "six_steps_revision"}:
                self.revised = name == "six_steps_revision"
                self.revisions += int(self.revised)
                values = deepcopy(payload.get("previous_steps", complete_steps()))
                for step in values.values():
                    for c in step["claims"]:
                        c.update(evidence_ids=[], kind="model_inference")
                values["solutions"]["claims"][0]["text"] = f"Corrected condition {self.revisions}" if self.revised else "Missing condition"
                return {s: values[s] for s in payload["steps"]}
            if name == "audit_focused":
                if self.revised and fail_review:
                    raise InvalidModelOutput("Synthetic final audit failure")
                cid = "evidence:solutions.c1"
                issues = [] if self.revised and not persistent else [{"check_id": cid, "assessment": "原文条件未补回",
                          "finding": "unsupported_claim" if self.revised else "missing_condition",
                          "severity": persistent or "major", "route": "correction", "needed": "补回条件"}]
                return focused_wire(payload["targets"], issues)
            raise AssertionError(name)
    cfg = config(); cfg.values["reflection"] = {"log_dir": str(tmp_path), "research_compact": True}
    result = reflective_answer(pipeline(cfg=cfg, llm=LLM()), "成人使用NIV面罩。请分析皮肤评估。", research_mode=True, web_mode="off")
    assert set(result["six_steps"]) == set(STEPS)
    assert result["timing"]["model_calls"] == (7 if persistent == "major" else 5)
    assert result["timing"]["search_requests"] == result["timing"]["fetch_requests"] == 0
    text = result["six_steps"]["solutions"]["claims"][0]["text"]
    assert text == ("Missing condition" if fail_review else f"Corrected condition {2 if persistent == 'major' else 1}")
    if persistent == "minor":
        assert result["stop_reason"] == "minor_issues_after_revision"
        assert result["unresolved_issues"] and result["timing"]["revision_rounds"] == 1
    assert bool(result["errors"]) is fail_review
    artifacts = load_artifacts(tmp_path, result["run_id"])
    assert any(name.endswith("_candidate_1.json") for name, _ in artifacts)
    assert any(name.endswith("_audits_0.json") for name, _ in artifacts)


@pytest.mark.parametrize("scenario", ["sufficient", "knowledge", "patient", "conflict", "failure", "replay"])
def test_compact_flow_keeps_web_and_replay_boundaries(tmp_path, monkeypatch, scenario):
    from app.reflection_workflow import reflective_answer
    from test_reflection_web_flow import WebFlowLLM, setup_flow
    class CompactWebLLM(WebFlowLLM):
        def structured(self, messages, schema, name, **kwargs):
            payload = json.loads(messages[1]["content"])
            if name in {"six_steps_draft", "six_steps_revision"}:
                values = {}
                for step in payload["steps"]:
                    adapted = {**payload, "step": step}
                    if name == "six_steps_revision":
                        adapted["previous_step"] = payload["previous_steps"][step]
                    messages2 = [messages[0], {"role": "user", "content": json.dumps(adapted)}]
                    value = super().structured(messages2, schema, "six_step_" + step, **kwargs)
                    if not value["claims"]:
                        c = claim(step + ".c1")
                        c.update(kind="model_inference", fact_ids=[], evidence_ids=[], text="Synthetic development observation")
                        value["claims"] = [c]
                    values[step] = value
                return values
            if name == "audit_focused":
                passed, issues = [], []
                for kind in ("case", "evidence", "safety"):
                    subset = {c.split(":", 1)[1]: {} for c in payload["targets"] if c.startswith(kind + ":")}
                    adapted = {**payload, "targets": subset}
                    rows = super().structured([messages[0], {"role": "user", "content": json.dumps(adapted)}],
                                              schema, "audit_" + kind, **kwargs)
                    for target, row in rows.items():
                        cid = kind + ":" + target
                        if row["finding"] == "none":
                            passed.append(cid)
                        else:
                            issues.append({"check_id": cid, **row})
                return focused_wire(passed, issues)
            return super().structured(messages, schema, name, **kwargs)
    engine, calls = setup_flow(tmp_path, monkeypatch, scenario)
    engine.config.values["reflection"]["research_compact"] = True
    engine.llm = CompactWebLLM(scenario)
    result = reflective_answer(engine, "NIV 皮肤评估", research_mode=True, web_mode="live")
    assert all(result["six_steps"][s]["claims"] for s in STEPS)
    if scenario in {"sufficient", "patient"}:
        assert calls == []
        if scenario == "patient":
            assert result["questions"] == ["Assessment time"]
    elif scenario == "failure":
        assert any(e["category"] == "network" for e in result["errors"])
    else:
        assert calls and not result["errors"], result["errors"]
        assert any(e["origin"] == "external" for e in result["citations"])
    if scenario == "replay":
        calls.clear(); engine.llm = CompactWebLLM(scenario)
        replay = reflective_answer(engine, "NIV 皮肤评估", research_mode=True, web_mode="replay", replay_run_id=result["run_id"])
        assert calls == [] and replay["citations"] == result["citations"]
        assert replay["timing"]["search_requests"] == replay["timing"]["fetch_requests"] == 0
