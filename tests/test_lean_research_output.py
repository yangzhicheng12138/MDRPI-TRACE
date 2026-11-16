"""Controlled contract tests, not real model answers or research evaluation data."""
from copy import deepcopy
import json

import pytest
from jsonschema import validate, ValidationError

from app.local_llm import InvalidModelOutput
from app.six_step_schema import STEPS, render_answer, validate_steps
from test_research_latency import complete_steps, sources
from test_reflection_v2 import facts


def lean_draft():
    values = complete_steps()
    original = values["solutions"]["claims"][0]
    for i in (2, 3):
        c = deepcopy(original)
        c.update(claim_id=f"solutions.c{i}", text=f"Controlled distinct measure {i}")
        values["solutions"]["claims"].append(c)
    result = {}
    for step, value in values.items():
        result[step] = {"claims": [
            {k: v for k, v in c.items() if k not in {"conditions", "rationale"}}
            for c in value["claims"]], "coverage": deepcopy(value["coverage"])}
        for key, row in list(result[step]["coverage"].items()):
            if row["status"] == "supported" and row["claim_ids"]:
                result[step]["coverage"][key] = row["claim_ids"]
    result["solutions"]["claims"][0]["text"] = "先保持治疗连续性，再调整固定；依据为已提供的观察原则。"
    result["solutions"]["coverage"]["measures"] = [f"solutions.c{i}" for i in (1, 2, 3)]
    return result


def complete_wire(value):
    return {"claims": [{k: v for k, v in c.items() if k not in {"conditions", "rationale"}}
                       for c in value["claims"]],
            "coverage": {key: deepcopy(row["claim_ids"] if row["status"] == "supported" else row)
                         for key, row in value["coverage"].items()}}


class Runner:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def call(self, prompt, payload, schema, name, **kwargs):
        self.calls.append((payload, schema))
        validate(self.output, schema)
        return deepcopy(self.output)


@pytest.mark.parametrize("group_limit,observed_cutoff", [(6, 5000), (3, 3250), (1, 1350)])
def test_draft_budget_exceeds_observed_truncation_after_context_split(group_limit, observed_cutoff):
    """DEV-RV04 cut off at all three ceilings; short wire is not short content."""
    from app.six_step_answer import generate_research_steps
    calls = []
    class SplitByContext:
        def call(self, prompt, payload, schema, name, **kwargs):
            group = payload["steps"]
            if len(group) > group_limit:
                raise InvalidModelOutput("Structured context budget exceeded")
            calls.append((group, kwargs["max_tokens"]))
            assert kwargs["max_tokens"] > observed_cutoff
            output = {step: lean_draft()[step] for step in group}
            validate(output, schema)
            return output
    after = generate_research_steps(SplitByContext(), "成人使用NIV", facts(), sources(), lean=True)
    assert set(after) == set(STEPS)
    assert len(calls) == 6 // group_limit
    assert all(len(value["claims"]) >= 1 for value in after.values())


def test_revision_returns_complete_step_without_delete_commands():
    from app.six_step_answer import generate_research_steps
    before = complete_steps()
    output = {"action": lean_draft()["action"]}
    runner = Runner(output)
    after = generate_research_steps(runner, "成人使用NIV", facts(), sources(), before,
        revisions={"action": {"affected_claim_ids": ["action.c1"], "affected_requirements": ["checks"], "issues": []}}, lean=True)
    assert after["action"]["claims"][0]["text"] == output["action"]["claims"][0]["text"]
    assert all(after[s] == before[s] for s in STEPS if s != "action")
    schema = runner.calls[0][1]
    assert "remove" not in schema["properties"]["action"]["properties"]
    blank = deepcopy(output); blank["action"]["claims"] = []
    with pytest.raises(ValidationError):
        validate(blank, schema)


def test_research_revision_receives_available_source_identity_without_changing_passage():
    """A retrieved study's traceable identity must reach the actual revision call."""
    from app.six_step_answer import generate_research_steps
    before = complete_steps()
    evidence = sources()
    study = {"evidence_id": "EXT_CONTROLLED", "text": "Controlled complete study passage.",
             "title": "Controlled study", "year": "2026", "origin": "external",
             "authors": "Controlled author", "organization": "Controlled journal",
             "doi": "10.0000/controlled", "source_url": "https://example.invalid/study"}
    evidence[study["evidence_id"]] = study
    runner = Runner({"action": complete_wire(before["action"])})
    generate_research_steps(runner, "虚拟任务：补充可核对研究", facts(), evidence, before,
        revisions={"action": {"affected_claim_ids": ["action.c1"],
                              "affected_requirements": [], "issues": []}}, lean=True)
    sent = next(s for s in runner.calls[0][0]["sources"] if s["evidence_id"] == study["evidence_id"])
    for key in ("title", "year", "text", "authors", "organization", "doi", "source_url"):
        assert sent.get(key) == study[key]
    assert evidence[study["evidence_id"]] == study


def test_revision_omits_unrequested_coverage_context_but_preserves_full_answer():
    from app.six_step_answer import generate_research_steps
    before = complete_steps()
    class ReviewAction:
        def call(self, prompt, payload, schema, name, **kwargs):
            for step, value in payload["previous_steps"].items():
                assert value["claims"] == before[step]["claims"]
                if step == "action":
                    assert value["coverage"] == before[step]["coverage"]
                else:
                    assert "coverage" not in value
            assert payload["sources"]
            return {"action": complete_wire(before["action"])}
    after = generate_research_steps(ReviewAction(), "成人使用NIV", facts(), sources(), before,
        revisions={"action": {"affected_claim_ids": ["action.c1"], "affected_requirements": [], "issues": []}}, lean=True)
    assert all(after[step] == before[step] for step in STEPS if step != "action")
    assert complete_wire(after["action"]) == complete_wire(before["action"])


def test_single_revision_context_recovery_keeps_prose_citations_and_output_room():
    """A long revision must fit without clipping text or reducing its output."""
    from app.six_step_answer import generate_research_steps
    before = complete_steps()
    before["evaluate"]["claims"][0].update(conditions="Keep this condition", rationale="Keep this reason")
    original = deepcopy(before)
    evidence = sources()
    required = {eid for s in before.values() for c in s["claims"] for eid in c["evidence_ids"]}
    extra = deepcopy(next(iter(evidence.values())))
    extra.update(evidence_id="optional_passage", text="Optional unrelated passage " * 40)
    evidence[extra["evidence_id"]] = extra
    calls = []
    class LongContext:
        def call(self, prompt, payload, schema, name, **kwargs):
            calls.append((deepcopy(payload), kwargs["max_tokens"]))
            visible = {s["evidence_id"]: s for s in payload["sources"]}
            assert required <= visible.keys()
            for eid in required:
                assert visible[eid]["text"] == evidence[eid]["text"]
            assert payload["previous_steps"]["evaluate"]["claims"][0]["conditions"] == "Keep this condition"
            for step, value in payload["previous_steps"].items():
                for actual, expected in zip(value["claims"], before[step]["claims"]):
                    assert {**actual, "conditions": actual.get("conditions", ""),
                            "rationale": actual.get("rationale", "")} == expected
            empty_fields = any(c.get(k) == "" for s in payload["previous_steps"].values()
                               for c in s["claims"] for k in ("conditions", "rationale"))
            if "optional_passage" in visible or empty_fields:
                raise InvalidModelOutput("Structured context budget exceeded")
            output = {"evaluate": complete_wire(before["evaluate"])}
            validate(output, schema)
            unavailable = deepcopy(output)
            unavailable["evaluate"]["claims"][0]["evidence_ids"] = ["optional_passage"]
            with pytest.raises(ValidationError):
                validate(unavailable, schema)
            return output
    after = generate_research_steps(LongContext(), "成人使用NIV", facts(), evidence, before,
        revisions={"evaluate": {"affected_claim_ids": ["evaluate.c1"], "affected_requirements": [], "issues": []}}, lean=True)
    assert len(calls) >= 2
    assert len({budget for _, budget in calls}) == 1
    assert all("contract_error_to_correct" not in payload for payload, _ in calls)
    assert before == original
    assert all(after[s] == before[s] for s in STEPS if s != "evaluate")
    assert complete_wire(after["evaluate"]) == complete_wire(before["evaluate"])


def test_single_revision_unavoidable_context_failure_does_not_increase_output_budget():
    from app.six_step_answer import generate_research_steps
    before = complete_steps()
    budgets = []
    class TooLarge:
        def call(self, prompt, payload, schema, name, **kwargs):
            budgets.append(kwargs["max_tokens"])
            assert payload["sources"]  # Required original evidence is never removed.
            assert "contract_error_to_correct" not in payload
            raise InvalidModelOutput("Structured context budget exceeded")
    with pytest.raises(InvalidModelOutput, match="Structured context budget exceeded"):
        generate_research_steps(TooLarge(), "成人使用NIV", facts(), sources(), before,
            revisions={"evaluate": {"affected_claim_ids": ["evaluate.c1"], "affected_requirements": [], "issues": []}}, lean=True)
    assert len(set(budgets)) == 1
    assert len(budgets) <= 2


@pytest.mark.parametrize("enabled", [False, True])
def test_revision_fingerprint_recovery_preserves_clinical_payload_and_legacy_branch(enabled):
    """An opt-in overflow recovery removes only machine hashes, not evidence."""
    from app.six_step_answer import generate_research_steps
    before, evidence = complete_steps(), sources()
    for source in evidence.values():
        source["evidence_scope"] = {
            "document_scope": {"population": ["adult"], "device_type": ["niv_mask"]},
            "passage_cues": {"conditions": [{"quote": "Only when treatment remains effective"}]},
            "scope_limit": "Retain the passage's conditions.",
            "source_location": {"source_pages": [7], "section": "Clinical conditions",
                "excerpt_start": 10, "excerpt_end": 99, "content_sha256": "a" * 64,
                "source_pdf_sha256": "b" * 64, "source_page_text_sha256": "c" * 64},
            "source_review_scope": {"applicability": "Adults; preserve treatment continuity"},
        }
    original_before, original_evidence = deepcopy(before), deepcopy(evidence)
    calls = []

    class ContextLimit:
        def call(self, prompt, payload, schema, name, **kwargs):
            calls.append((deepcopy(payload), kwargs["max_tokens"]))
            if any("content_sha256" in s["evidence_scope"]["source_location"]
                   for s in payload["sources"]):
                raise InvalidModelOutput("Structured context budget exceeded")
            expected = deepcopy(calls[0][0])
            for source in expected["sources"]:
                for key in ("content_sha256", "source_pdf_sha256", "source_page_text_sha256"):
                    source["evidence_scope"]["source_location"].pop(key)
            assert payload == expected
            output = {"evaluate": complete_wire(before["evaluate"])}
            validate(output, schema)
            return output

    kwargs = dict(revisions={"evaluate": {"affected_claim_ids": ["evaluate.c1"],
                  "affected_requirements": [], "issues": []}}, lean=True,
                  compact_revision_provenance=enabled)
    if enabled:
        after = generate_research_steps(ContextLimit(), "成人使用NIV", facts(), evidence, before, **kwargs)
        assert len(calls) == 2
        assert complete_wire(after["evaluate"]) == complete_wire(before["evaluate"])
        assert all(after[s] == before[s] for s in STEPS if s != "evaluate")
    else:
        with pytest.raises(InvalidModelOutput, match="Structured context budget exceeded"):
            generate_research_steps(ContextLimit(), "成人使用NIV", facts(), evidence, before, **kwargs)
        assert all("content_sha256" in s["evidence_scope"]["source_location"]
                   for payload, _ in calls for s in payload["sources"])
    assert len({budget for _, budget in calls}) == 1
    assert before == original_before and evidence == original_evidence


def test_revision_fingerprint_projection_handles_absent_scope_without_removing_other_data():
    from app.six_step_answer import _revision_source_provenance_without_fingerprints
    values = [{"text": "Preserve content_sha256 in clinical prose", "evidence_scope": None},
              {"text": "Full passage", "evidence_scope": {"source_location": None}},
              {"text": "Another passage", "content_sha256": "top-level identity"}]
    assert _revision_source_provenance_without_fingerprints(values) == values


def test_revision_fingerprint_recovery_is_bounded_when_context_still_cannot_fit():
    from app.six_step_answer import generate_research_steps
    before, evidence = complete_steps(), sources()
    for source in evidence.values():
        source["evidence_scope"] = {"source_location": {"content_sha256": "a" * 64}}
    calls = []

    class TooLarge:
        def call(self, prompt, payload, schema, name, **kwargs):
            calls.append((deepcopy(payload), kwargs["max_tokens"]))
            assert "contract_error_to_correct" not in payload
            raise InvalidModelOutput("Structured context budget exceeded")

    with pytest.raises(InvalidModelOutput, match="Structured context budget exceeded"):
        generate_research_steps(TooLarge(), "成人使用NIV", facts(), evidence, before,
            revisions={"evaluate": {"affected_claim_ids": ["evaluate.c1"],
                       "affected_requirements": [], "issues": []}}, lean=True,
            compact_revision_provenance=True)
    # At most one hash projection and the existing empty-field recovery.
    assert 2 <= len(calls) <= 3
    assert len({budget for _, budget in calls}) == 1
    assert all(payload["sources"] for payload, _ in calls)


def test_lean_draft_preserves_model_text_sources_and_explicit_coverage():
    from app.six_step_answer import generate_research_steps
    raw = lean_draft()
    runner = Runner(raw)
    result = generate_research_steps(runner, "成人NIV皮肤观察", facts(), sources(), lean=True)
    validate_steps(result, facts(), sources(), research_mode=True)
    c = result["solutions"]["claims"][0]
    assert c["text"] == raw["solutions"]["claims"][0]["text"]
    assert c["evidence_ids"] == raw["solutions"]["claims"][0]["evidence_ids"]
    assert c["conditions"] == c["rationale"] == ""
    assert len(result["solutions"]["claims"]) == 3
    assert result["solutions"]["coverage"]["measures"]["status"] == "supported"
    text = render_answer({"answer_mode": "research_six_steps", "six_steps": result})
    assert text.count(c["text"]) == 1 and "适用条件：；" not in text
    assert result["recognize"]["coverage"]["unknowns"] == raw["recognize"]["coverage"]["unknowns"]


def test_lean_draft_cannot_silently_pass_missing_or_empty_coverage():
    from app.six_step_answer import generate_research_steps
    runner = Runner(lean_draft())
    generate_research_steps(runner, "成人NIV皮肤观察", facts(), sources(), lean=True)
    schema = runner.calls[0][1]
    for missing in (True, False):
        broken = lean_draft()
        if missing:
            del broken["solutions"]["coverage"]["measures"]
        else:
            broken["solutions"]["coverage"]["measures"] = []
        with pytest.raises(ValidationError):
            validate(broken, schema)


def test_complete_revision_preserves_returned_content_and_unrequested_steps():
    from app.six_step_answer import generate_research_steps
    before = complete_steps()
    for i in (2, 3):
        untouched = deepcopy(before["solutions"]["claims"][0])
        untouched.update(claim_id=f"solutions.c{i}", text=f"Unchanged condition {i}", conditions="", rationale="")
        before["solutions"]["claims"].append(untouched)
    output = {"solutions": complete_wire(before["solutions"])}
    output["solutions"]["claims"][0]["text"] = "Correct the error while maintaining treatment."
    runner = Runner(output)
    after = generate_research_steps(runner, "成人NIV皮肤观察", facts(), sources(), before,
        revisions={"solutions": {"affected_claim_ids": ["solutions.c1"], "affected_requirements": ["measures"], "issues": []}}, lean=True)
    assert after["solutions"]["claims"][1:] == before["solutions"]["claims"][1:]
    assert after["solutions"]["claims"][0]["text"] == output["solutions"]["claims"][0]["text"]
    assert all(after[s] == before[s] for s in STEPS if s != "solutions")
    assert before["solutions"]["claims"][0]["text"] != after["solutions"]["claims"][0]["text"]


def test_lean_unknown_citation_still_fails_existing_validation():
    from app.six_step_answer import generate_research_steps
    class Unchecked(Runner):
        def call(self, *args, **kwargs):
            result = lean_draft()
            result["solutions"]["claims"][0]["evidence_ids"] = ["not_a_source"]
            return result
    with pytest.raises(InvalidModelOutput):
        generate_research_steps(Unchecked(None), "成人NIV皮肤观察", facts(), sources(), lean=True)


def test_lean_context_split_keeps_all_steps_and_prior_dependencies():
    from app.six_step_answer import generate_research_steps
    class Split:
        calls = []
        def call(self, prompt, payload, schema, name, **kwargs):
            self.calls.append(payload["steps"])
            if len(payload["steps"]) > 3:
                raise InvalidModelOutput("Structured context budget exceeded")
            raw = {s: lean_draft()[s] for s in payload["steps"]}
            validate(raw, schema)
            return raw
    runner = Split()
    value = generate_research_steps(runner, "成人NIV皮肤观察", facts(), sources(), lean=True)
    assert len(runner.calls) == 3 and set(value) == set(STEPS)
    assert value["action"]["claims"][0]["depends_on"] == ["solutions.c1"]
    validate_steps(value, facts(), sources(), research_mode=True)


def test_lean_revision_feedback_is_sent_once_without_losing_step_associations():
    from app.six_step_answer import generate_research_steps
    issue = {"target_id": "solutions.c1", "issue_type": "missing_condition", "severity": "major", "reason": "Controlled shared feedback"}
    revisions = {s: {"affected_claim_ids": [f"{s}.c1"], "affected_requirements": [], "issues": [deepcopy(issue)]}
                 for s in ("solutions", "action", "evaluate")}
    class SharedFeedback:
        def call(self, prompt, payload, schema, name, **kwargs):
            assert len(payload["issues"]) == 1
            for step in payload["steps"]:
                associated = [payload["issues"][key] for key in payload["revisions"][step]["issue_ids"]]
                assert associated == revisions[step]["issues"]
                assert "issues" not in payload["revisions"][step]
            return {s: complete_wire(payload["previous_steps"][s]) for s in payload["steps"]}
    before = complete_steps()
    after = generate_research_steps(SharedFeedback(), "成人NIV皮肤观察", facts(), sources(), before,
                                    revisions=revisions, lean=True)
    assert {s: complete_wire(v) for s, v in after.items()} == {s: complete_wire(v) for s, v in before.items()}


@pytest.mark.parametrize("repair_dependency", [True, False, None])
def test_split_revision_allows_later_steps_to_repair_deleted_dependencies(repair_dependency):
    from app.six_step_answer import generate_research_steps
    before = complete_steps()
    revisions = {s: {"affected_claim_ids": [f"{s}.c1"],
                     "affected_requirements": list(STEPS[s][1]), "issues": []}
                 for s in ("solutions", "action")}
    if repair_dependency is None:
        revisions.pop("action")
    class Split:
        def call(self, prompt, payload, schema, name, **kwargs):
            if len(payload["steps"]) > 1:
                raise InvalidModelOutput("Structured context budget exceeded")
            step = payload["steps"][0]
            c = {k: v for k, v in before[step]["claims"][0].items() if k not in {"conditions", "rationale"}}
            if step == "solutions":
                c["claim_id"] = "solutions.c2"
            elif repair_dependency:
                c["depends_on"] = ["solutions.c2"]
            coverage = deepcopy(before[step]["coverage"])
            if step == "solutions":
                for row in coverage.values():
                    row["claim_ids"] = ["solutions.c2" if cid == "solutions.c1" else cid for cid in row["claim_ids"]]
            wire = {key: row["claim_ids"] if row["status"] == "supported" else row for key, row in coverage.items()}
            extra = []
            if step == "solutions":
                for i in (3, 4):
                    other = deepcopy(c); other["claim_id"] = f"solutions.c{i}"; extra.append(other)
            return {step: {"claims": [c, *extra], "coverage": wire}}
    if not repair_dependency:
        with pytest.raises(InvalidModelOutput):
            generate_research_steps(Split(), "成人使用NIV", facts(), sources(), before, revisions=revisions, lean=True)
    else:
        after = generate_research_steps(Split(), "成人使用NIV", facts(), sources(), before, revisions=revisions, lean=True)
        assert after["action"]["claims"][0]["depends_on"] == ["solutions.c2"]
        validate_steps(after, facts(), sources(), research_mode=True)


def test_revision_budget_covers_all_returned_claims_within_one_step():
    from app.six_step_answer import generate_research_steps
    budgets = []
    class Unchanged:
        def call(self, prompt, payload, schema, name, **kwargs):
            budgets.append(kwargs["max_tokens"])
            output = {"action": complete_wire(payload["previous_steps"]["action"])}
            validate(output, schema)
            return output
    for count in (1, 4):
        before = complete_steps()
        for i in range(2, count + 1):
            c = deepcopy(before["action"]["claims"][0]); c["claim_id"] = f"action.c{i}"
            before["action"]["claims"].append(c)
        revision = {"action": {"affected_claim_ids": ["action.c1"], "affected_requirements": [], "issues": []}}
        after = generate_research_steps(Unchanged(), "成人使用NIV", facts(), sources(), before,
                                        revisions=revision, lean=True)
        assert complete_wire(after["action"]) == complete_wire(before["action"])
    assert budgets[1] > budgets[0]


def test_complete_revision_preserves_model_priority_order():
    from app.six_step_answer import _expand_lean
    before = complete_steps()["prioritize"]
    second = deepcopy(before["claims"][0]); second["claim_id"] = "prioritize.c2"
    before["claims"].append(second)
    wire = complete_wire(before)
    wire["claims"][0]["text"] = "Controlled replacement, same priority position"
    after = _expand_lean(wire)
    assert [c["claim_id"] for c in after["claims"]] == ["prioritize.c1", "prioritize.c2"]
    assert complete_wire(after)["claims"][1] == wire["claims"][1]


def test_lean_complete_revision_is_reaudited_and_raw_outputs_are_archived(tmp_path):
    from app.reflection_workflow import reflective_answer
    from app.evidence_snapshot import load_artifacts
    from test_rag import config, pipeline
    from test_reflection_v2 import ScriptedLLM
    from test_research_latency import focused_wire
    class LLM(ScriptedLLM):
        revised = False
        def structured(self, messages, schema, name, **kwargs):
            payload = json.loads(messages[1]["content"])
            if name == "research_task_scope":
                return {"literature_requested": False, "requirement_quote": "", "assigned_steps": []}
            if name == "case_facts":
                return {"classifications": {key: {"category": "device" if i == 0 else "not_fact", "state": "present"}
                        for i, key in enumerate(payload["candidate_clauses"])}, "missing_information": [], "case_conflicts": []}
            if name == "six_steps_draft":
                raw = lean_draft()
                for step in raw.values():
                    for c in step["claims"]:
                        c.update(kind="model_inference", fact_ids=[], evidence_ids=[])
                validate(raw, schema)
                return raw
            if name == "six_steps_revision":
                self.revised = True
                raw = {}
                for step in payload["steps"]:
                    patch = complete_wire(payload["previous_steps"][step])
                    if step == "solutions":
                        patch["claims"][0]["text"] = "已补回条件，维持治疗连续性。"
                    raw[step] = patch
                validate(raw, schema)
                return raw
            if name == "audit_focused":
                issues = [] if self.revised else [{"check_id": "safety:solutions.c1", "assessment": "条件缺失",
                    "finding": "missing_condition", "severity": "major", "route": "correction"}]
                result = focused_wire(payload["targets"], [i for i in issues if i["check_id"] in payload["targets"]])
                return result
            raise AssertionError(name)
    cfg = config(); cfg.values["reflection"] = {"log_dir": str(tmp_path), "research_compact": True, "research_lean_output": True}
    value = reflective_answer(pipeline(cfg=cfg, llm=LLM()), "成人使用NIV面罩。请分析皮肤评估。", research_mode=True, web_mode="off")
    assert value["errors"] == []
    assert value["timing"]["revision_rounds"] == 1
    assert value["six_steps"]["solutions"]["claims"][0]["text"] == "已补回条件，维持治疗连续性。"
    assert len(value["six_steps"]["solutions"]["claims"]) == 3
    artifacts = load_artifacts(tmp_path, value["run_id"])
    requests = [d for n, d in artifacts if "model_request" in n and d["name"] == "audit_focused"]
    assert sum("safety:solutions.c1" in json.loads(d["messages"][1]["content"])["targets"]
               for d in requests) == 2
    raw = [d["value"] for n, d in artifacts if "model_response" in n and d["name"] == "six_steps_revision"][0]
    assert len(raw["solutions"]["claims"]) == 3
    assert any("candidate_1" in n for n, _ in artifacts)
