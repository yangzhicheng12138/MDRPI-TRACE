"""Offline transport/controller checks for plain-text question reviews.

All model responses are controlled fixtures. These checks establish transport,
provenance and workflow behavior, not the clinical quality of generated items.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import httpx
import pytest
import yaml
from jsonschema import Draft202012Validator, ValidationError

from app.evidence_snapshot import RunBudget
from app.local_llm import InvalidModelOutput, ModelUnavailable
from modes import question_generation as workflow
from modes import question_generation_backend as backend_module
from modes.question_generation_audit import validate_audited_review
from modes.question_generation_schema import blind_review_schema, validate_review
from modes.question_review_text import TextReviewFormatError


KEY = "controlled-text-review-key-DoNotPersist-123456789"
HIDDEN = "TEACHER_ONLY_TEXT_REVIEW_SENTINEL"
CASE = "成人准备佩戴无创通气面罩，面颊皮肤完整，器械接触点基线尚未记录。"
LOCAL_QUOTE = "佩戴前记录器械接触部位的组织基线"
WEB_QUOTE = "维持必要的呼吸支持"
BLIND_TEXT = "可答性：可答\n独立试答：识别皮肤完整这一线索，记录接触部位基线。\n歧义：无"


class Log:
    def __init__(self):
        self.records = []

    def write(self, name, value):
        self.records.append((name, deepcopy(value)))


def blueprint():
    return {"topic": "面罩治疗前的线索识别", "learning_objective": "识别线索并说明一项需要评估的内容。",
        "population": "adult", "device_type": "niv_mask", "difficulty": "基础",
        "question_type": "short_answer", "count": 1, "requirements": "保留固定病例事实。",
        "material_ids": ["controlled-text-material"]}


def materials():
    return [{"id": "controlled-text-material", "title": "纯文本审核控制病例", "case_facts": CASE,
        "population": "adult", "populations": ["adult"], "device_types": ["niv_mask"],
        "usage": "development_only", "review_status": "approved_ai", "excerpt": HIDDEN,
        "answer": HIDDEN, "rubric": HIDDEN, "hints": [HIDDEN],
        "references": [{"id": "controlled-document", "title": "控制依据"}],
        "teaching_anchors": [
            {"id": "recognize_R1", "step": "recognize", "text": "识别病例已有的器械和皮肤线索。",
             "acceptable": "接受等价表述。", "reference_ids": ["controlled-document"]},
            {"id": "recognize_R2", "step": "recognize", "text": "指出与已有线索关联的评估目标。",
             "acceptable": "未知保持未知。", "reference_ids": ["controlled-document"]}]}]


def evidence():
    rows = []
    for identity, origin, text in [
        ("L1", "local", LOCAL_QUOTE + "，评估接口是否适配；未知患者结果应保持未知。"),
        ("W1", "external", WEB_QUOTE + "，并在调整接口后复评治疗效果。"),
    ]:
        rows.append({"evidence_id": identity, "id": identity, "origin": origin,
            "title": "控制测试来源 " + identity, "text": text * 4,
            "source_url": "https://example.org/text-review/" + identity, "location": "原文第2节",
            "doc_id": "controlled-document-" + identity, "population": ["adult"],
            "device_type": ["niv_mask"], "read_success": True,
            "material_level": "full_text_excerpt" if origin == "local" else "web_page_excerpt"})
    return {"sources": rows, "events": [], "local_success": True, "web_success": True,
        "rag_attempted": True, "web_attempted": True}


def question():
    return {"title": "佩戴前的评估", "stem": CASE,
        "task": "指出病例中的一条线索，并说明一项需要继续评估的内容。",
        "options": [], "correct_options": [], "reference_answer": HIDDEN,
        "scoring_points": [
            {"id": "R1", "criterion": "识别已有线索", "acceptable_answers": ["面颊皮肤完整"], "evidence_ids": ["L1"]},
            {"id": "R2", "criterion": "指出相关评估目标", "acceptable_answers": ["接触部位基线"], "evidence_ids": ["W1"]}],
        "hints": ["留意已有观察。", "比较已知与未知。", "结合线索说明评估目标。"],
        "common_misconceptions": [HIDDEN], "evidence_ids": ["L1", "W1"]}


def blind():
    return {"answerable": True, "attempted_answer": "识别皮肤完整这一线索，记录接触部位基线。", "ambiguity_notes": []}


def full_text(*, audited=True, verdict="通过"):
    lines = ["结论：" + verdict]
    if audited:
        lines += ["病例事实：一致", "病例事实说明：保持已给病例事实，未知仍为未知。",
            "目标评分：一致", "目标评分说明：只核对题面要求的线索与评估内容。",
            "重复判断：不适用", "重复判断说明：没有其他题目的重复候选。"]
    if verdict == "需修订":
        lines += ["问题：1", "类型：考核不符", "位置：reference_answer", "原句：" + HIDDEN,
            "原因：参考答案需要清楚说明线索与评估目标的关联。", "建议：收窄到题目所问的评估内容。", "来源：L1"]
    else:
        lines += ["问题：无"]
    if audited:
        for criterion, identity, quote in [("R1", "L1", LOCAL_QUOTE), ("R2", "W1", WEB_QUOTE)]:
            lines += ["证据核查：" + criterion, "评分点支持：充分支持", "评分点理由：原文原则结合已知病例支持本评分点。",
                "核查来源：" + identity, "来源支持：充分支持", "原文：" + quote]
    lines += ["补证：无", "总结：控制测试审核结论。"]
    return "\n".join(lines)


def envelope(content, *, finish_reason="stop"):
    return {"id": "controlled-text-response", "model": "deepseek-flash",
        "choices": [{"finish_reason": finish_reason, "message": {"content": content}}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 22,
            "prompt_cache_hit_tokens": 100, "prompt_cache_miss_tokens": 20}}


@pytest.fixture
def factory(tmp_path, monkeypatch):
    monkeypatch.setenv("QUESTION_GENERATION_TEXT_CONTROLLED_KEY", KEY)
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"llm": {"model": "unchanged-local-model",
        "base_url": "http://127.0.0.1:8001/v1"}}), encoding="utf-8")
    trial = tmp_path / "trial.yaml"
    trial.write_text(yaml.safe_dump({"base_config": str(base), "question_generation": {
        "api_key_env": "QUESTION_GENERATION_TEXT_CONTROLLED_KEY",
        "api_key_file": ".secrets/nonexistent-controlled-text-test-key", "reviewer_output": "text"}}), encoding="utf-8")

    def create(handler):
        def forbidden_local(config):
            raise AssertionError("Reviewing must never call the local generator")
        return backend_module.QuestionGenerationBackend(trial,
            transport=httpx.MockTransport(handler), local_factory=forbidden_local)
    return create


def call_review(backend, log=None, budget=None, *, audited=True, teaching_plan=None):
    return backend.review(blueprint(), materials(), evidence(),
        teaching_plan if teaching_plan is not None else {"audit_required": audited},
        question(), blind(), log or Log(), budget or RunBudget(30))


def assert_text_wire(wire):
    assert "response_format" not in wire
    assert not {"schema", "json_schema", "response_schema"}.intersection(wire)
    assert wire["thinking"] == {"type": "disabled"}
    assert wire["model"] == "deepseek-flash" and wire["temperature"] == 0
    assert wire["stream"] is False
    sent_content = "\n".join(message["content"] for message in wire["messages"])
    assert '"properties"' not in sent_content and '"additionalProperties"' not in sent_content
    system = wire["messages"][0]["content"]
    assert "Return a compact complete JSON object" not in system
    assert "只返回要求的 JSON" not in system and "只返回指定 JSON" not in system


def test_blind_text_wire_is_answer_free_and_returns_original_internal_contract(factory):
    requests, log, budget = [], Log(), RunBudget(30)

    def handler(request):
        assert request.headers["authorization"] == "Bearer " + KEY
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=envelope(BLIND_TEXT))
    result = factory(handler).blind_review(blueprint(), materials(), question(), log, budget)
    assert result == blind()
    Draft202012Validator(blind_review_schema()).validate(result)
    assert len(requests) == budget.counts["model_calls"] == 1
    assert_text_wire(requests[0])
    assert HIDDEN not in json.dumps(requests[0], ensure_ascii=False)
    variable = json.loads(requests[0]["messages"][-1]["content"])
    assert set(variable["question"]) == {"title", "stem", "task", "options"}
    assert all(set(row) == {"id", "title", "case_facts"} for row in variable["student_visible_materials"])
    response = next(value for name, value in log.records if name == "question_model_response")
    assert response["value"] == result and response["usage"]["prompt_cache_hit_tokens"] == 100
    assert response["metadata"]["finish_reason"] == "stop"
    raw = next(value for name, value in log.records if name == "question_model_text_response")
    assert raw["text"] == BLIND_TEXT and raw["usage"] == response["usage"]
    assert KEY not in json.dumps(log.records, ensure_ascii=False)


@pytest.mark.parametrize("audited", [False, True], ids=["legacy-review-contract", "audited-review-contract"])
def test_full_text_wire_omits_schema_but_internal_result_passes_original_validation(factory, audited):
    requests, log = [], Log()

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=envelope(full_text(audited=audited)))
    result = call_review(factory(handler), log, audited=audited)
    assert result["verdict"] == "accept" and result["issues"] == result["evidence_gaps"] == []
    assert_text_wire(requests[0])
    assert len(requests) == 1
    if audited:
        assert validate_audited_review(result, evidence(), question()) == result
        assert [row["criterion_id"] for row in result["quality_audit"]["evidence_support"]] == ["R1", "R2"]
    else:
        assert validate_review(result, ["L1", "W1"]) == result
        assert "quality_audit" not in result


@pytest.mark.parametrize("broken", [
    full_text().replace("结论：通过\n", ""),
    full_text().replace("核查来源：L1", "核查来源：UNKNOWN_SOURCE"),
    full_text().replace("原文：" + LOCAL_QUOTE, "原文：这句话没有出现在原文中"),
    full_text().replace("核查来源：L1", "核查来源：W1").replace("原文：" + LOCAL_QUOTE, "原文：" + WEB_QUOTE),
    full_text().replace("病例事实：一致", "病例事实：不确定"),
], ids=["missing-verdict", "unknown-source", "fabricated-quote", "wrong-criterion-source", "unresolved-accept"])
def test_invalid_text_never_becomes_an_accept_and_has_only_one_repair(factory, broken):
    calls, log, budget = [], Log(), RunBudget(30)

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=envelope(broken))
    with pytest.raises((InvalidModelOutput, ValidationError, TextReviewFormatError)):
        call_review(factory(handler), log, budget)
    assert len(calls) == budget.counts["model_calls"] == 2
    assert len([value for name, value in log.records if name == "question_model_format_error"]) == 2
    assert not [value for name, value in log.records if name == "question_model_response"]
    assert all("format_repair" not in json.loads(wire["messages"][-1]["content"]) for wire in calls[:1])
    assert "format_repair" in json.loads(calls[1]["messages"][-1]["content"])
    for wire in calls:
        assert_text_wire(wire)


def test_one_text_repair_identifies_missing_field_and_keeps_fixed_prefix(factory):
    calls, log, budget = [], Log(), RunBudget(30)
    malformed = full_text().replace("结论：通过\n", "")

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=envelope(malformed if len(calls) == 1 else full_text()))
    assert call_review(factory(handler), log, budget)["verdict"] == "accept"
    assert len(calls) == budget.counts["model_calls"] == 2
    assert calls[0]["messages"][:2] == calls[1]["messages"][:2]
    first = json.loads(calls[0]["messages"][-1]["content"])
    repaired = json.loads(calls[1]["messages"][-1]["content"])
    repair = repaired.pop("format_repair")
    assert repaired == first and "结论" in repair
    assert "不要重写题目" in repair and LOCAL_QUOTE not in repair and WEB_QUOTE not in repair
    errors = [value for name, value in log.records if name == "question_model_format_error"]
    assert len(errors) == 1 and errors[0]["content"] == malformed
    assert errors[0]["usage"]["prompt_cache_hit_tokens"] == 100
    for wire in calls:
        assert_text_wire(wire)


@pytest.mark.parametrize("status", [401, 429, 503])
def test_text_http_failure_is_logged_once_without_response_body_or_retry(factory, status):
    calls, log, budget = [], Log(), RunBudget(30)

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="UPSTREAM_PRIVATE_BODY Bearer " + KEY)
    with pytest.raises(ModelUnavailable, match=str(status)):
        call_review(factory(handler), log, budget)
    assert len(calls) == budget.counts["model_calls"] == 1
    errors = [value for name, value in log.records if name == "question_model_error"]
    assert len(errors) == 1 and str(status) in errors[0]["message"]
    serialized = json.dumps(log.records, ensure_ascii=False)
    assert KEY not in serialized and "UPSTREAM_PRIVATE_BODY" not in serialized
    assert not [name for name, value in log.records if name == "question_model_format_error"]


def test_text_network_failure_does_not_retry_or_persist_exception_credentials(factory):
    calls, log = [], Log()

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("UPSTREAM_PRIVATE_EXCEPTION " + KEY, request=request)
    with pytest.raises(ModelUnavailable):
        call_review(factory(handler), log)
    assert len(calls) == 1
    assert len([name for name, value in log.records if name == "question_model_error"]) == 1
    serialized = json.dumps(log.records, ensure_ascii=False)
    assert KEY not in serialized and "UPSTREAM_PRIVATE_EXCEPTION" not in serialized


def test_text_input_credential_is_rejected_before_any_request_or_request_log(factory):
    calls, log = [], Log()
    item = question()
    item["task"] = "Do not persist " + KEY
    backend = factory(lambda request: calls.append(request) or httpx.Response(200, json=envelope(BLIND_TEXT)))
    with pytest.raises(ValueError, match="Credential"):
        backend.blind_review(blueprint(), materials(), item, log, RunBudget(30))
    assert not calls and not log.records


@pytest.mark.parametrize("content", [
    "malformed output echoing " + KEY,
    BLIND_TEXT.replace("独立试答：", "独立试答：" + KEY),
], ids=["malformed-content", "parsed-content"])
def test_text_output_credentials_are_redacted_on_both_format_attempts(factory, content):
    calls, log = [], Log()

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=envelope(content))
    with pytest.raises((InvalidModelOutput, ValidationError, TextReviewFormatError)):
        factory(handler).blind_review(blueprint(), materials(), question(), log, RunBudget(30))
    assert len(calls) == 2
    assert KEY not in json.dumps(log.records, ensure_ascii=False)


def test_explicit_json_option_preserves_old_reviewer_transport(factory):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=envelope(json.dumps(blind(), ensure_ascii=False)))
    backend = factory(handler)
    backend.options["reviewer_output"] = "json"
    assert backend.blind_review(blueprint(), materials(), question(), Log(), RunBudget(30)) == blind()
    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "additionalProperties" in calls[0]["messages"][0]["content"]




def run_service(factory, monkeypatch, tmp_path, verdicts):
    calls, generations = [], []
    decisions = iter(verdicts)

    def handler(request):
        wire = json.loads(request.content)
        calls.append(wire)
        variable = json.loads(wire["messages"][-1]["content"])
        if "independent_attempt" in variable:
            return httpx.Response(200, json=envelope(full_text(verdict=next(decisions))))
        return httpx.Response(200, json=envelope(BLIND_TEXT))
    backend = factory(handler)
    monkeypatch.setattr(workflow, "load_materials", lambda *args: materials())
    monkeypatch.setattr(backend, "retrieve", lambda *args, **kwargs: evidence())

    def generate(specification, selected, packet, plan, log, budget, *, index, previous=None, review=None):
        generations.append({"previous": deepcopy(previous), "review": deepcopy(review)})
        item = question()
        if previous:
            item["reference_answer"] = "识别皮肤完整线索并记录接触部位基线。"
        return item
    monkeypatch.setattr(backend, "generate", generate)
    service = workflow.QuestionGenerationService(tmp_path / "text-studio", backend)
    queued = service.create(blueprint())
    return service, service.run(queued["job_id"]), calls, generations


def test_clinical_revision_from_text_enters_original_loop_and_binds_final_acceptance(factory, monkeypatch, tmp_path):
    service, result, calls, generations = run_service(factory, monkeypatch, tmp_path, ["需修订", "通过"])
    assert result["status"] == "completed"
    row = result["questions"][0]
    assert row["status"] == "accepted" and row["version"] == row["accepted_version"] == 2
    assert len(generations) == 2 and len(calls) == 4
    assert generations[1]["previous"] == question()
    assert generations[1]["review"]["verdict"] == "revise"
    assert generations[1]["review"]["issues"][0]["kind"] == "assessment_mismatch"
    assert all("format_repair" not in json.loads(wire["messages"][-1]["content"]) for wire in calls)
    for wire in calls:
        assert_text_wire(wire)
    approval = row["approval"]
    assert row["accepted_question_sha256"] == approval["question_sha256"] == workflow._digest(row["question"])
    assert row["accepted_evidence_snapshot_id"] == approval["evidence_snapshot_id"] == row["evidence_snapshot_id"]
    assert approval["review_sha256"] == workflow._digest(row["review"])
    assert approval["blind_review_sha256"] == workflow._digest(row["blind_review"])
    assert approval["batch_plan_sha256"] == result["batch_plan_sha256"]
    assert approval["item_plan_sha256"] == workflow._digest(row["plan"])
    assert service.get(result["job_id"])["questions"][0]["status"] == "accepted"

    path = service.jobs / (result["job_id"] + ".json")
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["questions"][0]["question"]["reference_answer"] = "篡改后的答案不得继承通过。"
    path.write_text(json.dumps(stored, ensure_ascii=False), encoding="utf-8")
    assert service.get(result["job_id"])["questions"][0]["status"] == "needs_review"


def test_text_clinical_revisions_keep_original_two_revision_limit(factory, monkeypatch, tmp_path):
    service, result, calls, generations = run_service(factory, monkeypatch, tmp_path, ["需修订"] * 3)
    row = result["questions"][0]
    assert result["status"] != "completed" and row["status"] == "revision_needed"
    assert row["version"] == 3 and len(generations) == 3 and len(calls) == 6
    assert not row.get("approval")
    assert all(generation["review"]["verdict"] == "revise" for generation in generations[1:])
    assert all("format_repair" not in json.loads(wire["messages"][-1]["content"]) for wire in calls)
