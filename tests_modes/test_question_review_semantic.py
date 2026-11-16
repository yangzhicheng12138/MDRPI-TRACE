"""Evidence summaries reach revision without a verbatim/length gate."""
import json

import httpx
import pytest

from modes.question_generation_audit import validate_audited_review
from modes.question_review_text import parse_review_text, TextReviewFormatError
from tests_modes.test_question_review_text_backend import (
    factory, call_review, full_text, envelope, evidence, question, LOCAL_QUOTE, WEB_QUOTE,
)
from tests_modes.test_question_generation_workflow import FakeBackend, service_factory, run, review


def summary_report():
    return full_text().replace("原文：" + LOCAL_QUOTE,
        "依据概述：使用接口前，应先记录其接触组织的初始状况，作为后续比较依据。"
    ).replace("原文：" + WEB_QUOTE,
        "依据概述：调整接口的同时仍需保证呼吸治疗，调整后还要检查治疗是否有效。")


def parse(content, mode="semantic"):
    return parse_review_text(content, evidence=evidence(), question=question(),
                             audited=True, evidence_mode=mode)


def test_paraphrases_are_stored_as_summaries_with_original_decisions():
    result = parse(summary_report())
    assert result["verdict"] == "accept"
    assert result["issues"] == result["evidence_gaps"] == []
    source = result["quality_audit"]["evidence_support"][0]["sources"][0]
    assert "quote" not in source and source["summary"].startswith("使用接口前")
    assert validate_audited_review(result, evidence(), question(), evidence_mode="semantic") == result
    with pytest.raises(TextReviewFormatError):
        parse(summary_report(), "verbatim")


def test_long_evidence_explanations_are_preserved_without_truncation():
    explanation = "Before fitting the interface, document the baseline condition of the contact tissue. " * 12
    report = full_text().replace("原文：" + LOCAL_QUOTE, "依据概述：" + explanation)
    result = parse(report)
    actual = result["quality_audit"]["evidence_support"][0]["sources"][0]["summary"]
    assert actual == explanation.strip() and len(actual) > 360
    with pytest.raises(TextReviewFormatError):
        parse(report, "verbatim")


@pytest.mark.parametrize("mutation", [
    lambda s: s.replace("核查来源：L1", "核查来源：NONEXISTENT"),
    lambda s: s.replace("核查来源：L1", "核查来源：W1"),
    lambda s: s.replace("证据核查：R2", "证据核查：R1"),
    lambda s: s.replace("病例事实：一致", "病例事实：不确定"),
    lambda s: s.replace("来源支持：充分支持", "来源支持：不支持"),
    lambda s: s.replace("原文：" + LOCAL_QUOTE, "依据概述：无"),
])
def test_semantic_mode_keeps_source_binding_and_explicit_decisions(mutation):
    with pytest.raises(TextReviewFormatError):
        parse(mutation(full_text()))


@pytest.mark.parametrize("wire_format", ["text", "json"])
def test_remote_review_uses_semantic_instructions_and_returns_without_retry(factory, wire_format):
    requests = []
    content = summary_report() if wire_format == "text" else json.dumps(parse(summary_report()), ensure_ascii=False)

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=envelope(content))

    backend = factory(handler)
    backend.options.update(review_evidence_mode="semantic", reviewer_output=wire_format)
    result = call_review(backend)
    assert len(requests) == 1 and result["verdict"] == "accept"
    system = requests[0]["messages"][0]["content"]
    assert "quote 必须逐字摘录" not in system
    assert "从该来源逐字连续摘录" not in system
    assert "对照完整证据 text 判断含义是否忠实" in system
    fixed = json.loads(requests[0]["messages"][1]["content"])
    assert [row["text"] for row in fixed["evidence"]] == [row["text"] for row in evidence()["sources"]]
    assert "summary" in result["quality_audit"]["evidence_support"][0]["sources"][0]


def test_saved_legacy_policy_still_selects_verbatim_review(factory):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=envelope(full_text()))

    backend = factory(handler)
    backend.options["review_evidence_mode"] = "semantic"
    result = call_review(backend, teaching_plan={"audit_required": True, "review_evidence_mode": "verbatim"})
    assert "quote 必须逐字摘录" in requests[0]["messages"][0]["content"]
    assert "quote" in result["quality_audit"]["evidence_support"][0]["sources"][0]


class SemanticBackend(FakeBackend):
    options = {**FakeBackend.options, "review_evidence_mode": "semantic"}

    def review(self, *args, **kwargs):
        result = super().review(*args, **kwargs)
        for check in result["quality_audit"]["evidence_support"]:
            for source in check["sources"]:
                source.pop("quote")
                source["summary"] = "使用器械前核对其接触组织的状态，并考虑治疗功能。"
        return result


def test_summary_feedback_reaches_revision_and_survives_service_reopen(service_factory):
    backend = SemanticBackend(reviews=[review("revise"), review("accept")])
    service = service_factory(backend)
    result = run(service)
    assert result["status"] == "completed"
    assert result["options"]["review_evidence_mode"] == "semantic"
    assert len(backend.generate_calls) == 2
    feedback = backend.generate_calls[1]["review"]
    assert feedback["verdict"] == "revise" and feedback["issues"]
    assert "summary" in feedback["quality_audit"]["evidence_support"][0]["sources"][0]
    # Reopening uses the saved job policy, not the current backend default.
    reopened = service_factory(FakeBackend(), root=service.root).get(result["job_id"])
    assert reopened["questions"][0]["status"] == "accepted"
    assert not reopened["questions"][0].get("approval_invalidated", False)


def test_unknown_policy_is_rejected_instead_of_silently_relaxing_review():
    with pytest.raises(ValueError, match="Unsupported review evidence mode"):
        parse(full_text(), "typo")


def test_semantic_json_compat_does_not_discard_two_conflicting_evidence_fields():
    result = parse(full_text())
    source = result["quality_audit"]["evidence_support"][0]["sources"][0]
    source["quote"] = "另一个冲突版本"
    with pytest.raises(TextReviewFormatError):
        parse(json.dumps(result, ensure_ascii=False))
