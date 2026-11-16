"""Contract failures that would otherwise admit ungrounded or ungradable items."""
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator, ValidationError

from modes.question_generation_schema import (
    anchor_plan_schema, blind_review_schema, blueprint_schema, question_schema,
    review_schema, validate_blueprint, validate_question, validate_review,
)


def blueprint():
    return {
        "topic": "面罩相关皮肤保护", "learning_objective": "识别两条风险线索并解释关联",
        "population": "adult", "device_type": "口鼻面罩", "difficulty": "进阶",
        "question_type": "short_answer", "material_ids": ["DEV-MASK-01"],
    }


def question():
    return {
        "title": "面罩风险线索", "stem": "固定病例原文。", "task": "识别两条线索并解释其关系。",
        "options": [], "correct_options": [], "reference_answer": "结合固定与局部变化说明关联。",
        "scoring_points": [
            {"id": "R1", "criterion": "识别固定相关线索", "acceptable_answers": ["固定带受力不均"], "evidence_ids": ["E1"]},
            {"id": "R2", "criterion": "解释局部组织变化", "acceptable_answers": ["局部红斑提示需检查器械接触部位"], "evidence_ids": ["E2"]},
        ],
        "hints": ["关注器械接触部位。", "比较固定与局部变化。", "说明两者之间可能的关系。"],
        "common_misconceptions": ["不能把未提供的检查结果当成正常。"],
        "evidence_ids": ["E1", "E2"],
    }


def issue(kind="evidence_mismatch"):
    return {"kind": kind, "location": "reference_answer", "text": "被核查的原句。",
            "explanation": "该来源适用人群不同。", "evidence_ids": ["E1"],
            "requested_change": "更换符合目标人群的依据。"}


def test_blueprint_defaults_do_not_mutate_request_and_trim_material_ids():
    request = blueprint()
    request["topic"] = "  面罩相关皮肤保护  "
    request["material_ids"] = [" DEV-MASK-01 "]
    original = deepcopy(request)
    result = validate_blueprint(request)
    assert request == original
    assert result["topic"] == "面罩相关皮肤保护"
    assert result["material_ids"] == ["DEV-MASK-01"]
    assert result["count"] == 1 and result["requirements"] == ""


@pytest.mark.parametrize("patch", [
    {"material_ids": []}, {"material_ids": ["DEV1", "DEV1"]},
    {"population": "any"}, {"difficulty": "随便"}, {"count": 0},
    {"count": 6}, {"count": True}, {"learning_objective": "   "},
    {"unreviewed_material_ok": True},
])
def test_blueprint_rejects_bypass_and_unbounded_requests(patch):
    with pytest.raises(ValidationError):
        validate_blueprint({**blueprint(), **patch})


def test_blueprint_requires_explicit_selected_materials():
    value = blueprint()
    del value["material_ids"]
    with pytest.raises(ValidationError):
        validate_blueprint(value)


def test_question_requires_supported_scoring_points_and_three_hints():
    assert validate_question(question(), ["E1", "E2"], "short_answer") == question()
    for mutate in (
        lambda q: q["scoring_points"][0].update(evidence_ids=["NOT_READ"]),
        lambda q: q["scoring_points"][0].update(evidence_ids=[]),
        lambda q: q.update(hints=q["hints"][:2]),
        lambda q: q.update(scoring_points=q["scoring_points"][:1]),
        lambda q: q.update(evidence_ids=["E1"]),
        lambda q: q["scoring_points"][1].update(id="R1"),
        lambda q: q["scoring_points"][1].update(criterion="   "),
        lambda q: q.update(reference_answer="\n  "),
    ):
        value = question()
        mutate(value)
        with pytest.raises(ValidationError):
            validate_question(value, ["E1", "E2"], "short_answer")


def test_single_choice_cross_field_key_and_duplicate_options_are_checked():
    value = question()
    value.update(options=[{"label": label, "text": f"选项{label}"} for label in "ABC"],
                 correct_options=["B"])
    value["scoring_points"] = value["scoring_points"][:1]
    value["scoring_points"][0]["criterion"] = "选择B，其判断符合题干任务和证据。"
    assert validate_question(value, ["E1", "E2"], "single_choice") == value
    for mutate in (
        lambda q: q.update(correct_options=["D"]),
        lambda q: q.update(correct_options=["A", "B"]),
        lambda q: q["options"][1].update(label="A"),
        lambda q: q["options"][1].update(text="  选项A  "),
        lambda q: q.update(options=q["options"][:2]),
    ):
        broken = deepcopy(value)
        mutate(broken)
        with pytest.raises(ValidationError):
            validate_question(broken, ["E1", "E2"], "single_choice")
    with pytest.raises(ValidationError):
        validate_question(value, ["E1", "E2"], "short_answer")


def test_single_choice_has_one_key_based_criterion_while_short_answer_needs_two_or_three():
    single = question()
    single.update(options=[{"label": label, "text": f"选项{label}"} for label in "ABC"],
                  correct_options=["B"])
    with pytest.raises(ValidationError):
        validate_question(single, ["E1", "E2"], "single_choice")
    single["scoring_points"] = single["scoring_points"][:1]
    single["scoring_points"][0]["criterion"] = "选择B：该选项对应题目要求，理由由解析说明。"
    assert validate_question(single, ["E1", "E2"], "single_choice")["correct_options"] == ["B"]
    short = question()
    short["scoring_points"] = short["scoring_points"][:1]
    with pytest.raises(ValidationError):
        validate_question(short, ["E1", "E2"], "short_answer")


def test_review_cannot_pass_with_unresolved_problem_or_gap():
    passed = {"verdict": "accept", "issues": [], "evidence_gaps": [], "summary": "未发现需要修订的问题。"}
    assert validate_review(passed, ["E1"]) == passed
    for patch in (
        {"issues": [issue()]},
        {"evidence_gaps": [{"query": "pediatric mask pressure injury", "reason": "缺少目标人群资料"}]},
        {"verdict": "needs_evidence"},
        {"verdict": "revise"},
        {"verdict": "reject"},
    ):
        with pytest.raises(ValidationError):
            validate_review({**passed, **patch}, ["E1"])


def test_material_gap_is_a_distinct_issue_and_missing_source_can_remain_unresolved():
    row = issue("material_gap")
    row["evidence_ids"] = []
    value = {"verdict": "needs_evidence", "issues": [row],
             "evidence_gaps": [{"query": "medical device pressure injury pediatric", "reason": "尚未读到儿童依据"}],
             "summary": "补证后重新核查，不直接判医学错误。"}
    assert validate_review(value, ["E1"])["issues"][0]["kind"] == "material_gap"
    value["issues"][0]["evidence_ids"] = ["INVENTED"]
    with pytest.raises(ValidationError):
        validate_review(value, ["E1"])


def test_blind_contract_does_not_accept_reference_key_or_rubric():
    value = {"answerable": True, "attempted_answer": "独立作答。", "ambiguity_notes": []}
    Draft202012Validator(blind_review_schema()).validate(value)
    with pytest.raises(ValidationError):
        Draft202012Validator(blind_review_schema()).validate({**value, "correct_options": ["A"]})


def test_every_contract_is_valid_draft202012_and_rejects_missing_evidence_pool():
    for schema in (blueprint_schema(), anchor_plan_schema(["E1"]),
                   question_schema(["E1"], "short_answer"), question_schema(["E1"], "single_choice"),
                   blind_review_schema(), review_schema(["E1"])):
        Draft202012Validator.check_schema(schema)
    with pytest.raises(ValueError):
        question_schema([], "short_answer")
    with pytest.raises(ValueError):
        question_schema(["E1"], "ordering")
