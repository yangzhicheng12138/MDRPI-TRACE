"""Prose decisions retain full review coverage and exact source provenance."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from modes.question_generation_audit import validate_audited_review
from modes.question_review_text import (
    TextReviewFormatError, parse_review_text, review_text_instructions,
)


@pytest.fixture
def evidence():
    return {"sources": [
        {"evidence_id": "L1", "text": "佩戴前记录器械接触部位的组织基线，评估接口是否适配。"},
        {"evidence_id": "W1", "text": "维持必要的呼吸支持，并在调整接口后复评治疗效果。"},
        {"evidence_id": "W2", "text": "呼吸支持期间继续检查接口接触组织。"},
    ]}


@pytest.fixture
def question():
    return {"scoring_points": [{"id": "R1", "evidence_ids": ["L1"]},
                               {"id": "R2", "evidence_ids": ["W1"]}]}


BASE = """结论：通过
问题：无
补证：无
总结：已核查所给范围。"""

QUALITY = """病例事实：一致
病例事实说明：未加入未提供的患者事实。
目标评分：一致
目标评分说明：任务与两个评分点对齐。
重复判断：不适用
重复判断说明：没有相似候选。
"""

CHECK1 = """证据核查：R1
评分点支持：充分支持
评分点理由：原文支持评估接触组织基线。
核查来源：L1
来源支持：充分支持
原文：佩戴前记录器械接触部位的组织基线
"""

CHECK2 = """证据核查：R2
评分点支持：充分支持
评分点理由：保留治疗需求这一约束。
核查来源：W1
来源支持：充分支持
原文：维持必要的呼吸支持
"""

AUDITED = BASE.replace("问题：无", QUALITY + "问题：无").replace("补证：无", CHECK1 + CHECK2 + "补证：无")

ISSUE = """问题：1
类型：表述歧义
位置：task
原句：请选择处理
原因：未明确处理范围。
建议：明确为接触组织评估步骤。
来源：L1、W1
"""

GAP = """补证：1
检索词：NIV 接口 组织基线 评估
补证理由：需要核实该病例接口选择的适用边界。
"""


def parse(content, evidence, question, **kwargs):
    return parse_review_text(content, evidence=evidence, question=question, audited=kwargs.pop("audited", True), **kwargs)


def test_text_instructions_request_prose_and_explicit_model_decisions():
    full = review_text_instructions(audited=True)
    assert "JSON" not in full and "schema" not in full
    for label in ("结论：", "病例事实：", "目标评分：", "重复判断：", "问题：", "证据核查：", "补证："):
        assert label in full
    assert "证据核查：" not in review_text_instructions(audited=False)
    assert "可答性：" in review_text_instructions(audited=True, blind=True)


def test_complete_prose_preserves_existing_audit_contract(evidence, question):
    value = parse(AUDITED, evidence, question)
    assert validate_audited_review(value, evidence, question) == value
    assert value["verdict"] == "accept" and value["issues"] == []
    assert value["quality_audit"]["evidence_support"][0]["status"] == "supported"


def test_base_review_and_multiline_issue_are_structured_without_dropping_a_problem(evidence, question):
    report = BASE.replace("通过", "需修订").replace("问题：无", ISSUE)
    report = report.replace("未明确处理范围。", "未明确处理范围。\n同一问法会得到两类答案。")
    value = parse(report, evidence, question, audited=False)
    assert value["issues"][0]["kind"] == "ambiguity"
    assert value["issues"][0]["evidence_ids"] == ["L1", "W1"]
    assert "\n同一问法" in value["issues"][0]["explanation"]
    assert "quality_audit" not in value


def test_markdown_titles_numbering_fullwidth_colons_and_blank_lines(evidence, question):
    report = AUDITED.replace("结论：通过", "# 审核报告\n1. **结论**：**通过**")
    report = report.replace("病例事实：一致", "## 病例事实：一致")
    report = report.replace("证据核查：R1", "### 证据核查 R1")
    report = report.replace("来源支持：充分支持", "- 来源支持: 充分支持")
    report = report.replace("原文：佩戴前记录器械接触部位的组织基线", "原文：\n佩戴前记录器械接触\n\n部位的组织基线")
    value = parse(report, evidence, question)
    assert value["quality_audit"]["evidence_support"][0]["sources"][0]["quote"] == "佩戴前记录器械接触部位的组织基线"


@pytest.mark.parametrize("decision", ["**结论**：**需修订**", "结论：需修订。", "**结论：** 需修订。", "结论：**需修订。**"])
def test_emphasized_labels_values_and_sentence_punctuation_preserve_exact_decisions(decision, evidence, question):
    report = BASE.replace("结论：通过", decision).replace("问题：无", ISSUE)
    assert parse(report, evidence, question, audited=False)["verdict"] == "revise"


@pytest.mark.parametrize("decision", ["- **可答性：** 可答", "可答性：可答。", "**可答性**：**可答**"])
def test_blind_emphasized_colon_and_sentence_punctuation(decision):
    value = parse_review_text(decision + "\n独立试答：依据材料作答。\n歧义：无", evidence=None,
                              question=None, audited=False, blind=True)
    assert value["answerable"] is True


def test_colons_in_multiline_cited_sentences_and_source_text_are_preserved(evidence, question):
    evidence["sources"][0]["text"] = "观察记录如下：\n血压：120/80 mmHg，皮肤完整。"
    report = AUDITED.replace("原文：佩戴前记录器械接触部位的组织基线", "原文：观察记录如下：\n血压：120/80 mmHg，皮肤完整。")
    assert parse(report, evidence, question)["quality_audit"]["evidence_support"][0]["sources"][0]["quote"] == evidence["sources"][0]["text"]
    report = BASE.replace("通过", "需修订").replace("问题：无", ISSUE.replace("原句：请选择处理", "原句：记录如下：\n血压：120/80 mmHg"))
    assert "\n血压：" in parse(report, evidence, question, audited=False)["issues"][0]["text"]


def test_explicit_unsupported_source_and_gap_never_become_supported(evidence, question):
    report = AUDITED.replace("结论：通过", "结论：需补证").replace("补证：无", GAP)
    report = report.replace(CHECK1, CHECK1.replace("充分支持", "不支持").replace("原文：佩戴前记录器械接触部位的组织基线", "原文：无"))
    value = parse(report, evidence, question)
    check = value["quality_audit"]["evidence_support"][0]
    assert check["status"] == "unsupported" and check["sources"][0]["status"] == "unsupported"
    assert check["sources"][0]["quote"] == "" and value["evidence_gaps"][0]["query"].startswith("NIV")


@pytest.mark.parametrize("remove", ["结论：通过\n", "问题：无\n", "补证：无\n", "病例事实：一致\n",
                                    "目标评分：一致\n", "重复判断：不适用\n", "来源支持：充分支持\n"])
def test_missing_decisions_and_explicit_none_are_errors_not_defaults(remove, evidence, question):
    with pytest.raises(TextReviewFormatError):
        parse(AUDITED.replace(remove, "", 1), evidence, question)


@pytest.mark.parametrize("change", [
    ("结论：通过", "结论：通过或需修订"),
    ("病例事实：一致", "病例事实：似乎合理"),
    ("评分点支持：充分支持", "评分点支持：可能充分支持"),
    ("来源支持：充分支持", "来源支持：支持/不支持"),
])
def test_ambiguous_or_unknown_model_status_fails_with_location(change, evidence, question):
    with pytest.raises(TextReviewFormatError, match="第[0-9]+行"):
        parse(AUDITED.replace(*change, 1), evidence, question)


@pytest.mark.parametrize("suffix", ["\n结论：通过", "\n未知问题：漏掉的临床风险", "\n## 未记录的问题"])
def test_extra_fields_and_hidden_problems_are_not_silently_ignored(suffix, evidence, question):
    with pytest.raises(TextReviewFormatError):
        parse(AUDITED + suffix, evidence, question)


def test_pass_cannot_drop_issues_or_contradict_quality(evidence, question):
    for report in (AUDITED.replace("问题：无", ISSUE),
                   AUDITED.replace("问题：无", "问题：无\n" + ISSUE),
                   AUDITED.replace("病例事实：一致", "病例事实：无依据"),
                   AUDITED.replace("来源支持：充分支持", "来源支持：有限支持", 1)):
        with pytest.raises(TextReviewFormatError):
            parse(report, evidence, question)


@pytest.mark.parametrize("label", ["类型", "位置", "原句", "原因", "建议"])
def test_incomplete_issue_never_disappears(label, evidence, question):
    report = BASE.replace("通过", "需修订").replace("问题：无", ISSUE)
    report = "\n".join(line for line in report.splitlines() if not line.startswith(label + "："))
    with pytest.raises(TextReviewFormatError, match="问题1"):
        parse(report, evidence, question, audited=False)


def test_omitted_issue_source_is_an_empty_list_without_inferring_ids_from_prose(evidence, question):
    issue = ISSUE.replace("来源：L1、W1\n", "").replace("未明确处理范围。", "未明确处理范围。所引L1、W1不能直接支持该判断。")
    report = BASE.replace("通过", "需修订").replace("问题：无", issue)
    value = parse(report, evidence, question, audited=False)
    assert value["issues"][0]["evidence_ids"] == []
    assert value["issues"][0]["explanation"] == "未明确处理范围。所引L1、W1不能直接支持该判断。"
    assert value["issues"][0]["requested_change"] == "明确为接触组织评估步骤。"


def test_explicit_unknown_issue_source_still_fails_instead_of_being_discarded(evidence, question):
    report = BASE.replace("通过", "需修订").replace("问题：无", ISSUE.replace("来源：L1、W1", "来源：L1、unknown"))
    with pytest.raises(TextReviewFormatError):
        parse(report, evidence, question, audited=False)


def test_one_gap_section_can_contain_multiple_complete_search_pairs(evidence, question):
    second = "检索词：无创通气 面罩 鼻梁 敷料 水状胶体 压力性损伤 预防\n补证理由：需核实接触点预防性保护的直接证据。\n"
    third = "检索词：NIV baseline skin assessment\n补证理由：需核实首次佩戴前的基线记录依据。\n"
    report = BASE.replace("通过", "需补证").replace("补证：无", GAP + second + third)
    value = parse(report, evidence, question, audited=False)
    assert value["evidence_gaps"] == [
        {"query": "NIV 接口 组织基线 评估", "reason": "需要核实该病例接口选择的适用边界。"},
        {"query": "无创通气 面罩 鼻梁 敷料 水状胶体 压力性损伤 预防", "reason": "需核实接触点预防性保护的直接证据。"},
        {"query": "NIV baseline skin assessment", "reason": "需核实首次佩戴前的基线记录依据。"},
    ]


@pytest.mark.parametrize("pairs", [
    "检索词：第一个\n检索词：第二个\n补证理由：第二个的理由。",
    "检索词：第一个\n补证理由：第一个的理由。\n检索词：第二个",
    "检索词：第一个\n补证理由：第一个的理由。\n补证理由：第二个的理由。",
])
def test_incomplete_or_ambiguous_repeated_gap_pair_is_not_overwritten(pairs, evidence, question):
    report = BASE.replace("通过", "需补证").replace("补证：无", "补证：1\n" + pairs)
    with pytest.raises(TextReviewFormatError):
        parse(report, evidence, question, audited=False)


def test_inline_gap_pairs_keep_three_gap_limit_and_cannot_hide_in_an_acceptance(evidence, question):
    pairs = "".join(f"检索词：query{index}\n补证理由：reason{index}\n" for index in range(4))
    too_many = BASE.replace("通过", "需补证").replace("补证：无", "补证：1\n" + pairs)
    acceptance = BASE.replace("补证：无", GAP + "检索词：另一个词\n补证理由：另一个理由。")
    for report in (too_many, acceptance):
        with pytest.raises(TextReviewFormatError):
            parse(report, evidence, question, audited=False)


@pytest.mark.parametrize("case, response_number, expected", [
    ("Q004", "0007", None), ("Q004", "0010", (4, 1)),
    ("Q019", "0007", (3, 1)), ("Q019", "0010", (4, 1)),
    ("Q003", "0007", (1, 1)), ("Q003", "0010", (1, 0)),
])
def test_replay_preserved_low_question_pilot_reports_offline(case, response_number, expected):
    # Archived reports are intentionally outside source control. Portable
    # regressions above cover these same structures when the archive is absent.
    case_path = Path(__file__).resolve().parents[1] / "reports/question_generation/20261022T021104Z_low_question_revision_pilot/cases" / case
    job_path = case_path / "revision_job.json"
    if not job_path.exists():
        pytest.skip("preserved pilot archive is not present in this checkout")
    response_paths = list((case_path / "workspace/logs").glob(f"*/{response_number}_question_model_text_response.json"))
    assert len(response_paths) == 1
    job = json.loads(job_path.read_text(encoding="utf-8"))
    response = json.loads(response_paths[0].read_text(encoding="utf-8"))
    assert response["name"] == "question_full_review"
    if expected is None:
        # The first Q004 report genuinely omitted a clinical decision. The new
        # structural tolerances must not synthesize a distinctness judgment.
        with pytest.raises(TextReviewFormatError, match="重复判断"):
            parse(response["text"], job["evidence"], job["questions"][0]["question"])
    else:
        value = parse(response["text"], job["evidence"], job["questions"][0]["question"])
        assert value["verdict"] == "revise"
        assert (len(value["issues"]), len(value["evidence_gaps"])) == expected
        assert all(issue["evidence_ids"] == [] for issue in value["issues"])
        assert validate_audited_review(value, job["evidence"], job["questions"][0]["question"]) == value


def test_every_actual_criterion_and_each_cited_source_is_required(evidence, question):
    for report in (AUDITED.replace(CHECK2, ""), AUDITED.replace("证据核查：R2", "证据核查：R1"),
                   AUDITED.replace("核查来源：L1", "核查来源：W1"),
                   AUDITED.replace("核查来源：L1", "核查来源：unknown")):
        with pytest.raises(TextReviewFormatError):
            parse(report, evidence, question)
    question["scoring_points"][0]["evidence_ids"].append("W2")
    with pytest.raises(TextReviewFormatError):
        parse(AUDITED, evidence, question)
    extra = "核查来源：W2\n来源支持：充分支持\n原文：呼吸支持期间继续检查接口接触组织\n"
    value = parse(AUDITED.replace(CHECK1, CHECK1 + extra), evidence, question)
    assert len(value["quality_audit"]["evidence_support"][0]["sources"]) == 2


def test_repeated_same_source_does_not_replace_missing_other_source(evidence, question):
    duplicate = "核查来源：L1\n来源支持：充分支持\n原文：佩戴前记录器械接触部位的组织基线\n"
    with pytest.raises(TextReviewFormatError):
        parse(AUDITED.replace(CHECK1, CHECK1 + duplicate), evidence, question)


@pytest.mark.parametrize("quote", ["佩戴前应评估皮肤状况", "佩戴前记录……组织基线", "佩戴前记录组织基线", "无"])
def test_fabricated_paraphrased_or_spliced_quotes_are_not_accepted(quote, evidence, question):
    with pytest.raises(TextReviewFormatError):
        parse(AUDITED.replace("原文：佩戴前记录器械接触部位的组织基线", "原文：" + quote), evidence, question)


def test_whitespace_matching_returns_an_actual_saved_contiguous_span(evidence, question):
    original = "佩戴前记录器械接触\n  部位的组织基线"
    evidence["sources"][0]["text"] = "前言。" + original + "，评估接口是否适配。"
    result = parse(AUDITED, evidence, question)
    quote = result["quality_audit"]["evidence_support"][0]["sources"][0]["quote"]
    assert quote == original and quote in evidence["sources"][0]["text"]


def test_pdf_page_ornament_is_retained_in_the_returned_actual_span(evidence, question):
    original = "无创通气设备的\n· · 1602 选择应根据临床情况确定"
    evidence["sources"][0]["text"] = "前言。" + original + "。后文。"
    report = AUDITED.replace("佩戴前记录器械接触部位的组织基线", "无创通气设备的选择应根据临床情况确定")
    result = parse(report, evidence, question)
    quote = result["quality_audit"]["evidence_support"][0]["sources"][0]["quote"]
    assert quote == original
    assert validate_audited_review(result, evidence, question) == result


@pytest.mark.parametrize("pair", [("“", "”"), ('"', '"'), ("‘", "’"), ("'", "'"), ("`", "`")])
def test_one_external_pair_of_quote_marks_can_wrap_a_real_excerpt(pair, evidence, question):
    actual = "佩戴前记录器械接触部位的组织基线"
    report = AUDITED.replace("原文：" + actual, "原文：" + pair[0] + actual + pair[1])
    result = parse(report, evidence, question)
    assert result["quality_audit"]["evidence_support"][0]["sources"][0]["quote"] == actual


def test_actual_source_quote_marks_are_preserved_before_trying_external_wrappers(evidence, question):
    actual = "“佩戴前记录器械接触部位的组织基线”"
    evidence["sources"][0]["text"] = actual + "，评估接口是否适配。"
    result = parse(AUDITED.replace("原文：佩戴前记录器械接触部位的组织基线", "原文：" + actual), evidence, question)
    assert result["quality_audit"]["evidence_support"][0]["sources"][0]["quote"] == actual


@pytest.mark.parametrize("quote", ["““佩戴前记录器械接触部位的组织基线””", "佩戴前记录‘器械’接触部位的组织基线", "“佩戴前记录……组织基线”"])
def test_quote_wrapper_compatibility_does_not_strip_internal_or_nested_quotes(quote, evidence, question):
    with pytest.raises(TextReviewFormatError):
        parse(AUDITED.replace("原文：佩戴前记录器械接触部位的组织基线", "原文：" + quote), evidence, question)


@pytest.mark.parametrize("fragment", [" 1602 ", "\n1602 ", "\n· 1602 ", "\n- 1602 ", " · · 1602 ", "\n· · 1602mg "])
def test_clinical_numbers_and_non_page_fragments_are_never_removed(fragment, evidence, question):
    evidence["sources"][0]["text"] = "无创通气设备的" + fragment + "选择应根据临床情况确定"
    report = AUDITED.replace("佩戴前记录器械接触部位的组织基线", "无创通气设备的选择应根据临床情况确定")
    with pytest.raises(TextReviewFormatError):
        parse(report, evidence, question)


def test_supported_empty_quote_is_rejected_even_if_criterion_is_supported(evidence, question):
    with pytest.raises(TextReviewFormatError):
        parse(AUDITED.replace("原文：佩戴前记录器械接触部位的组织基线", "原文：无"), evidence, question)


def test_original_schema_array_limits_remain_authoritative(evidence, question):
    issues = "\n".join(ISSUE.replace("问题：1", f"问题：{index}") for index in range(1, 10))
    gaps = "\n".join(GAP.replace("补证：1", f"补证：{index}") for index in range(1, 5))
    for report in (BASE.replace("通过", "需修订").replace("问题：无", issues),
                   BASE.replace("通过", "需补证").replace("补证：无", gaps)):
        with pytest.raises(TextReviewFormatError):
            parse(report, evidence, question, audited=False)


def test_seventeen_checked_sources_exceed_the_existing_source_limit(evidence, question):
    identities = [f"S{index}" for index in range(17)]
    evidence["sources"] = [{"evidence_id": identity, "text": "同一原文。"} for identity in identities]
    question["scoring_points"] = [{"id": "R1", "evidence_ids": identities}]
    sources = "".join(f"核查来源：{identity}\n来源支持：充分支持\n原文：同一原文\n" for identity in identities)
    check = "证据核查：R1\n评分点支持：充分支持\n评分点理由：逐条核查。\n" + sources
    report = AUDITED.replace(CHECK1 + CHECK2, check)
    with pytest.raises(TextReviewFormatError):
        parse(report, evidence, question)


def test_blind_review_only_needs_its_own_explicit_prose():
    report = "可答性：不可答\n独立试答：无\n歧义：处理对象未限定。\n歧义：治疗条件未给出。"
    value = parse_review_text(report, evidence=None, question=None, audited=True, blind=True)
    assert value == {"answerable": False, "attempted_answer": "", "ambiguity_notes": ["处理对象未限定。", "治疗条件未给出。"]}


@pytest.mark.parametrize("report", ["独立试答：无\n歧义：无", "可答性：可答\n独立试答：已作答",
                                    "可答性：可答\n独立试答：已作答\n歧义：无\n歧义：有两种理解。"])
def test_blind_missing_or_contradictory_fields_fail(report):
    with pytest.raises(TextReviewFormatError):
        parse_review_text(report, evidence=None, question=None, audited=False, blind=True)


def test_legacy_complete_valid_objects_remain_compatible(evidence, question):
    value = parse(AUDITED, evidence, question)
    assert parse(json.dumps(value, ensure_ascii=False), evidence, question) == value
    assert parse("```json\n" + json.dumps(value) + "\n```", evidence, question) == value


def test_legacy_json_does_not_receive_defaults_or_allow_duplicate_keys(evidence, question):
    value = parse(AUDITED, evidence, question)
    missing = deepcopy(value)
    del missing["quality_audit"]["case_facts"]["status"]
    for content in (json.dumps(missing), '{"verdict":"reject","verdict":"accept"}', '{"verdict":"accept"',
                    json.dumps({**value, "quality_audit": []}),
                    json.dumps({**value, "quality_audit": {**value["quality_audit"], "evidence_support": [{"criterion_id": {}}]}})):
        with pytest.raises(TextReviewFormatError):
            parse(content, evidence, question)


def test_errors_do_not_echo_source_body_or_invented_quote(evidence, question):
    secret_body = "PRIVATE_SOURCE_TEXT_患者资料不应显示在格式错误中"
    value = parse(AUDITED, evidence, question)
    value["quality_audit"]["evidence_support"][0]["sources"][0]["quote"] = secret_body
    with pytest.raises(TextReviewFormatError) as caught:
        parse(json.dumps(value), evidence, question)
    assert secret_body not in str(caught.value)
    assert len(str(caught.value)) < 180 and "原文" in str(caught.value)


@pytest.mark.parametrize("content", [None, "", "   ", [], "这道题看起来没有问题，可以通过。"])
def test_empty_or_unstructured_reports_do_not_imply_acceptance(content, evidence, question):
    with pytest.raises(TextReviewFormatError):
        parse(content, evidence, question)
