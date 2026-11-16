"""Turn question-bound model judgements into a quote-backed learning record."""
import re
import unicodedata

from .language import localized
from .learner_evidence import checked_quotes, is_acknowledgement, quote_catalog

STATUS_LABELS = {
    "met": ("已覆盖", "Covered"), "partial": ("部分覆盖", "Partly covered"),
    "missing": ("暂未找到对应作答", "Not yet found in your answer"),
    "incorrect": ("需要纠正", "Needs correction"), "uncertain": ("待确认", "Needs clarification"),
}


def choice_question_type(item):
    kind = item.get("question_type")
    if kind in {"single_choice", "multiple_choice"}:
        return kind
    # Legacy multiple-choice case snapshots may lack the newer type field.
    # Missing answer metadata must never send such a question to a model.
    return "single_choice" if item.get("kind") == "case_mcq" else None


def choice_invitation(kind, language):
    return (localized(language, "请选择一个选项，或输入对应的选项字母。", "Select one option or enter its letter.")
            if kind == "single_choice" else
            localized(language, "请选择所有正确选项，或输入对应的选项字母。", "Select all correct options or enter their letters."))


def choice_question_text(text):
    """Remove the obsolete optional-writing invitation only when displaying choices."""
    return text.replace("；理由可补充。", "。")


OBJECTIVE_TYPES = {"single_choice", "multiple_choice", "true_false", "ordering"}


def objective_question_type(item):
    kind = item.get("question_type")
    if kind in OBJECTIVE_TYPES:
        return kind
    return {"case_mcq": "single_choice", "case_sequence": "ordering"}.get(item.get("kind"))


def objective_invitation(kind, language):
    if kind == "true_false":
        return localized(language, "请选择“正确”或“错误”，然后提交答案。", "Select True or False, then submit your answer.")
    if kind == "ordering":
        return localized(language, "请调整各项顺序，然后提交答案。", "Arrange all options, then submit your answer.")
    return (localized(language, "请选择一个选项，然后提交答案。", "Select one option, then submit your answer.")
            if kind == "single_choice" else
            localized(language, "请选择所有正确选项，然后提交答案。", "Select all correct options, then submit your answer."))


def objective_rubric(item, rubric):
    """One closed result records accuracy, never unsubmitted explanations."""
    kind = objective_question_type(item)
    if kind not in {"true_false", "ordering"}:
        return rubric
    label = {"true_false": {"zh": "判断正确", "en": "Correct true/false judgement"},
             "ordering": {"zh": "排序正确", "en": "Correct ordering"}}[kind]
    return {"version": "objective_rules_20261013_v1", "origin": "published_objective_answer_rule",
            "review_status": (rubric or {}).get("review_status", item.get("review_status", "")),
            "required": [{"id": "O1", "criterion": label,
                          "acceptable": {"zh": "提交的判断或顺序与本题发布答案一致。", "en": "The submitted judgement or order matches the published key."},
                          "incorrect": {"zh": "提交的判断或顺序与本题发布答案不一致。", "en": "The submitted judgement or order differs from the published key."},
                          "follow_up": {language: objective_invitation(kind, language) for language in ("zh", "en")}}],
            "optional_extensions": {}, "basis_question": item.get("case"), "basis_answer": item.get("answer", {}).get("en")}


def objective_question_text(text):
    """Presentation-only removal of obsolete explanation requirements."""
    return (choice_question_text(text).replace("；理由可补充", "")
            .replace("（判断改错）", "（判断题）")
            .replace("请判断正确或错误并说明理由；错误时请给出更正。", "请选择正确或错误。")
            .replace("可写字母顺序，并说明安全条件或依赖关系。", "请调整各项顺序。"))


def validate_objective_rule(item, language="zh"):
    """Reject missing or changed published keys, including cached reviews."""
    kind = objective_question_type(item)
    if kind not in {"true_false", "ordering"}:
        return
    unavailable = localized(language, "本题标准答案规则不可用，请重新选择题目。",
                            "This question's answer rule is unavailable. Choose another question.")
    if "objective_rule_source" not in item:
        # An old ordering snapshot has an explicit, complete ordered key.
        # A new truth/order key must retain its published source binding.
        if kind == "ordering" and "correct_order" not in item:
            return
        raise ValueError(unavailable)
    from .bank_expansion import load_objective_rules, objective_rule_fingerprint
    data = load_objective_rules()
    stamp = item["objective_rule_source"]
    record = data["items"].get(item.get("id"))
    if (not isinstance(stamp, dict) or not isinstance(record, dict)
            or stamp.get("version") != data.get("version")
            or record.get("question_type") != kind
            or not stamp.get("source_sha256")
            or stamp["source_sha256"] != record.get("source_sha256")
            or objective_rule_fingerprint(item) != stamp["source_sha256"]):
        raise ValueError(unavailable)
    key = "correct_truth" if kind == "true_false" else "correct_order"
    actual, expected = item.get(key), record.get(key)
    valid = (type(actual) is bool and type(expected) is bool if kind == "true_false" else
             isinstance(actual, list) and isinstance(expected, list)
             and all(isinstance(label, str) for label in actual))
    if not valid or actual != expected:
        raise ValueError(unavailable)


def objective_assessment(item, rubric, answers, language):
    kind = objective_question_type(item)
    if not kind:
        return None
    validate_objective_rule(item, language)
    if kind in {"single_choice", "multiple_choice"}:
        return choice_assessment(item, rubric, answers, language)
    if not answers:
        raise ValueError(localized(language, "请先提交答案。", "Submit your answer first."))
    options = item.get("options", {})
    expected = item.get("correct_truth") if kind == "true_false" else item.get("correct_order", item.get("correct_options"))
    invalid = (not rubric or len(rubric.get("required", [])) != 1
               or kind == "true_false" and not isinstance(expected, bool)
               or kind == "ordering" and (not isinstance(expected, list) or not expected or not options
                   or len(expected) != len(options) or len(set(expected)) != len(expected)
                   or set(expected) != set(options) or any(not re.fullmatch(r"[A-Z]", key) for key in options)))
    if invalid:
        raise ValueError(localized(language, "本题标准答案规则不可用，请重新选择题目。", "This question's answer rule is unavailable. Choose another question."))
    latest = unicodedata.normalize("NFKC", answers[-1]).strip()
    selected = None
    if kind == "true_false":
        token = re.sub(r"^(?:(?:我(?:的)?)?(?:判断|答案)(?:是|为)?\s*[:：]?\s*|(?:my\s+)?(?:judgement|judgment|answer)(?:\s+is)?\s*[:：]?\s*)", "", latest, flags=re.I)
        token = token.strip().rstrip("。.!！").strip().casefold()
        if token in {"正确", "对", "是", "true", "t", "yes"}:
            selected = True
        elif token in {"错误", "错", "否", "false", "f", "no"}:
            selected = False
    else:
        token = re.sub(r"^(?:(?:我(?:的)?)?(?:顺序|排序|答案)(?:是|为)?\s*[:：]?\s*|(?:my\s+)?(?:order|sequence|answer)(?:\s+is)?\s*[:：]?\s*)", "", latest, flags=re.I)
        token = token.strip().rstrip("。.!！").strip()
        if re.fullmatch(r"[A-Za-z](?:\s*(?:→|->|>|、|,|，|/|\s)\s*[A-Za-z])*", token) or re.fullmatch(r"[A-Z]{2,}", token):
            selected = re.findall(r"[A-Za-z]", token.upper())
            if len(selected) != len(options) or len(set(selected)) != len(selected) or set(selected) != set(options):
                selected = None
    status = "uncertain" if selected is None else "met" if selected == expected else "incorrect"
    note = (objective_invitation(kind, language) if selected is None else
            localized(language, "回答正确。", "Correct.") if status == "met" else
            localized(language, "回答错误。", "Incorrect."))
    evidence = [{"quote_id": row["id"]} for row in quote_catalog(answers) if row["attempt"] == len(answers)] if selected is not None else []
    return {"criteria": {rubric["required"][0]["id"]: {"status": status, "evidence": evidence, "explanation": note,
                "decision_origin": "authored_objective_rule", "selected_truth": selected if kind == "true_false" else None,
                "selected_order": selected if kind == "ordering" else None}},
            "reflection_explained": False, "assessment_method": "authored_objective_rules_v1", "objective_question_type": kind}


def choice_assessment(item, rubric, answers, language):
    """Evaluate a closed option set using its authored key, never a model guess.

    Single/multiple choices never fall back to semantic model grading. Accept
    explicit letters and exact option wording; ambiguous input needs selection.
    Open answers and tasks with explanation requirements keep semantic grading.
    """
    kind = choice_question_type(item)
    options, expected = item.get("options", {}), item.get("correct_options", [])
    if not kind:
        return None
    if (not rubric or len(rubric.get("required", [])) != 1 or not options or not expected
            or not set(expected).issubset(options)
            or any(not re.fullmatch(r"[A-Z]", key) for key in options)
            or kind == "single_choice" and len(set(expected)) != 1):
        raise ValueError(localized(language, "本题选项或标准答案配置异常，暂时无法判分，请重新选择题目。",
                                   "This question's options or answer key are unavailable. Choose another question."))
    if not answers:
        raise ValueError(localized(language, "请先提交选项。", "Submit your selection first."))
    selected, selection_clause = None, None
    catalog = quote_catalog(answers)
    # A later explicit selection replaces the former set. A prose-only reply
    # cannot quietly restore an older choice or discard a newer correction.
    latest = unicodedata.normalize("NFKC", answers[-1]).strip()
    prefix = (r"(?:(?:我(?:的)?)?(?:重新选择|改选|选择|选|答案|选项)(?:是|为)?\s*[:：]?\s*|"
               r"(?:my\s+)?(?:answers?|choices?)\b(?:\s+is|\s+are)?\s*[:：]?\s*|"
               r"(?:i\s+)?(?:choose|select)\b\s*[:：]?\s*)")
    prefixed = re.match(r"^" + prefix, latest, re.I)
    selection_text = latest[prefixed.end():] if prefixed else latest
    def option_text(value):
        return unicodedata.normalize("NFKC", value).strip().rstrip("。.!！").strip().casefold()
    labels = {}
    for table in (options, item.get("options_en", {})):
        for key, label in table.items():
            if key in options and isinstance(label, str):
                labels.setdefault(option_text(label), set()).add(key)
    wording = labels.get(option_text(selection_text), set())
    if len(wording) == 1:
        selected = set(wording)
    # Each option must be a whole letter token. Requiring its right boundary
    # avoids reading the first letter of a prose word such as "because".
    letter = r"[A-Za-z](?![A-Za-z])"
    separator = r"(?:\s*[、,，/&+]\s*|\s+(?:and\s+)?)"
    match = re.match(r"(" + letter + r"(?:" + separator + letter + r")*)", selection_text, re.I)
    if selected is None and not match:
        # Compact letters (ABD) are a common typed multiple-choice input.
        match = re.match(r"([A-Z]{2,})(?![A-Za-z])", selection_text)
    if selected is None and match:
        token = match.group(1)
        selected = set(re.findall(r"[A-Za-z]", re.sub(r"\band\b", "", token, flags=re.I).upper()))
        tail = selection_text[match.end():]
        # Bare English prose ("I would ...", "A pressure injury ...") is a
        # semantic answer, not a letter selection. Explicit selections and
        # punctuation-delimited UI input may have a separate optional reason.
        alternative = re.match(r"\s*(?:或|或者|or\b)", tail, re.I)
        reason_start = re.match(r"^\s*[,，:：]?\s*(?:理由|原因|因为|reason\b|because\b|since\b)", tail, re.I)
        punctuation = re.match(r"^\s*[。.!！;；:：\n]", tail)
        dangling_separator = re.match(r"^\s*[、,，/&+]", tail) and not reason_start
        if (not alternative and not dangling_separator and tail.strip() and not punctuation and not reason_start):
            selected = None
        # Alternatives are not a definite selection. A second selection phrase
        # needs clarification rather than invented union/difference semantics.
        if alternative or dangling_separator or re.match(
                r"[。.!！;；,，\s]*(?:不是|不选|再选|也选|改选|还(?:选择|选|包括|有)|加上|另外(?:选择|选)?|"
                r"(?:我(?:的)?)?(?:选择|答案)(?:是|为)?[:：]?\s*|"
                r"(?:i\s+)?(?:also(?:\s+(?:choose|select|include|add))?|not|plus|include|add|choose|select|answer is)\s+)[A-Za-z]", tail, re.I):
            selected = None
        # A later correction embedded in prose is ambiguous; do not guess which
        # selection should win or silently pass an earlier correct letter.
        if re.search(r"(?:改选|更正(?:为|成)?|重新选择|不选|不是|答案改为)\s*[:：]?\s*[A-Za-z]|"
                     r"\b(?:instead|not|change(?:\s+(?:it|my\s+answer))?\s+to|correct(?:ion)?\s*:)\s+[A-Za-z]\b", tail, re.I):
            selected = None
        remainder = re.sub(r"^[。.!！;；:：,，\s]+", "", tail)
        if (re.match(r"[A-Za-z](?![A-Za-z])|[A-Z]{2,}(?![A-Za-z])", remainder)
                or re.search(prefix + r"[A-Za-z](?![A-Za-z])", tail, re.I)):
            selected = None
    selection_clause = [row for row in catalog if row["attempt"] == len(answers)]
    if selected is not None and (not selected or not selected.issubset(options)):
        selected = None
    if kind == "single_choice" and selected is not None and len(selected) != 1:
        selected = None
    if selected is None:
        status, evidence = "uncertain", []
        explanation = choice_invitation(kind, language)
    else:
        expected_set = set(expected)
        extras = selected - expected_set
        status = "incorrect" if extras else "met" if selected == expected_set else "partial"
        evidence = [{"quote_id": row["id"]} for row in selection_clause]
        if status == "met":
            explanation = localized(language, "回答正确。", "Correct.")
        elif kind == "single_choice":
            explanation = localized(language, "回答错误。", "Incorrect.")
        elif extras:
            explanation = localized(language, "回答错误，存在错选，请重新选择。", "Incorrect: one or more selected options are wrong. Select again.")
        else:
            explanation = localized(language, "回答错误，选项不完整，请重新选择。", "Incorrect: the selection is incomplete. Select again.")
    return {"criteria": {rubric["required"][0]["id"]: {
        "status": status, "evidence": evidence, "explanation": explanation,
        "decision_origin": "authored_choice_set", "selected_options": sorted(selected) if selected is not None else None}},
        "reflection_explained": False, "assessment_method": "authored_option_set_v2",
        "objective_question_type": kind}


def assessment_note(result, criterion, language):
    """Visible notes use authored requirements and actual quotes, not model claims."""
    if result.get("decision_origin") in {"authored_choice_set", "authored_objective_rule"}:
        return result["explanation"]
    status = result["status"]
    if status == "met":
        return localized(language, "本项由下方所引的学生原话支持。", "This point is supported by the learner words quoted below.")
    if status in {"partial", "missing"}:
        remaining = result.get("remaining_requirement", "")
        requirement = criterion["criterion"][language]
        if not remaining or remaining not in requirement:
            remaining = requirement
        return localized(language, "本项尚待补充的必答要求：", "The required point still to address: ") + remaining
    if status == "incorrect":
        claim, rule = result.get("error_quote"), result.get("conflict_rule_quote")
        if claim and rule:
            return localized(language, "请对照实际原话与本项规则：", "Compare your actual words with this criterion's rule: ") + f"\n> {claim}\n\n> {rule}"
        return localized(language, "请对照所引原话与本项题目规则修正。", "Compare the quoted words with the rule for this criterion and correct the conflict.")
    return localized(language, "本项含义或判断依据尚未确认，可补充说明或重新核对已有作答。",
                     "This point or its assessment basis remains unconfirmed. Clarify it or recheck your existing answer.")


def ground_review(raw, rubric, answers, previous_reviews, language):
    """Check quote provenance, not clinical meaning; no keyword-based marking."""
    previous = {row["id"]: row for row in (previous_reviews[-1].get("criterion_checks", []) if previous_reviews else [])}
    checks, warnings = [], []
    objective = raw.get("objective_question_type")
    only_acknowledgements = not objective and bool(answers) and all(is_acknowledgement(answer) for answer in answers)
    for criterion in rubric["required"]:
        result = raw["criteria"][criterion["id"]]
        if objective and result.get("decision_origin") in {"authored_choice_set", "authored_objective_rule"}:
            # In a truth question "yes"/"对" are actual selections. The open
            # answer acknowledgement filter must not erase this rule evidence.
            originals = {(row["attempt"], row["text"]) for row in quote_catalog(answers)}
            evidence = [row for row in result["evidence"] if (row.get("attempt"), row.get("quote")) in originals]
            rejected = [row for row in result["evidence"] if row not in evidence]
        else:
            evidence, rejected = checked_quotes(result["evidence"], answers)
        warnings.extend({"criterion": criterion["id"], "rejected_evidence": cited} for cited in rejected)
        status, explanation = result["status"], assessment_note(result, criterion, language)
        if status in {"met", "partial", "incorrect"} and not result["evidence"]:
            warnings.append({"criterion": criterion["id"], "reason": "missing_learner_quote", "model_status": status})
        if len(evidence) != len(result["evidence"]) or (status in {"met", "partial", "incorrect"} and not evidence):
            status = "uncertain"
            explanation = localized(language, "本次判断未能定位到学生原话，保留为待确认。", "This judgement could not be linked to the learner's original words and remains unconfirmed.")
        if only_acknowledgements:
            # This is a provenance decision, not a clinical answer-key check.
            status, evidence = "missing", []
            explanation = localized(language, "目前只有确认或同意，还没有本题要点的作答；教师提示不计作你的答案。",
                "So far you have only acknowledged or agreed; no answer to this point has been supplied. A tutor hint does not count as your answer.")
        before = previous.get(criterion["id"])
        previous_count = previous_reviews[-1].get("answer_count", 0) if previous_reviews else 0
        improved = bool(before and len(answers) > previous_count and before["status"] != "met" and status == "met"
                        and any(e["attempt"] == len(answers) and not any(e["quote"] in old for old in answers[:-1])
                                for e in evidence))
        check = {"id": criterion["id"], "criterion": criterion["criterion"], "status": status,
                 "evidence": evidence, "explanation": explanation, "newly_demonstrated": improved,
                 "decision_origin": "acknowledgement_only" if only_acknowledgements else result.get("decision_origin", "model")}
        # Keep the visible rule and actual conflicting words when a later
        # supplement retains a previous error. Their provenance stays local.
        for key in ("decision_basis", "remaining_requirement", "selected_options", "selected_truth", "selected_order"):
            if key in result:
                check[key] = result[key]
        if status == "incorrect":
            for key in ("error_quote", "conflict_rule_quote"):
                if result.get(key):
                    check[key] = result[key]
        checks.append(check)
    ready = all(row["status"] == "met" for row in checks)
    focus = next((row["id"] for status in ("incorrect", "uncertain", "partial", "missing") for row in checks if row["status"] == status), "none")
    objective = raw.get("objective_question_type")
    follow_up = (localized(language, "回答正确。", "Correct.") if ready else objective_invitation(objective, language)) if objective else (
                 localized(language, "本题必答要点已覆盖，可以结束并总结。", "The required points for this question are covered; we can summarize.") if ready else
                 next(row["follow_up"][language] for row in rubric["required"] if row["id"] == focus))
    return {"grading_mode": "question_rubric", "rubric_version": rubric["version"],
            "assessment_method": raw.get("assessment_method", "model_semantic"), "objective_question_type": objective,
            "criterion_checks": checks, "evidence_warnings": warnings,
            "status_contract_warnings": raw.get("status_contract_warnings", []),
            "verification_unavailable": raw.get("verification_unavailable", []),
            "optional_extensions": {} if objective else rubric["optional_extensions"], "language": language,
            "strengths": [row["criterion"][language] for row in checks if row["status"] == "met"],
            "improvements": [row["criterion"][language] for row in checks if row["newly_demonstrated"]],
            "gaps": [row["criterion"][language] for row in checks if row["status"] in {"partial", "missing", "uncertain"}],
            "material_errors": [row["criterion"][language] for row in checks if row["status"] == "incorrect"],
            "ready_for_summary": ready, "reflection_explained": raw["reflection_explained"],
            "focus": focus, "feedback": follow_up}


def assessment_text(review, language):
    checks = review["criterion_checks"]
    objective = review.get("objective_question_type")
    if objective:
        if review["ready_for_summary"] and not review["material_errors"]:
            return localized(language, "回答正确。", "Correct.")
        statuses = {row["status"] for row in checks}
        if "uncertain" in statuses or "missing" in statuses:
            return choice_invitation(objective, language) if objective in {"single_choice", "multiple_choice"} else objective_invitation(objective, language)
        if objective in {"single_choice", "true_false", "ordering"}:
            return localized(language, "回答错误。", "Incorrect.")
        return (localized(language, "回答错误，存在错选，请重新选择。", "Incorrect: one or more selected options are wrong. Select again.")
                if "incorrect" in statuses else
                localized(language, "回答错误，选项不完整，请重新选择。", "Incorrect: the selection is incomplete. Select again."))
    covered = sum(row["status"] == "met" for row in checks)
    parts = [localized(language, "**本题逐项核对**", "**Check against this question**"),
             localized(language, f"必答要点：已覆盖 {covered}/{len(checks)}。可选拓展不影响本题完成。",
                       f"Required points covered: {covered}/{len(checks)}. Optional extensions do not affect completion.")]
    for row in checks:
        label = localized(language, *STATUS_LABELS[row["status"]])
        part = f"**{row['criterion'][language]} — {label}**"
        if row["evidence"]:
            for evidence in row["evidence"]:
                source = localized(language, f"第{evidence['attempt']}次作答原话：", f"Your words in answer {evidence['attempt']}: ")
                part += "\n\n" + source + "\n> " + evidence["quote"].replace("\n", "\n> ")
        else:
            part += "\n\n" + localized(language, "尚无可引用的学生原话；教师的提示不计作你的答案。", "No learner quotation is available; a tutor hint does not count as your answer.")
        # On language changes keep original learner quotations, never translate
        # them into fictitious original words. The static criterion is bilingual.
        if row["explanation"] and review.get("language") == language:
            part += "\n\n" + localized(language, "核对说明：", "Assessment note: ") + row["explanation"]
        if row["newly_demonstrated"]:
            part += "\n\n" + localized(language, "本次补充后覆盖了这一要点。", "This point is now covered by your supplement.")
        if row.get("judgment_revised"):
            part += "\n\n" + localized(language, "重新核对后调整了本项判断，本次没有新增作答。", "This judgement changed after rechecking; no new answer was submitted.")
        if row.get("answer_context") == "after_hint":
            part += "\n\n" + localized(language, "这段作答在查看对应提示后提交。", "This answer was submitted after viewing a hint for this point.")
        elif row.get("answer_context") == "after_reference":
            part += "\n\n" + localized(language, "这段作答在查看参考答案后提交。", "This answer was submitted after viewing the reference answer.")
        elif row.get("answer_context") == "after_explanation":
            part += "\n\n" + localized(language, "这段作答在教学讲解后提交；讲解不占提示次数。", "This answer was submitted after a teaching explanation, which does not use a hint opportunity.")
        parts.append(part)
    if review.get("verification_unavailable"):
        parts.append(localized(language, "额外核对未完成，已保留本次初评与实际作答；可以重新核对已有答案。",
                               "The additional check did not finish. The initial assessment and your answer are retained; you can recheck the existing answer."))
    parts.append(localized(language, "**可选拓展（不计入缺漏）**\n\n", "**Optional extensions (not missing requirements)**\n\n") + review["optional_extensions"][language])
    return "\n\n".join(parts)
